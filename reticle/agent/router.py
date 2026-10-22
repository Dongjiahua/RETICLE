"""Expert router: the agent's VLM decides which RETICLE expert, if any, should measure an image, from the image, the request and each
expert's one-line card. The reply is constrained to the expert names by a JSON schema; no reasoning step.
  route(port, image, request) -> 'metastasis' | 'mitosis' | 'polyp' | 'none'
Evaluation on a sample of test items: the right expert is the one each benchmark uses, and none for PathMMU and MultiCaRe-Dx. In the
'generic' condition the region benchmarks' questions, which name the tissue and the task, are replaced by a request that names neither
(as for an upload in the demo), so the choice rests on the image.
  python -m reticle.agent.router eval --port 8930 --per_set 50 --out $RETICLE_DATA/runs/router_v1"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random

from PIL import Image

from reticle.agent.consult import post, to_b64
from reticle.agent.vlm_qa import read_jsonl
from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
EXPERTS = {
    'metastasis': 'lymph-node metastasis expert: finds tumour deposits in an H&E region of a lymph node and measures the largest one '
                  '(none, isolated tumour cells, micrometastasis or macrometastasis)',
    'mitosis': 'mitosis expert: counts mitotic figures in an H&E field of invasive breast carcinoma',
    'polyp': 'colorectal polyp expert: names the polyp type and dysplasia grade of an H&E region of colorectal mucosa or a colorectal polyp',
    'none': 'no expert: the image or the request is outside these experts (other organs or stains, general questions, diagnosis or '
            'treatment of a whole case)'}
PROMPT = ('You route a pathology request to the measuring expert that should read the image before you answer. Look at the image '
          'and the request, and choose one option.\n'+'\n'.join(f'- {k}: {v}' for k, v in EXPERTS.items())+
          '\n\nRequest: {request}\n\nReturn JSON: {{"expert": "<option>"}}')
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['expert'], 'properties': {'expert': {'enum': list(EXPERTS)}}}
GENERIC = 'Read this H&E image with the expert that fits it, if any.'   # an upload without a question: the image decides
# benchmark: (queries, expert it uses); the region benchmarks' experts are fixed per benchmark in reticle.agent.region_qa
SETS = {'c16': ('runs/camelyon16_ms_v1/inputs/queries_c16_test.jsonl', 'metastasis'), 'mdg': ('runs/midog_ms_v1/inputs/queries_mdg_test.jsonl', 'mitosis'),
        'u8': ('runs/unitopatho_ms_v1/inputs/queries_u8_test.jsonl', 'polyp'), 'u7': ('runs/unitopatho_ms_v1/inputs/queries_u7_test.jsonl', 'polyp'),
        'pmmu': ('runs/pathmmu_v3/inputs/queries_test.jsonl', 'none'), 'mcdx': ('runs/mcdx_v1/inputs/items.jsonl', 'none')}


PARAMS = dict(temperature=0.0, max_tokens=20, response_format=dict(type='json_schema', json_schema=dict(name='route', schema=SCHEMA)))
CHAT_KW = dict(enable_thinking=False)


def conversation(image, request):
    """The routing request in the conversation format of reticle.agent.region_qa (image as PIL)."""
    return [dict(role='user', content=[dict(type='image_pil', image_pil=image), dict(type='text', text=PROMPT.format(request=request))])]


def parse(text):
    choice = json.loads(text.rsplit('</think>', 1)[-1].strip())['expert']
    assert choice in EXPERTS, choice
    return choice


def route(port, image, request):
    """The expert the VLM chooses for this image and request ('none' if no expert applies)."""
    body = dict(model='qwen', chat_template_kwargs=CHAT_KW, **PARAMS,
                messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url='data:image/png;base64,'+to_b64(image))),
                                                     dict(type='text', text=PROMPT.format(request=request))])])
    return parse(post(port, body)['choices'][0]['message']['content'])


def request_of(s, q):
    if s == 'mcdx':   # the first consultation turn
        return f"Case: {q['presentation']}\nQuestion: What is the most likely diagnosis?"
    options = q['options'] if isinstance(q['options'], list) else json.loads(q['options'].replace("'", '"'))
    return q['question']+'\n'+'\n'.join(options)


def evaluate(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    todo = []
    for s, (path, want) in SETS.items():
        qs = read_jsonl(DATA/path); random.Random(a.seed).shuffle(qs)
        for q in qs[:a.per_set]:
            todo.append(dict(set=s, id=q['id'], want=want, condition='question', image_path=q['image_path'], request=request_of(s, q)))
            if want != 'none':
                todo.append(dict(set=s, id=q['id'], want=want, condition='generic', image_path=q['image_path'], request=GENERIC))

    def one(t):
        return dict(t, got=route(a.port, Image.open(t['image_path']).convert('RGB'), t['request']), request=t['request'][:200])
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, todo))
    (out/'routes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary = {}
    for cond in ('question', 'generic'):
        for s in SETS:
            rs = [r for r in rows if r['condition'] == cond and r['set'] == s]
            if rs:
                summary[f'{cond}/{s}'] = dict(n=len(rs), correct=sum(r['got'] == r['want'] for r in rs), chosen=dict(Counter(r['got'] for r in rs)))
        rs = [r for r in rows if r['condition'] == cond]
        summary[f'{cond}/all'] = dict(n=len(rs), correct=sum(r['got'] == r['want'] for r in rs))
    (out/'summary.json').write_text(json.dumps(dict(per_set=a.per_set, seed=a.seed, experts=EXPERTS, results=summary), indent=2)+'\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('eval'); e.add_argument('--port', type=int, default=8930); e.add_argument('--per_set', type=int, default=50)
    e.add_argument('--seed', type=int, default=0); e.add_argument('--workers', type=int, default=16); e.add_argument('--out', required=True)
    a = p.parse_args()
    evaluate(a)
