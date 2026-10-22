"""Open pathology VLM baselines: LLaVA-Med v1.5 (chaoyinshe/llava-med-v1.5-mistral-7b-hf @ 627be53) and Quilt-LLaVA v1.5-7B
(wisdomik/Quilt-Llava-v1.5-7b @ 1bdf5f8, converted with convert_llava_to_hf.py), each served by vLLM (max-model-len 4096). Same images
(the question image, at most 1,024 px, padded to a square with the CLIP mean colour and resized to 336 px as in LLaVA-1.5), same
questions and option order as the RETICLE arms, greedy decoding, each model's own chat template.
  mc     multiple choice with the LLaVA evaluation prompt ("Answer with the option's letter from the given choices directly."):
         PathMMU, CAMELYON16-MS, MIDOG-MS, UniToPatho-MS 7 mm / 800 um test -> arm <model> (PathMMU) or <set>_<model> in the benchmark's qa
         dir, scored by reticle/evaluation/score_regions.py / score_pathmmu.py next to RETICLE. When an answer names no option letter or
         option text, the Qwen3.8-27B server given by --judge_ports (no reasoning, greedy) reads it and returns the option it commits to, or none.
  mcdx   MultiCaRe-Dx turn 1 (presentation + figure -> most likely diagnosis + differential, in English: the models are English-only)
         -> answers_<model>.jsonl, judged by reticle/evaluation/consultation.py judge
  python -m reticle.evaluation.baselines.chat_vlms mc --model quilt --port 8980 --judge_ports 8931 [--sets pmmu c16 mdg u7 u8]
  python -m reticle.evaluation.baselines.chat_vlms mcdx --model quilt --port 8980"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import re
import urllib.request

from PIL import Image

from reticle.paths import DATA

BASE = (DATA/'runs')
SETS = {  # set: (test queries, qa dir, arm prefix)
    'pmmu': (BASE/'pathmmu_v3/inputs/queries_test.jsonl', BASE/'pathmmu_v3/qa', ''),
    'c16': (BASE/'camelyon16_ms_v1/inputs/queries_c16_test.jsonl', BASE/'camelyon16_ms_v1/qa', 'c16_'),
    'mdg': (BASE/'midog_ms_v1/inputs/queries_mdg_test.jsonl', BASE/'midog_ms_v1/qa', 'mdg_'),
    'u7': (BASE/'unitopatho_ms_v1/inputs/queries_u7_test.jsonl', BASE/'unitopatho_ms_v1/qa', 'u7_'),
    'u8': (BASE/'unitopatho_ms_v1/inputs/queries_u8_test.jsonl', BASE/'unitopatho_ms_v1/qa', 'u8_')}
MC = "{q}\n{o}\nAnswer with the option's letter from the given choices directly."
DX = ('Clinical presentation: {p}\nThis is an H&E image of the biopsy or resection specimen. What is the most likely diagnosis? '
      'Give the single most likely diagnosis first, then 2-3 differential diagnoses with reasons.')
MEAN = (122, 116, 104)  # CLIP image mean x 255, LLaVA-1.5 expand2square


def image_url(path):
    im = Image.open(path).convert('RGB')
    if max(im.size) > 1024:
        s = 1024/max(im.size); im = im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS)
    sq = Image.new('RGB', (max(im.size),)*2, MEAN); sq.paste(im, ((sq.width-im.width)//2, (sq.height-im.height)//2))
    buf = io.BytesIO(); sq.resize((336, 336), Image.BICUBIC).save(buf, 'PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


def ask(port, path, text, max_tokens):
    body = dict(model='ext', temperature=0.0, max_tokens=max_tokens,
                messages=[dict(role='user', content=[dict(type='image_url', image_url=dict(url=image_url(path))), dict(type='text', text=text)])])
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    r = json.load(urllib.request.urlopen(req, timeout=600))
    return r['choices'][0]['message']['content'] or '', r['choices'][0]['finish_reason']


def parse(text, options):
    letters = [o[0] for o in options]; t = text.strip()
    m = re.match(r'^\W*(?:answer|option)?\s*[:：]?\s*\(?([A-Z])\b', t, re.I)
    if m and m.group(1).upper() in letters:
        return m.group(1).upper()
    hits = [o[0] for o in options if o[3:].strip().lower() and o[3:].strip().lower() in t.lower()]  # the option text itself
    return hits[0] if len(hits) == 1 else None


EXTRACT = {'type': 'object', 'additionalProperties': False, 'required': ['choice'], 'properties': {'choice': {'type': 'string'}}}


def extract(judge_port, q, text):
    """When no letter or option text is found, the 27B (no reasoning, greedy) reads the answer and returns the single option it commits to,
    or 'none'."""
    body = dict(model='qwen', temperature=0.0, max_tokens=20, chat_template_kwargs=dict(enable_thinking=False),
                response_format=dict(type='json_schema', json_schema=dict(name='extract', schema=EXTRACT)),
                messages=[dict(role='user', content='A model answered a multiple-choice question.\nQuestion: '+q['question']+'\nOptions:\n'+'\n'.join(q['options'])+
                               '\nModel answer: '+text[:1500]+'\n\nWhich single option letter does the model answer commit to? Reply with the letter only, or "none" '
                               'if it commits to no single option.')])
    req = urllib.request.Request(f'http://127.0.0.1:{judge_port}/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    c = json.loads(json.load(urllib.request.urlopen(req, timeout=300))['choices'][0]['message']['content'])['choice'].strip().strip('()').upper()[:1]
    return c if c in [o[0] for o in q['options']] else None


def mc(model, port, sets, workers, judge_ports=None, max_tokens=256):
    import itertools
    cyc = itertools.cycle(judge_ports or [])
    for s in sets:
        qf, qa, prefix = SETS[s]; out = qa/f'{prefix}{model}.test.0.jsonl'
        if out.exists():
            continue
        qs = [json.loads(x) for x in open(qf)]

        def one(q):
            text, fin = ask(port, q['image_path'], MC.format(q=q['question'], o='\n'.join(q['options'])), max_tokens)
            c = parse(text, q['options']); how = 'regex' if c else None
            if c is None and judge_ports:
                c = extract(next(cyc), q, text); how = '27b' if c else 'none'
            return dict(id=q['id'], arm=f'{prefix}{model}', choice=c, extracted_by=how, raw=text[-500:], finish=fin, output_tokens=0)
        with ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(one, qs))
        out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        from collections import Counter
        print(s, model, json.dumps(dict(n=len(rows), unparsed=sum(r['choice'] is None for r in rows), by=Counter(r['extracted_by'] for r in rows))), flush=True)


def mcdx(model, port, workers):
    items = [json.loads(x) for x in open(BASE/'mcdx_v1/inputs/items.jsonl')]; out = BASE/f'mcdx_v1/answers_{model}.jsonl'
    if out.exists():
        return

    def one(it):
        text, fin = ask(port, it['image_path'], DX.format(p=it['presentation']), 600)
        return dict(id=it['id'], arm=model, turns=[dict(question='t1', answer=dict(answer=text, differential=[]), finish=fin)])
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, items))
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)); print('mcdx', model, len(rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['mc', 'mcdx']); ap.add_argument('--model', required=True, choices=['llavamed', 'quilt'])
    ap.add_argument('--port', type=int, required=True); ap.add_argument('--sets', nargs='*', default=list(SETS)); ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--judge_ports', default='', help='answer extraction by the 27B when no option is found'); ap.add_argument('--max_tokens', type=int, default=256)
    a = ap.parse_args(); jp = [int(x) for x in a.judge_ports.split(',')] if a.judge_ports else None
    mc(a.model, a.port, a.sets, a.workers, jp, a.max_tokens) if a.mode == 'mc' else mcdx(a.model, a.port, a.workers)
