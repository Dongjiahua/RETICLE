"""MultiCaRe-Dx: three-turn consultations (diagnosis -> work-up -> treatment) with reticle/agent/consult.py over vLLM servers, the blinded
judge of the first-turn diagnosis (the same VLM, text only, sees the gold fields and one answer at a time, never the arm), and the paired
summary.

Arms (both reticle/agent/consult.py version 5: figure + presentation, reasoning):
  qwen      the same VLM without retrieval
  reticle   + evidence retrieval and cited answers
Baselines answered by reticle/evaluation/baselines: musk (nearest published case), llavamed, quilt.
Metrics: t1_top1 (same disease entity as the gold diagnosis), t1_top3 (gold among the differential). Paired item bootstrap (10,000
replicates).
  python -m reticle.evaluation.consultation answer --arm reticle --ports 8931,8932 [--limit n]   (one GPU for the evidence index; resumable)
  python -m reticle.evaluation.consultation judge --ports 8931,8932
  python -m reticle.evaluation.consultation summary          -> $RETICLE_DATA/results/consultation.json"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import re
import threading

import numpy as np
from PIL import Image

from reticle.paths import CONFIGS, DATA

BASE = DATA
RUN = BASE/'runs/mcdx_v1'
GOLD = BASE/'scoring/mcdx_v1/gold.jsonl'
CFG = json.loads((CONFIGS/'evaluation/consultation.json').read_text())   # the questions of the reported runs (verbatim)
T1, T2, T3 = CFG['turns']['t1'], CFG['turns']['t2'], CFG['turns']['t3']
ARMS = {'qwen': False, 'reticle': True}   # arm: evidence retrieval
REFERENCES = ('musk', 'llavamed', 'quilt')   # baselines answered by reticle/evaluation/baselines


def arm_files(arm):
    """answers_<arm>.jsonl and its shard files answers_<arm>.s<k>.jsonl."""
    return sorted(f for f in RUN.glob(f'answers_{arm}*.jsonl') if f.name == f'answers_{arm}.jsonl' or re.fullmatch(rf'answers_{re.escape(arm)}\.s\d+\.jsonl', f.name))


def answer(arm, ports, limit, threads=6, shard=0, nshards=1):
    from reticle.agent.consult import Library, Session
    items = [json.loads(x) for x in open(RUN/'inputs/items.jsonl')][:limit]; gold = {g['id']: g for g in map(json.loads, open(GOLD))}
    out = RUN/(f'answers_{arm}.jsonl' if nshards == 1 else f'answers_{arm}.s{shard}.jsonl')
    done = {json.loads(x)['id'] for f in arm_files(arm) for x in open(f)}  # any earlier file of this arm
    ret = ARMS[arm]; lib = Library('cuda') if ret else None
    cyc = itertools.cycle(ports); lock = threading.Lock(); fh = out.open('a')

    def run(it):
        with lock:
            port = next(cyc)
        image = Image.open(it['image_path']).convert('RGB')
        try:
            s = Session(lib, port, image, retrieval=ret, exclude={gold[it['id']]['article_id']}, near_dup=0.90 if ret else None, version=5)
            for q in (T1.format(presentation=it['presentation']), T2, T3):
                s.ask(q)
        except Exception as exc:
            print('error', it['id'], exc, flush=True); return
        ev = [{k: e.get(k) for k in ('evidence_id', 'kind', 'article_id', 'case_id', 'title', 'similarity', 'match', 'turn', 'sentences')} for e in s.evidence]
        row = dict(id=it['id'], arm=arm, turns=[dict(question=t['question'], answer=t['answer'], new_evidence=t['new_evidence'], search_terms=t['search_terms'],
                                                     gate=t.get('gate'), hypotheses=t.get('hypotheses'), usage=t['usage']) for t in s.turns], evidence=ev)
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False)+'\n'); fh.flush()
    todo = [it for it in items if it['id'] not in done][shard::nshards]; print(arm, 'todo', len(todo), flush=True)
    with ThreadPoolExecutor(threads*len(ports)) as ex:
        list(ex.map(run, todo))
    print(arm, 'done', flush=True)


JUDGE_T1 = {'type': 'object', 'additionalProperties': False, 'required': ['reference', 'answer_top', 'top1_match', 'in_differential'],
            'properties': {'reference': {'type': 'string'}, 'answer_top': {'type': 'string'}, 'top1_match': {'type': 'boolean'}, 'in_differential': {'type': 'boolean'}}}
def chat(port, text, schema, max_tokens=600):
    from reticle.agent.consult import post  # retries vLLM grammar failures
    body = dict(model='qwen', messages=[dict(role='user', content=text)], temperature=0.0, max_tokens=max_tokens, chat_template_kwargs=dict(enable_thinking=False),
                response_format=dict(type='json_schema', json_schema=dict(name='judge', schema=schema)))
    for attempt in range(3):
        try:
            return json.loads(post(port, body)['choices'][0]['message']['content'])
        except json.JSONDecodeError:
            body['max_tokens'] = int(body['max_tokens']*2)
    raise ValueError('judge returned no valid JSON')


def render(ans):
    d = '; '.join(f"{x['diagnosis']} ({x['likelihood']})" for x in ans.get('differential', []))
    return ans['answer'][:3000]+('\nDifferential: '+d if d else '')


def judge(ports, only=None):
    gold = {g['id']: g for g in map(json.loads, open(GOLD))}; cyc = itertools.cycle(ports); lock = threading.Lock()
    arms = sorted({f.stem[8:].split('.')[0] for f in RUN.glob('answers_*.jsonl')} & (set(only) if only else set(ARMS) | set(REFERENCES[1:])))
    for arm in arms:
        out = RUN/f'judged_{arm}.jsonl'; done = {json.loads(x)['id'] for x in open(out)} if out.exists() else set()
        rows = list({r['id']: r for f in arm_files(arm) for r in map(json.loads, open(f))}.values())
        rows = [r for r in rows if r['id'] not in done]; fh = out.open('a')

        def one(r):
            try:
                return one_(r)
            except Exception as exc:
                print('judge error', arm, r['id'], exc, flush=True)

        def one_(r):
            with lock:
                port = next(cyc)
            g = gold[r['id']]; res = dict(id=r['id'], arm=arm)
            ref = f"Gold diagnosis: {g['final_diagnosis']} (aliases: {', '.join(g['diagnosis_aliases'])}). Published title: {g['title']}. Figure caption: {g['caption'][:600]}"
            res['t1'] = chat(port, ref+'\n\nCandidate answer (may be Chinese):\n'+render(r['turns'][0]['answer'])+
                             '\n\nanswer_top = the single diagnosis the candidate commits to. top1_match = true only if it is the same disease entity as the gold '
                             '(synonyms and more specific subtypes count; a different entity, a broad category such as "tumor" or "inflammation", or no committed '
                             'diagnosis count as false). in_differential = true if the gold entity is the committed diagnosis or one of the listed differential diagnoses.', JUDGE_T1)
            with lock:
                fh.write(json.dumps(res, ensure_ascii=False)+'\n'); fh.flush()
        with ThreadPoolExecutor(8*len(ports)) as ex:
            list(ex.map(one, rows))
        print('judged', arm, flush=True)


def summary():
    def load(arm):
        p = RUN/f'judged_{arm}.jsonl'
        return {json.loads(x)['id']: json.loads(x) for x in open(p)} if p.exists() else {}
    arms = {a: load(a) for a in (*ARMS, *REFERENCES)}
    metrics = {'t1_top1': lambda j: float(j['t1']['top1_match']), 't1_top3': lambda j: float(j['t1']['in_differential'] or j['t1']['top1_match'])}
    res = dict(arms={}, comparisons=[]); rng = np.random.default_rng(0)
    for a, d in arms.items():
        res['arms'][a] = {m: (round(100*float(np.mean(v)), 2), len(v)) for m, f in metrics.items() for v in [[f(j) for j in d.values()]] if v}
    for x, y in [('reticle', r) for r in ('qwen',)+REFERENCES]:
        for m, f in metrics.items():
            ids = [i for i in arms[x] if i in arms[y]]
            if not ids:
                continue
            a = np.array([f(arms[x][i]) for i in ids]); b = np.array([f(arms[y][i]) for i in ids]); bt = rng.integers(0, len(ids), (10000, len(ids)))
            dd = (a[bt]-b[bt]).mean(1)
            res['comparisons'].append(dict(arm=x, ref=y, metric=m, n=len(ids), diff_pp=round(100*float((a-b).mean()), 2),
                                           ci95=[round(100*float(np.percentile(dd, 2.5)), 2), round(100*float(np.percentile(dd, 97.5)), 2)]))
    out = DATA/'results/consultation.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False)+'\n'); print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['answer', 'judge', 'summary']); ap.add_argument('--arm', default='reticle', choices=list(ARMS))
    ap.add_argument('--ports', default='8931'); ap.add_argument('--limit', type=int, default=None); ap.add_argument('--threads', type=int, default=6); ap.add_argument('--arms', default=''); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); a = ap.parse_args()
    ports = [int(p) for p in a.ports.split(',')]
    {'answer': lambda: answer(a.arm, ports, a.limit, a.threads, a.shard, a.nshards), 'judge': lambda: judge(ports, a.arms.split(',') if a.arms else None), 'summary': summary}[a.mode]()
