"""Score PathMMU arms against the frozen answers: accuracy per arm and source, and paired differences versus a reference arm with an
image-grouped bootstrap (10,000 replicates). Answers are read from runs/pathmmu_v3/qa/<arm>.<split>.<shard>.jsonl (reticle/agent/vlm_qa.py).
  python -m reticle.evaluation.score_pathmmu --split test --reference qwen --out $RETICLE_DATA/results/pathmmu_test.json"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from reticle.paths import DATA


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().split('\n') if x]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--qa', default=str(DATA/'runs/pathmmu_v3/qa'))
    p.add_argument('--gold', default=str(DATA/'scoring/pathmmu_v3/gold.jsonl'))
    p.add_argument('--split', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    gold = {g['id']: g for g in read_jsonl(a.gold) if g['split'] == a.split}
    arms = defaultdict(dict)
    for f in Path(a.qa).glob(f'*.{a.split}.*.jsonl'):
        for r in read_jsonl(f):
            arms[r['arm']][r['id']] = r
    complete = {k: v for k, v in arms.items() if set(v) == set(gold)}
    ids = sorted(gold); images = sorted({gold[i]['image_sha256'] for i in ids})
    image_of = np.array([images.index(gold[i]['image_sha256']) for i in ids])
    correct = {k: np.array([v[i]['choice'] == gold[i]['answer'] for i in ids], dtype=float) for k, v in complete.items()}
    rng = np.random.default_rng(20261018)
    draws = rng.integers(0, len(images), size=(10000, len(images)))
    counts = np.stack([np.bincount(d, minlength=len(images)) for d in draws])  # replicate weight per image
    weights = counts[:, image_of]
    result = dict(split=a.split, n=len(ids), images=len(images), reference=a.reference, arms={})
    for k, c in sorted(correct.items()):
        rows = complete[k]
        by_source = defaultdict(list)
        for i, x in zip(ids, c):
            by_source[gold[i]['source']].append(x)
        entry = dict(correct=int(c.sum()), accuracy=round(100*c.mean(), 2),
                     by_source={s: f'{int(sum(v))}/{len(v)} = {100*np.mean(v):.2f}%' for s, v in sorted(by_source.items())},
                     unparsed=sum(rows[i]['choice'] is None for i in ids),
                     truncated=sum(rows[i]['finish'] == 'length' for i in ids),
                     mean_output_tokens=round(float(np.mean([rows[i]['output_tokens'] for i in ids])), 1))
        if a.reference in correct and k != a.reference:
            d = c-correct[a.reference]
            boot = (weights*d).sum(1)/weights.sum(1)
            entry['minus_reference_pp'] = round(100*d.mean(), 2)
            entry['ci95_pp'] = [round(float(100*np.percentile(boot, 2.5)), 2), round(float(100*np.percentile(boot, 97.5)), 2)]
            entry['fixed_broken'] = [int(((c == 1) & (correct[a.reference] == 0)).sum()), int(((c == 0) & (correct[a.reference] == 1)).sum())]
        result['arms'][k] = entry
    result['incomplete_arms'] = {k: len(v) for k, v in arms.items() if k not in complete}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for k, e in result['arms'].items():
        print(f"{k:28s} {e['accuracy']:6.2f}  {e.get('minus_reference_pp', ''):>6} {e.get('ci95_pp', '')}  unparsed={e['unparsed']} trunc={e['truncated']} tok={e['mean_output_tokens']}")
    if result['incomplete_arms']:
        print('incomplete:', result['incomplete_arms'])


if __name__ == '__main__':
    main()
