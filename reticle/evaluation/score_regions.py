"""Score the large-region benchmarks (UniToPatho-MS u8 / u7, CAMELYON16-MS c16, MIDOG-MS mdg): accuracy, balanced accuracy (mean
per-class recall, the primary metric), per-class recall, and paired differences with a slide-grouped bootstrap (10,000 replicates for
accuracy, 2,000 for balanced accuracy). CAMELYON16-MS also reports detection sensitivity / specificity (any tumour vs none) and the
recall on small deposits (isolated tumour cells + micrometastases). The comparisons come from the config ({set: [[arm, reference]]}).
Answers are read from runs/<benchmark>/qa/<arm>.<split>.<shard>.jsonl (written by reticle/agent/region_qa.py).
  python -m reticle.evaluation.score_regions --config configs/evaluation/region_comparisons.json --out $RETICLE_DATA/results/regions_test.json"""
import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import numpy as np

from reticle.paths import DATA

SETS = {'u8': (str(DATA/'runs/unitopatho_ms_v1'), str(DATA/'scoring/unitopatho_ms_v1/gold.jsonl'), ['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']),
        'u7': (str(DATA/'runs/unitopatho_ms_v1'), str(DATA/'scoring/unitopatho_ms_v1/gold.jsonl'), ['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']),
        'c16': (str(DATA/'runs/camelyon16_ms_v1'), str(DATA/'scoring/camelyon16_ms_v1/gold.jsonl'), ['none', 'itc', 'micro', 'macro']),
        'mdg': (str(DATA/'runs/midog_ms_v1'), str(DATA/'scoring/midog_ms_v1/gold.jsonl'), ['0', '1', '2-3', '4+'])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--out', required=True); ap.add_argument('--split', default='test')
    a = ap.parse_args(); cfg = json.loads(Path(a.config).read_text()); result = dict(split=a.split, arms={}, comparisons=[])
    rng = np.random.default_rng(20261019)
    for s, (run, gold_path, labels) in SETS.items():
        gold = {g['id']: g for g in map(json.loads, open(gold_path)) if g['split'] == f'{s}_{a.split}'}
        if not gold:
            continue
        arms = defaultdict(dict)
        for f in glob.glob(f'{run}/qa/*.{a.split}.*.jsonl'):
            for r in map(json.loads, open(f)):
                if r['id'] in gold:
                    arms[r['arm']][r['id']] = r
        ids = sorted(gold); k = len(labels); letters = 'ABCDEF'[:k]
        wsi = sorted({gold[i]['wsi'] for i in ids}); g_of = np.array([wsi.index(gold[i]['wsi']) for i in ids])
        y = np.array([letters.index(gold[i]['answer']) for i in ids])
        W = np.stack([np.bincount(rng.integers(0, len(wsi), len(wsi)), minlength=len(wsi))[g_of] for _ in range(10000)])
        complete = {n: v for n, v in arms.items() if all(i in v for i in ids)}
        choice = {n: np.array([letters.index(v[i]['choice']) if v[i].get('choice') and v[i]['choice'] in letters else -1 for i in ids]) for n, v in complete.items()}
        correct = {n: (c == y).astype(float) for n, c in choice.items()}

        def bal(c, w=None):
            w = np.ones(len(c)) if w is None else w
            return np.nanmean([np.sum(c[y == j]*w[y == j])/np.sum(w[y == j]) if np.sum(w[y == j]) > 0 else np.nan for j in range(k)])
        for n, c in correct.items():
            row = dict(set=s, n=len(ids), groups=len(wsi), accuracy=round(100*c.mean(), 2), balanced_accuracy=round(100*bal(c), 2),
                       per_class_recall={l: round(100*c[y == j].mean(), 1) for j, l in enumerate(labels)},
                       unparsed=int((choice[n] < 0).sum()), truncated=sum(complete[n][i].get('finish') == 'length' for i in ids),
                       median_prompt_tokens=float(np.median([complete[n][i].get('prompt_tokens', 0) for i in ids])),
                       median_output_tokens=float(np.median([complete[n][i].get('output_tokens', 0) for i in ids])))
            if s == 'c16':
                pos, det = y > 0, choice[n] > 0
                row.update(detection_sensitivity=round(100*det[pos].mean(), 1), detection_specificity=round(100*(~det[~pos]).mean(), 1),
                           small_deposit_recall=round(100*c[(y == 1) | (y == 2)].mean(), 1), small_deposit_detected=round(100*det[(y == 1) | (y == 2)].mean(), 1))
            result['arms'][n] = row
        for arm, ref in cfg['comparisons'].get(s, []):
            if arm not in correct or ref not in correct:
                continue
            d_acc = (W*(correct[arm]-correct[ref])).sum(1)/W.sum(1)
            d_bal = np.array([bal(correct[arm], w)-bal(correct[ref], w) for w in W[:2000]])
            result['comparisons'].append(dict(set=s, arm=arm, reference=ref, balanced_pp=round(100*(bal(correct[arm])-bal(correct[ref])), 2),
                                              balanced_ci95=[round(100*float(x), 2) for x in np.nanpercentile(d_bal, [2.5, 97.5])],
                                              accuracy_pp=round(100*(correct[arm].mean()-correct[ref].mean()), 2),
                                              accuracy_ci95=[round(100*float(x), 2) for x in np.percentile(d_acc, [2.5, 97.5])],
                                              fixed_broken=[int(((correct[arm] == 1) & (correct[ref] == 0)).sum()), int(((correct[arm] == 0) & (correct[ref] == 1)).sum())]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(result, indent=2)+'\n')
    for n, v in sorted(result['arms'].items(), key=lambda x: (x[1]['set'], x[0])):
        extra = f"  det sens {v['detection_sensitivity']} spec {v['detection_specificity']} small {v['small_deposit_recall']}" if v['set'] == 'c16' else ''
        print(f"{v['set']:4s} {n:26s} acc {v['accuracy']:6.2f}  bal {v['balanced_accuracy']:6.2f}  unparsed {v['unparsed']} trunc {v['truncated']}{extra}")
    for c in result['comparisons']:
        print(f"{c['set']} {c['arm']} - {c['reference']}: bal {c['balanced_pp']:+.2f} {c['balanced_ci95']}  acc {c['accuracy_pp']:+.2f} {c['accuracy_ci95']}  fixed/broken {c['fixed_broken']}")


if __name__ == '__main__':
    main()
