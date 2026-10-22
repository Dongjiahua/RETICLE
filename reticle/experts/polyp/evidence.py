"""Evidence of the RETICLE polyp expert (mixture.py) for the agent: the mixture's probabilities, what each expert (magnification /
architecture / grade / retrieval) says, and where to look: the most typical 800 um tiles and the 200 um sub-fields with the highest
probability of the deciding feature, in 0-1000 coordinates (reticle/agent/region_qa.py cuts the views there).
  test  evidence_u8_test_moe.jsonl / evidence_u7_test_moe.jsonl, and the expert's own answers on the 800 um regions (arm
        u8_reticle_expert: the mixture's top option, RETICLE's reported score on these regions)
  dev   the 800 um development pool: 600 training regions (100 per class) with out-of-fold mixture evidence (every expert, the gate and
        the retrieval library come from the other WSI folds); queries_u8_dev.jsonl, evidence_u8_dev_moe.jsonl and gold rows with split
        'u8_dev'. It calibrates the 800 um records (reticle/experts/build_contracts.py u8).
  python -m reticle.experts.polyp.evidence test | dev"""
import argparse
import json
import random

import numpy as np

from reticle.paths import DATA

RUN = DATA/'runs/unitopatho_ms_v1'
L = 'ABCDEF'
WHAT = {'low': 'whole region, ~2 um/px', 'mid': '2x2 sub-fields, ~1 um/px', 'high': '4x4 sub-fields, ~0.5 um/px, nuclei',
        'tile': 'single 200 um sub-fields, ~0.5 um/px', 'mil': 'attention over sub-fields', 'knn': 'vote of the 25 most similar training regions'}


def fmt(p):
    return ' '.join(f'{L[i]}) {p[i]:.2f}' for i in range(6))


def top(p):
    i = int(np.argmax(p))
    return f'{L[i]} {p[i]:.2f}'


def opinions(E, names, typ, grade):
    s = '; '.join(f'{n} ({WHAT[n]}): {top(E[k])}' for k, n in enumerate(names) if n in WHAT)
    return (s+f'; architecture head: normal {typ[0]:.2f}, hyperplastic {typ[1]:.2f}, tubular {typ[2]:.2f}, tubulovillous {typ[3]:.2f}; '
            f'grade head (if adenoma): high-grade {grade[1]:.2f}')


def main():
    sel = json.loads((RUN/'moe/selection.json').read_text()); cv = sel['selected']['cv_balanced']
    who = (f'Mixture of MUSK experts (7 classifiers on frozen MUSK features at three magnifications, combined by {sel["selected"]["name"]} mean; '
           f'balanced accuracy {cv:.0f}% in cross-validation over training slides; experts can be wrong, especially on dysplasia grade)')
    gold_train = {}
    for g in map(json.loads, open(DATA/'scoring/unitopatho_ms_v1/gold.jsonl')):
        gold_train[g['id']] = g['answer']
    d = np.load(RUN/'moe/u8_test.npz'); names = [str(x) for x in d['expert_names']]; train_ids = [str(x) for x in d['train_ids']]
    ev, rows = [], []
    for i, qid in enumerate(d['ids']):
        qid = str(qid); m = d['mixture'][i]; E = d['experts'][i]
        nn = ', '.join(f"{gold_train[train_ids[j]]} {s:.2f}" for j, s in zip(d['nn_idx'][i][:5], d['nn_sim'][i][:5]))
        sub = d['sub'][i]; cls = int(m.argmax()); hg = sub[:, 3]+sub[:, 5]
        key = hg if cls in (2, 3, 4, 5) else sub[:, cls]
        fine = [[int(j % 4)*250, int(j//4)*250, int(j % 4+1)*250, int(j//4+1)*250] for j in np.argsort(-key)[:3]]
        text = (f'{who}.\nMixture, probability per option: {fmt(m)}.\nIndividual experts (top option): {opinions(E, names, d["type"][i], d["grade"][i])}.\n'
                f'Retrieval: the 5 most similar training regions (label, cosine): {nn}.')
        ev.append(dict(id=qid, text=text, coarse=[], fine=fine, mixture=[round(float(x), 4) for x in m]))
        rows.append(dict(id=qid, arm='u8_reticle_expert', choice=L[cls], finish='stop', output_tokens=0))
    (RUN/'evidence_u8_test_moe.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ev))
    (RUN/'qa/u8_reticle_expert.test.0.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    t = np.load(RUN/'moe/u7_test_tiles.npz'); qs = {json.loads(x)['id']: json.loads(x) for x in open(RUN/'inputs/queries_u7_test.jsonl')}
    agg = json.loads((RUN/'moe/agg7.json').read_text())['selected'] if (RUN/'moe/agg7.json').exists() else None  # 7 mm aggregation (mixture.agg7)
    region = dict(zip([str(x) for x in np.load(RUN/'moe/u7_test_region.npz')['ids']], np.load(RUN/'moe/u7_test_region.npz')['probs'])) if agg else {}
    ev = []
    for i, qid in enumerate(t['ids']):
        qid = str(qid); size = np.load(qs[qid]['native'], mmap_mode='r').shape[0]
        keep = (t['tissue'][i] >= 0.3) & ~np.isnan(t['mixture'][i][:, 0])
        if not keep.any():
            keep = ~np.isnan(t['mixture'][i][:, 0])
        M, E, S, C = t['mixture'][i][keep], t['experts'][i][keep], t['sub'][i][keep], t['coords'][i][keep]
        mean, mx = M.mean(0), M.max(0); Em = E.mean(0)
        reg = region[qid] if agg else mean; cls = int(reg.argmax())
        typ, grade = t['type'][i][keep].mean(0), t['grade'][i][keep]
        box = lambda x, y, w: [round(1000*float(x)/size, 1), round(1000*float(y)/size, 1), round(1000*float(x+w)/size, 1), round(1000*float(y+w)/size, 1)]
        coarse, fine = [], []
        for j in np.argsort(-M[:, cls]):  # most typical 800 um tiles, not overlapping; in each, the most telling 200 um sub-field
            b = box(*C[j], 1812)
            if all(abs(b[0]-c[0]) >= 114 or abs(b[1]-c[1]) >= 114 for c in coarse):
                coarse.append(b)
                key = S[j][:, 3]+S[j][:, 5] if cls in (2, 3, 4, 5) else S[j][:, cls]; s = int(np.argmax(key))
                fine.append(box(C[j][0]+(s % 4)*453, C[j][1]+(s//4)*453, 453))
            if len(coarse) == 3:
                break
        hg = (S[..., 3]+S[..., 5]).reshape(-1); hg_at = np.unravel_index(np.argsort(-hg)[:3], S.shape[:2])
        hg_boxes = ', '.join(f"{[round(v) for v in box(C[a][0]+(b % 4)*453, C[a][1]+(b//4)*453, 453)]} {hg.reshape(S.shape[:2])[a, b]:.2f}" for a, b in zip(*hg_at))
        verdict = (f'Region-level reading (tile opinions combined by the rule chosen on training regions, {agg["aggregator"]}): {fmt(reg)}.\n' if agg else '')
        text = (f'{who}, applied to the {int(keep.sum())} tissue tiles (800 um each) of this region.\n{verdict}Mixture, mean over tiles: {fmt(mean)}.\n'
                f'Highest single-tile probability: {fmt(mx)}.\nIndividual experts, mean over tiles (top option): {opinions(Em, [str(x) for x in t["expert_names"]], typ, grade.mean(0))}; '
                f'grade head, highest tile: high-grade {grade[:, 1].max():.2f}.\n'
                f'Tiles most typical of option {L[cls]} (0-1000 coordinates): {", ".join(str([round(v) for v in b]) for b in coarse)}.\n'
                f'200 um fields with the highest high-grade dysplasia probability: {hg_boxes}.')
        ev.append(dict(id=qid, text=text, coarse=coarse, fine=fine, mean=[round(float(x), 4) for x in mean]))
    (RUN/'evidence_u7_test_moe.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ev))
    print('u8', len(d['ids']), 'u7', len(t['ids']))


def dev():
    from reticle.experts.polyp.mixture import EXPERTS, MOE, Expert, folds, gate_fit, train_data
    from reticle.experts.polyp.probes import GOLD, read_jsonl
    region, t2, t4, y, wsi, ids = train_data(); fold = folds(wsi)
    rng = random.Random(20261019); pool = sorted(i for c in range(6) for i in rng.sample(list(np.flatnonzero(y == c)), 100))
    sel = json.loads((MOE/'selection.json').read_text()); oof = np.load(MOE/'oof_train.npz')
    who = (f'Mixture of MUSK experts (7 classifiers on frozen MUSK features at three magnifications, combined by {sel["selected"]["name"]} mean; '
           f'balanced accuracy {sel["selected"]["cv_balanced"]:.0f}% in cross-validation over training slides; experts can be wrong, especially on dysplasia grade)')
    gold = {g['id']: g for g in read_jsonl(GOLD)}; qs = {q['id']: q for q in read_jsonl(RUN/'inputs/queries_u8_train.jsonl')}
    ev = {}
    for k in range(5):
        idx = [i for i in pool if fold[i] == k]
        if not idx:
            continue
        tr = fold != k; ex = Expert(region[tr], t2[tr], t4[tr], y[tr]); gate = gate_fit(sel['selected']['name'], oof['P'][tr], oof['y'][tr])
        out, extra = ex.predict(region[idx], t2[idx], t4[idx]); P = np.stack([out[e] for e in EXPERTS], 1); M = gate(P)
        lib = np.flatnonzero(tr)
        for j, i in enumerate(idx):
            nn = ', '.join(f'{L[y[lib[n]]]} {s:.2f}' for n, s in zip(extra['nn_idx'][j][:5], extra['nn_sim'][j][:5]))
            text = (f'{who}.\nMixture, probability per option: {fmt(M[j])}.\nIndividual experts (top option): {opinions(P[j], EXPERTS, extra["type"][j], extra["grade"][j])}.\n'
                    f'Retrieval: the 5 most similar training regions (label, cosine): {nn}.')
            ev[ids[i]] = dict(id=ids[i], text=text, coarse=[], fine=[], mixture=[round(float(x), 4) for x in M[j]])
        print('fold', k, len(idx), flush=True)
    order = [ids[i] for i in pool]
    (RUN/'inputs/queries_u8_dev.jsonl').write_text(''.join(json.dumps(dict(qs[i], split='dev'))+'\n' for i in order))
    (RUN/'evidence_u8_dev_moe.jsonl').write_text(''.join(json.dumps(ev[i])+'\n' for i in order))
    rows = [g for g in read_jsonl(GOLD) if g['split'] != 'u8_dev']+[dict(gold[i], split='u8_dev') for i in order]
    GOLD.write_text(''.join(json.dumps(g)+'\n' for g in rows))
    print('u8_dev', len(order), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', nargs='?', default='test', choices=['test', 'dev']); a = ap.parse_args()
    {'test': main, 'dev': dev}[a.mode]()
