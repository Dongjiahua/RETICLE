"""Expert records (docs/expert_interface.md) for the polyp expert, plus the calibration helpers the other experts reuse.
  u8  800 um regions: the gated mixture's probabilities, temperature fitted on the 600 development regions (out-of-fold mixture,
      reticle/experts/polyp/evidence.py dev); regions = the three 200 um sub-fields that most support the answer.
  u7  7 mm regions: the learned tile aggregation, temperature fitted on the 619 training regions (out-of-fold tile experts and
      aggregator, same WSI folds as reticle/experts/polyp/mixture.py agg7); regions = the three most typical 800 um tiles.
Writes runs/contract_v1/{u8,u7}_{calib,test}.jsonl and results/contract_v1/calibration.json (calibration on the pool and on test).
The metastasis and mitosis records are written by their own readers (metastasis/segmenter.py read, mitosis/crossfit.py read).
  python -m reticle.experts.build_contracts u8 u7"""
import argparse
import json

import numpy as np

from reticle.experts.contract import apply_temperature, fit_temperature, record, report, write
from reticle.paths import DATA

REPORT = DATA/'results/contract_v1/calibration.json'
L6 = ['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']


def logp(p):
    return np.log(np.clip(np.asarray(p, float), 1e-6, 1))


def head_cv(X, Y, groups, k, seeds=(0, 1, 2)):
    """Multinomial probe on standardised features: out-of-fold probabilities (5 group folds) and a head fitted on all rows."""
    from reticle.experts.polyp.mixture import probe
    g = sorted(set(groups)); fold_of = {v: i % 5 for i, v in enumerate(np.random.default_rng(0).permutation(g))}; fold = np.array([fold_of[v] for v in groups])
    oof = np.zeros((len(Y), k))
    for f in range(5):
        tr = fold != f
        if not (fold == f).any():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-6
        oof[fold == f] = np.mean([probe((X[tr]-mu)/sd, Y[tr], k, s, epochs=100, lr=0.01)((X[fold == f]-mu)/sd) for s in seeds], 0)
    mu, sd = X.mean(0), X.std(0)+1e-6; heads = [probe((X-mu)/sd, Y, k, s, epochs=100, lr=0.01) for s in seeds]
    return oof, (lambda Z: np.mean([h((Z-mu)/sd) for h in heads], 0))


def u8():
    RUN = DATA/'runs/unitopatho_ms_v1'; gold = {}
    for g in map(json.loads, open(DATA/'scoring/unitopatho_ms_v1/gold.jsonl')):
        gold[(g['split'], g['id'])] = 'ABCDEF'.index(g['answer'])
    dev = [json.loads(x) for x in open(RUN/'evidence_u8_dev_moe.jsonl')]; Pd = np.array([r['mixture'] for r in dev]); Yd = np.array([gold[('u8_dev', r['id'])] for r in dev])
    t = fit_temperature(logp(Pd), Yd); Pdc = apply_temperature(logp(Pd), t)
    d = np.load(RUN/'moe/u8_test.npz'); ids = [str(x) for x in d['ids']]; Pt = apply_temperature(logp(d['mixture']), t); Yt = np.array([gold[('u8_test', i)] for i in ids])
    rows = []
    for i, qid in enumerate(ids):
        cls = int(Pt[i].argmax()); sub = d['sub'][i]; key = sub[:, 3]+sub[:, 5] if cls in (2, 3, 4, 5) else sub[:, cls]
        rois = [dict(box=[(j % 4)*250, (j//4)*250, (j % 4+1)*250, (j//4+1)*250], score=key[j]) for j in np.argsort(-key)[:3]]
        rows.append(record(qid, 'u8', 'unito_moe_geo', Pt[i], rois))
    write('u8', 'calib', [record(r['id'], 'u8', 'unito_moe_geo', p) for r, p in zip(dev, Pdc)]); write('u8', 'test', rows)
    return dict(temperature=t, calib=report(Pdc, Yd), calib_raw=report(Pd, Yd), test=report(Pt, Yt), test_raw=report(d['mixture'], Yt))


def u7():
    """Out-of-fold 7 mm region probabilities of the learned aggregator on the 619 training regions (same folds as mixture.agg7)."""
    from reticle.experts.polyp.mixture import EXPERTS, GOLD, HIRES, MOE, Expert, agg_features, folds, gate_fit, read_jsonl, train_data, probe
    region, t2, t4, y, wsi, _ = train_data(); fold = folds(wsi); fold_of = dict(zip(wsi, fold)); gold = {g['id']: g for g in read_jsonl(GOLD)}
    oof = np.load(MOE/'oof_train.npz'); gate = gate_fit(json.loads((MOE/'selection.json').read_text())['selected']['name'], oof['P'], oof['y'])
    parts = [np.load(p) for p in sorted(HIRES.glob('u7_train.*.npz'))]; cat = lambda k: np.concatenate([p[k] for p in parts])
    qidx, rg, a2, a4 = cat('qidx'), cat('region').astype(np.float32), cat('t2').astype(np.float32), cat('t4').astype(np.float32)
    tf = np.array([fold_of[gold[f'U7000:train:{q}']['wsi']] for q in qidx]); E = np.zeros((len(qidx), len(EXPERTS), 6))
    for k in range(5):
        ex = Expert(region[fold != k], t2[fold != k], t4[fold != k], y[fold != k]); out, _ = ex.predict(rg[tf == k], a2[tf == k], a4[tf == k])
        E[tf == k] = np.stack([out[e] for e in EXPERTS], 1)
    M = gate(E); regs = sorted(set(qidx.tolist()))
    X = np.stack([agg_features(E[qidx == q], M[qidx == q]) for q in regs]); Y = np.array([L6.index(gold[f'U7000:train:{q}']['label']) for q in regs])
    RF = np.array([fold_of[gold[f'U7000:train:{q}']['wsi']] for q in regs]); P = np.zeros((len(regs), 6))
    for k in range(5):  # the learned aggregator (agg7) cross-validated over the same WSI folds
        if (RF == k).any():
            P[RF == k] = np.mean([probe(X[RF != k], Y[RF != k], 6, s, epochs=50, lr=0.01)(X[RF == k]) for s in range(3)], 0)
    t = fit_temperature(logp(P), Y); Pc = apply_temperature(logp(P), t)
    RUN = DATA/'runs/unitopatho_ms_v1'; d = np.load(MOE/'u7_test_region.npz'); ids = [str(x) for x in d['ids']]
    Pt = apply_temperature(logp(d['probs']), t); Yt = np.array([L6.index(gold[i]['label']) for i in ids])
    tiles = np.load(MOE/'u7_test_tiles.npz'); qs = {json.loads(x)['id']: json.loads(x) for x in open(RUN/'inputs/queries_u7_test.jsonl')}; rows = []
    for i, qid in enumerate(ids):
        size = np.load(qs[qid]['native'], mmap_mode='r').shape[0]; keep = (tiles['tissue'][i] >= 0.3) & ~np.isnan(tiles['mixture'][i][:, 0])
        Mi, Ci = tiles['mixture'][i][keep], tiles['coords'][i][keep]; cls = int(Pt[i].argmax()); rois = []
        for j in np.argsort(-Mi[:, cls]):
            b = [1000*Ci[j][0]/size, 1000*Ci[j][1]/size, 1000*(Ci[j][0]+1812)/size, 1000*(Ci[j][1]+1812)/size]
            if all(abs(b[0]-r['box'][0]) >= 114 or abs(b[1]-r['box'][1]) >= 114 for r in rois):
                rois.append(dict(box=b, score=Mi[j, cls]))
            if len(rois) == 3:
                break
        rows.append(record(qid, 'u7', 'unito_moe_agg7', Pt[i], rois))
    write('u7', 'calib', [record(f'U7000:train:{q}', 'u7', 'unito_moe_agg7', p) for q, p in zip(regs, Pc)]); write('u7', 'test', rows)
    return dict(temperature=t, calib=report(Pc, Y), calib_raw=report(P, Y), test=report(Pt, Yt), test_raw=report(d['probs'], Yt))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('tasks', nargs='+', choices=['u8', 'u7']); a = ap.parse_args()
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    for task in a.tasks:
        rep[task] = {'u8': u8, 'u7': u7}[task](); print(task, json.dumps(rep[task]), flush=True)
        REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(json.dumps(rep, indent=2)+'\n')
