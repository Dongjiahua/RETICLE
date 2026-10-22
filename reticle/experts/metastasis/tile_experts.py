"""Tile-level mixture of MUSK probes for CAMELYON16-MS: tumour-vs-normal classifiers on frozen MUSK features (features.py).
Experts: high / mid / low (linear probes at ~0.45 / ~0.9 / ~1.8 um/px), ms (probe on the three scales concatenated), knn (25-NN vote
over training tiles, high power); gates avg / geo / stack. Expert and gate are chosen by 5-fold slide-grouped CV on the expert-train
slides (tile AUC, then balanced accuracy). Each region gets a tumour-probability map on the high-power grid (27 x 27 fields of ~180 um;
each field also gets the mid / low probability of the tile containing it). The maps feed the reader candidates of the segmenter
(segmenter.read: out-of-fold maps of the trreg pool are computed there, the test maps are written here to feats/maps_test.npz).
  python -m reticle.experts.metastasis.tile_experts      -> feats/maps_test.npz, expert_selection.json"""
import argparse
import json

import numpy as np
from scipy import ndimage

from reticle.benchmarks.camelyon16_ms import REGION, RUN
from reticle.experts.metastasis.features import FEATS, SCALES
from reticle.experts.polyp.mixture import cuda, l2, probe

EXPERTS = ['high', 'mid', 'low', 'ms', 'knn']
GATES = ['avg', 'geo', 'stack']


def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1); n1 = y.sum(); n0 = len(y)-n1
    return float((r[y == 1].sum()-n1*(n1+1)/2)/(n1*n0))


class Experts:
    def __init__(self, F, y, seeds=(0, 1, 2)):
        S = lambda x: [probe(x, y, 2, s) for s in seeds]
        self.p = {k: S(F[k]) for k in ('high', 'mid', 'low')}; self.p['ms'] = S(np.concatenate([F['high'], F['mid'], F['low']], 1))
        self.lib = cuda(l2(F['high'])); self.ylib = y; self.prior = np.bincount(y, minlength=2)/len(y)

    def knn(self, x, k=25):
        import torch
        top = (cuda(l2(x)) @ self.lib.T).topk(k, dim=1); w = torch.exp((top.values-1)/0.05).cpu().numpy(); idx = top.indices.cpu().numpy()
        P = np.stack([(w*(self.ylib[idx] == c)).sum(1)/self.prior[c] for c in (0, 1)], 1)
        return P[:, 1]/P.sum(1), (self.ylib[idx] == 1).sum(1)

    def predict(self, F):  # -> [N, E] tumour probabilities, knn tumour counts
        out = [np.mean([f(F[k])[:, 1] for f in self.p[k]], 0) for k in ('high', 'mid', 'low')]
        out.append(np.mean([f(np.concatenate([F['high'], F['mid'], F['low']], 1))[:, 1] for f in self.p['ms']], 0))
        pk, cnt = self.knn(F['high']); out.append(pk)
        return np.stack(out, 1), cnt


def gate_fit(kind, P, y):
    lg = lambda Q: np.log(np.clip(np.stack([1-Q, Q], -1), 1e-6, 1)).reshape(len(Q), -1)
    if kind == 'avg':
        return lambda Q: Q.mean(1)
    if kind == 'geo':
        return lambda Q: (lambda a, b: b/(a+b))(np.exp(np.log(np.clip(1-Q, 1e-6, 1)).mean(1)), np.exp(np.log(np.clip(Q, 1e-6, 1)).mean(1)))
    f = probe(lg(P), y, 2, 0, epochs=30, lr=0.01)
    return lambda Q: f(lg(Q))[:, 1]


def balanced(y, p, k):
    return float(np.mean([(p[y == c] == c).mean() for c in range(k) if (y == c).any()]))


def select(tr):
    F = {k: tr[k].astype(np.float32) for k in ('high', 'mid', 'low')}; y = tr['tumour'].astype(int); slides = tr['slide']
    ws = sorted(set(slides)); fold_of = {w: i % 5 for i, w in enumerate(np.random.default_rng(0).permutation(ws))}; fold = np.array([fold_of[s] for s in slides])
    P = np.zeros((len(y), len(EXPERTS)))
    for k in range(5):
        ex = Experts({n: v[fold != k] for n, v in F.items()}, y[fold != k]); P[fold == k] = ex.predict({n: v[fold == k] for n, v in F.items()})[0]
    rows = [dict(kind='expert', name=e, cv_auc=round(auc(y, P[:, i]), 4), cv_balanced=round(100*balanced(y, (P[:, i] > 0.5).astype(int), 2), 2)) for i, e in enumerate(EXPERTS)]
    for g in GATES:
        s = np.zeros(len(y))
        for k in range(5):
            s[fold == k] = gate_fit(g, P[fold != k], y[fold != k])(P[fold == k])
        rows.append(dict(kind='gate', name=g, cv_auc=round(auc(y, s), 4), cv_balanced=round(100*balanced(y, (s > 0.5).astype(int), 2), 2)))
    best = max((r for r in rows if r['kind'] == 'gate'), key=lambda r: (r['cv_auc'], r['cv_balanced']))
    return rows, best, P, y


def region_maps(ex, gate, split, q):
    f = np.load(FEATS/split/f"{q['id']}.npz"); n = SCALES['high'][1]
    pos = {s: np.linspace(0, REGION-SCALES[s][0], SCALES[s][1]).round().astype(int) for s in SCALES}
    hi = f['high_idx']
    if len(hi) == 0:
        return np.full((n, n), np.nan), np.full((n, n, len(EXPERTS)), np.nan), np.zeros((n, n))
    cy = pos['high'][hi//n]+192; cx = pos['high'][hi % n]+192
    feats = {'high': f['high'].astype(np.float32)}
    for s in ('mid', 'low'):  # the encoded tile of this scale whose centre is nearest to each high-power field
        size, m = SCALES[s]; idx = f[s+'_idx']; c = np.stack([pos[s][idx//m]+size//2, pos[s][idx % m]+size//2], 1)
        if len(idx) == 0:
            feats[s] = feats['high']; continue
        near = np.argmin((cy[:, None]-c[None, :, 0])**2+(cx[:, None]-c[None, :, 1])**2, 1); feats[s] = f[s][near].astype(np.float32)
    P, cnt = ex.predict(feats); M = gate(P)
    grid, E, C = np.full((n, n), np.nan), np.full((n, n, len(EXPERTS)), np.nan), np.zeros((n, n))
    grid[hi//n, hi % n] = M; E[hi//n, hi % n] = P; C[hi//n, hi % n] = cnt
    return grid, E, C


def read_region(grid, tau, mpp):
    hot = np.nan_to_num(grid) >= tau
    if not hot.any():
        return 0, 0, None, 0.0
    lab, k = ndimage.label(hot, structure=np.ones((3, 3))); objs = ndimage.find_objects(lab)
    pitch = (REGION-384)/(SCALES['high'][1]-1)
    ext = [(max(o[0].stop-o[0].start, o[1].stop-o[1].start)-1)*pitch*mpp+384*mpp for o in objs]
    j = int(np.argmax(ext)); e = ext[j]; size = int((lab == j+1).sum())
    cls = 1 if e <= 200 else 2 if e <= 2000 else 3
    return cls, size, objs[j], e


def main():
    tr = np.load(FEATS/'train.npz'); rows, best, oof_P, _ = select(tr)
    for r in rows:
        print(json.dumps(r), flush=True)
    F = {k: tr[k].astype(np.float32) for k in ('high', 'mid', 'low')}; y = tr['tumour'].astype(int)
    ex = Experts(F, y); gate = gate_fit(best['name'], oof_P, y)  # avg / geo have no parameters; stack is fit on out-of-fold expert outputs
    qs = [json.loads(x) for x in open(RUN/'inputs/queries_c16_test.jsonl')]; maps = [region_maps(ex, gate, 'test', q) for q in qs]
    np.savez(RUN/'feats/maps_test.npz', ids=np.array([q['id'] for q in qs]), grid=np.stack([m[0] for m in maps]), experts=np.stack([m[1] for m in maps]))
    (RUN/'expert_selection.json').write_text(json.dumps(dict(protocol='5-fold slide-grouped CV on expert-train tiles', rows=rows, selected=best), indent=2)+'\n')
    print('done; gate', best['name'], flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args(); main()
