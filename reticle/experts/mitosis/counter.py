"""Frozen mixture of MUSK probes for MIDOG-MS: mitotic figure vs not (look-alikes and random tissue) on frozen MUSK features
(features.py). Experts: cell (~45 um view), ctx (~90 um view), ms (both concatenated), knn (25-NN vote, cell view); gate (avg / geo /
stack) chosen by 5-fold case-grouped CV AUC against look-alikes on the expert-train cells (expert_selection.json). Its probability map
on the 128 px point grid is averaged with the fine-tuned detectors' map in the 'ens+frozen' readers of crossfit.py (the reported
round-3 reading). Also the counting shared by every mitosis reader: local maxima (8-neighbourhood) of a map >= tau (peaks) and the
count options 0 / 1 / 2-3 / >= 4 (bin_of).
  python -m reticle.experts.mitosis.counter      -> expert_selection.json"""
import argparse
import json

import numpy as np

from reticle.benchmarks.midog_ms import RUN
from reticle.experts.metastasis.tile_experts import auc
from reticle.experts.mitosis.features import FEATS, STEP
from reticle.experts.polyp.mixture import cuda, l2, probe

EXPERTS = ['cell', 'ctx', 'ms', 'knn']
GATES = ['avg', 'geo', 'stack']
L = 'ABCD'


class Experts:
    def __init__(self, F, y, seeds=(0, 1, 2)):
        S = lambda x: [probe(x, y, 2, s) for s in seeds]
        self.p = {'cell': S(F['cell']), 'ctx': S(F['ctx']), 'ms': S(np.concatenate([F['cell'], F['ctx']], 1))}
        self.lib = cuda(l2(F['cell'])); self.y = y; self.prior = np.bincount(y, minlength=2)/len(y)

    def predict(self, F):
        import torch
        m = lambda k, x: np.mean([f(x)[:, 1] for f in self.p[k]], 0)
        out = [m('cell', F['cell']), m('ctx', F['ctx']), m('ms', np.concatenate([F['cell'], F['ctx']], 1))]
        top = (cuda(l2(F['cell'])) @ self.lib.T).topk(25, dim=1); w = torch.exp((top.values-1)/0.05).cpu().numpy(); idx = top.indices.cpu().numpy()
        P = np.stack([(w*(self.y[idx] == c)).sum(1)/self.prior[c] for c in (0, 1)], 1); out.append(P[:, 1]/P.sum(1))
        return np.stack(out, 1)


def gate_fit(kind, P, y):
    lg = lambda Q: np.log(np.clip(np.stack([1-Q, Q], -1), 1e-6, 1)).reshape(len(Q), -1)
    if kind == 'avg':
        return lambda Q: Q.mean(1)
    if kind == 'geo':
        return lambda Q: (lambda a, b: b/(a+b))(np.exp(np.log(np.clip(1-Q, 1e-6, 1)).mean(1)), np.exp(np.log(np.clip(Q, 1e-6, 1)).mean(1)))
    f = probe(lg(P), y, 2, 0, epochs=30, lr=0.01)
    return lambda Q: f(lg(Q))[:, 1]


def peaks(pts, p, tau):  # local maxima on the grid (8-neighbourhood) with probability >= tau
    if len(pts) == 0:
        return []
    key = {(int(y), int(x)): v for (y, x), v in zip(pts, p)}; out = []
    for (y, x), v in zip(pts, p):
        if v >= tau and all(v >= key.get((int(y)+dy*STEP, int(x)+dx*STEP), -1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx):
            out.append((float(v), int(y), int(x)))
    return sorted(out, reverse=True)


def bin_of(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3


def region_probs(ex, gate, split, qid):
    f = np.load(FEATS/split/f'{qid}.npz'); F = {'cell': f['cell'].astype(np.float32), 'ctx': f['ctx'].astype(np.float32)}
    if len(f['pts']) == 0:
        return f['pts'], np.zeros(0), np.zeros((0, len(EXPERTS))), f['shape']
    P = ex.predict(F); return f['pts'], gate(P), P, f['shape']


def main():
    tr = np.load(FEATS/'train.npz'); F = {'cell': tr['cell'].astype(np.float32), 'ctx': tr['ctx'].astype(np.float32)}; y = tr['label'].astype(int)
    cases = sorted(set(tr['case'].tolist())); fold_of = {c: i % 5 for i, c in enumerate(np.random.default_rng(0).permutation(cases))}; fold = np.array([fold_of[c] for c in tr['case']])
    P = np.zeros((len(y), len(EXPERTS)))
    for k in range(5):
        P[fold == k] = Experts({n: v[fold != k] for n, v in F.items()}, y[fold != k]).predict({n: v[fold == k] for n, v in F.items()})
    hard = tr['kind'] == 'annotated'
    rows = [dict(kind='expert', name=e, cv_auc=round(auc(y, P[:, i]), 4), cv_auc_vs_lookalikes=round(auc(y[hard], P[hard, i]), 4)) for i, e in enumerate(EXPERTS)]
    for g in GATES:
        s = np.zeros(len(y))
        for k in range(5):
            s[fold == k] = gate_fit(g, P[fold != k], y[fold != k])(P[fold == k])
        rows.append(dict(kind='gate', name=g, cv_auc=round(auc(y, s), 4), cv_auc_vs_lookalikes=round(auc(y[hard], s[hard]), 4)))
    for r in rows:
        print(json.dumps(r), flush=True)
    best = max((r for r in rows if r['kind'] == 'gate'), key=lambda r: r['cv_auc_vs_lookalikes'])
    (RUN/'expert_selection.json').write_text(json.dumps(dict(protocol='5-fold case-grouped CV on expert-train cells', rows=rows, selected=best), indent=2)+'\n')
    print('done; gate', best['name'], flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__.split('\n')[0]).parse_args(); main()
