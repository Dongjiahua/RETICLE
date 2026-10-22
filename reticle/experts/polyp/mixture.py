"""The RETICLE polyp expert for UniToPatho-MS: a mixture of seven MUSK experts. All experts read frozen MUSK-V features (probes.py,
subfields.py): region = whole 800 um region with ms_aug (~2.1 um/px), t2 = 2x2 sub-tiles (~1.0 um/px), t4 = 4x4 sub-tiles (~0.52 um/px).
Experts (each a different view of the same region):
  low        linear probe, whole region                      (architecture at low power)
  mid        linear probe, mean of 2x2 sub-tiles             (glands at medium power)
  high       linear probe, mean + max of 4x4 sub-tiles       (nuclei at high power)
  tile       linear probe on single 4x4 sub-tiles (region label; region = mean over sub-tiles; also a sub-tile heat map)
  mil        gated-attention MIL over the region + 20 sub-tiles (learns which sub-tiles matter)
  knn        similarity-weighted 25-NN vote over training regions (retrieval expert; neighbours are cited in the evidence)
  typegrade  architecture head (normal / hyperplastic / tubular / tubulovillous; low + mid) x grade head (low / high; high power,
             adenomas only), composed into the 6 options
Gates over expert probabilities: avg, geometric mean, stacking (logistic regression on log-probabilities), input-conditioned softmax
gate. The gate is chosen by 5-fold WSI-grouped CV balanced accuracy on the official training slides; the chosen gate and all
experts are refit on all training regions and applied to the 800 um test regions and to every tissue tile of the 7 mm test regions.
A 7 mm answer combines the tile readings; the aggregation is chosen by CV on the 619 7 mm training regions (agg7).
  python -m reticle.experts.polyp.mixture select | apply | agg7"""
import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from reticle.experts.polyp.probes import GOLD, LABELS, RUN, read_jsonl
from reticle.experts.polyp.subfields import OUT as HIRES, balanced

MOE = RUN/'moe'
EXPERTS = ['low', 'mid', 'high', 'tile', 'mil', 'knn', 'typegrade']
GATES = ['avg', 'geo', 'stack', 'gate']
TYPE_OF = [0, 1, 2, 2, 3, 3]  # NORM, HP, TA, TA, TVA, TVA
GRADE_OF = [-1, -1, 0, 1, 0, 1]


def cuda(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).cuda()


def probe(x, y, k, seed, epochs=10, lr=0.1):  # MUSK paper probe (xavier, AdamW, cosine, batch 256) with class-balanced CE
    torch.manual_seed(seed); x, y = cuda(x), torch.as_tensor(y).cuda()
    lin = torch.nn.Linear(x.shape[1], k).cuda(); torch.nn.init.xavier_uniform_(lin.weight); torch.nn.init.zeros_(lin.bias)
    n = torch.bincount(y, minlength=k).float(); w = n.sum()/(k*n.clamp(min=1))
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=0); total = epochs*((len(x)+255)//256); step = 0
    for _ in range(epochs):
        perm = torch.randperm(len(x), device='cuda')
        for i in range(0, len(x), 256):
            for g in opt.param_groups:
                g['lr'] = 0.5*(1+np.cos(np.pi*step/total))*lr
            idx = perm[i:i+256]; loss = F.cross_entropy(lin(x[idx]), y[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    lin.eval()
    return lambda z: torch.softmax(lin(cuda(z)), -1).detach().cpu().numpy()


class MIL(torch.nn.Module):  # gated attention (Ilse et al. 2018) over 21 tokens: region (projected) + 20 sub-tiles
    def __init__(self, d=1024, h=256, k=6):
        super().__init__()
        self.reg = torch.nn.Linear(2048, d); self.emb = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.GELU(), torch.nn.Dropout(0.25))
        self.a, self.b, self.w = torch.nn.Linear(h, 128), torch.nn.Linear(h, 128), torch.nn.Linear(128, 1); self.out = torch.nn.Linear(h, k)

    def forward(self, region, tiles):
        z = self.emb(torch.cat([self.reg(region)[:, None], tiles], 1))
        att = torch.softmax(self.w(torch.tanh(self.a(z))*torch.sigmoid(self.b(z))).squeeze(-1), 1)
        return self.out((att[..., None]*z).sum(1)), att


def mil(region, tiles, y, seed, epochs=30):
    torch.manual_seed(seed); m = MIL().cuda(); R, T, Y = cuda(region), cuda(tiles), torch.as_tensor(y).cuda()
    n = torch.bincount(Y, minlength=6).float(); w = n.sum()/(6*n.clamp(min=1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); total = epochs*((len(Y)+127)//128); step = 0
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Y), device='cuda')
        for i in range(0, len(Y), 128):
            for g in opt.param_groups:
                g['lr'] = 0.5*(1+np.cos(np.pi*step/total))*1e-3
            idx = perm[i:i+128]; loss = F.cross_entropy(m(R[idx], T[idx])[0], Y[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    m.eval()

    def predict(region, tiles):
        out, att = [], []
        with torch.no_grad():
            for i in range(0, len(region), 512):
                o, a = m(cuda(region[i:i+512]), cuda(tiles[i:i+512])); out.append(torch.softmax(o, -1).cpu().numpy()); att.append(a.cpu().numpy())
        return np.concatenate(out), np.concatenate(att)
    return predict


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x/np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-6)


def knn_feat(region, t2, t4):
    return l2(np.concatenate([l2(region), l2(t2.mean(1)), l2(t4.mean(1))], 1))


def knn(lib, ylib, k=25, temp=0.05):
    L = cuda(lib); prior = np.bincount(ylib, minlength=6)/len(ylib)

    def predict(q, exclude=None):
        S = cuda(q) @ L.T
        top = S.topk(k, dim=1); w = torch.exp((top.values-1)/temp).cpu().numpy(); idx = top.indices.cpu().numpy()
        P = np.zeros((len(q), 6))
        for c in range(6):
            P[:, c] = (w*(ylib[idx] == c)).sum(1)/prior[c]
        return P/P.sum(1, keepdims=True), idx, top.values.cpu().numpy()
    return predict


class Expert:
    """Fits every expert on (region, t2, t4, y); predict(region, t2, t4) -> {name: probs [N,6]} plus extras (sub-tile map, attention)."""
    def __init__(self, region, t2, t4, y, seeds=(0, 1, 2)):
        self.seeds = seeds; S = lambda f: [f(s) for s in seeds]
        self.low = S(lambda s: probe(region, y, 6, s))
        self.mid = S(lambda s: probe(t2.mean(1), y, 6, s))
        self.high = S(lambda s: probe(np.concatenate([t4.mean(1), t4.max(1)], 1), y, 6, s))
        self.tile = S(lambda s: probe(t4.reshape(-1, t4.shape[-1]), np.repeat(y, t4.shape[1]), 6, s))
        self.mil = S(lambda s: mil(region, np.concatenate([t2, t4], 1), y, s))
        self.knn = knn(knn_feat(region, t2, t4), y)
        ty = np.array([TYPE_OF[c] for c in y]); ad = np.array([GRADE_OF[c] >= 0 for c in y]); gy = np.array([max(GRADE_OF[c], 0) for c in y])
        self.type = S(lambda s: probe(np.concatenate([region, t2.mean(1)], 1), ty, 4, s))
        self.grade = S(lambda s: probe(np.concatenate([t4.mean(1), t4.max(1)], 1)[ad], gy[ad], 2, s))

    def predict(self, region, t2, t4):
        avg = lambda fs, x: np.mean([f(x) for f in fs], 0)
        out = dict(low=avg(self.low, region), mid=avg(self.mid, t2.mean(1)), high=avg(self.high, np.concatenate([t4.mean(1), t4.max(1)], 1)))
        sub = avg(self.tile, t4.reshape(-1, t4.shape[-1])).reshape(len(t4), t4.shape[1], 6); out['tile'] = sub.mean(1)
        m = [f(region, np.concatenate([t2, t4], 1)) for f in self.mil]; out['mil'] = np.mean([p for p, _ in m], 0); att = np.mean([a for _, a in m], 0)
        out['knn'], nn_idx, nn_sim = self.knn(knn_feat(region, t2, t4))
        pt = avg(self.type, np.concatenate([region, t2.mean(1)], 1)); pg = avg(self.grade, np.concatenate([t4.mean(1), t4.max(1)], 1))
        out['typegrade'] = np.stack([pt[:, 0], pt[:, 1], pt[:, 2]*pg[:, 0], pt[:, 2]*pg[:, 1], pt[:, 3]*pg[:, 0], pt[:, 3]*pg[:, 1]], 1)
        extras = dict(sub=sub, att=att, nn_idx=nn_idx, nn_sim=nn_sim, type=pt, grade=pg)
        return out, extras


def gate_fit(kind, P, y, seed=0):
    """P: [N, E, 6] expert probabilities -> function mapping [M, E, 6] to mixture probabilities."""
    logp = lambda Q: np.log(np.clip(Q, 1e-6, 1)).reshape(len(Q), -1)
    if kind == 'avg':
        return lambda Q: Q.mean(1)
    if kind == 'geo':
        return lambda Q: (lambda g: g/g.sum(1, keepdims=True))(np.exp(np.log(np.clip(Q, 1e-6, 1)).mean(1)))
    if kind == 'stack':
        f = probe(logp(P), y, 6, seed, epochs=30, lr=0.01)
        return lambda Q: f(logp(Q))
    if kind == 'gate':  # input-conditioned softmax weights over experts; the gate sees all experts' log-probabilities
        torch.manual_seed(seed); E = P.shape[1]; g = torch.nn.Linear(E*6, E).cuda(); torch.nn.init.zeros_(g.weight); torch.nn.init.zeros_(g.bias)
        X, Pt, Y = cuda(logp(P)), cuda(P), torch.as_tensor(y).cuda()
        n = torch.bincount(Y, minlength=6).float(); w = n.sum()/(6*n.clamp(min=1)); opt = torch.optim.AdamW(g.parameters(), lr=0.01, weight_decay=1e-3)
        for _ in range(300):
            mix = (torch.softmax(g(X), -1)[..., None]*Pt).sum(1); loss = F.nll_loss(torch.log(mix.clamp(min=1e-6)), Y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        g.eval()
        return lambda Q: (lambda W: (W[..., None]*Q).sum(1))(torch.softmax(g(cuda(logp(Q))), -1).detach().cpu().numpy())
    raise ValueError(kind)


def train_data():
    gold = {g['id']: g for g in read_jsonl(GOLD)}
    qs = read_jsonl(RUN/'inputs/queries_u8_train.jsonl'); h = np.load(HIRES/'u8_train.npz')
    assert list(h['ids']) == [q['id'] for q in qs]
    y = np.array([LABELS.index(gold[q['id']]['label']) for q in qs]); wsi = np.array([gold[q['id']]['wsi'] for q in qs])
    return np.load(RUN/'linprobe/u8_train.npy').astype(np.float32), h['t2'].astype(np.float32), h['t4'].astype(np.float32), y, wsi, [q['id'] for q in qs]


def folds(wsi):  # 5 WSI-grouped folds of the official training slides
    ws = sorted(set(wsi)); fold_of = {w: int(np.random.default_rng(0).permutation(len(ws))[i] % 5) for i, w in enumerate(ws)}
    return np.array([fold_of[w] for w in wsi])


def select():
    MOE.mkdir(exist_ok=True)
    region, t2, t4, y, wsi, ids = train_data(); fold = folds(wsi)
    P = np.zeros((len(y), len(EXPERTS), 6))
    for k in range(5):
        tr, te = fold != k, fold == k
        ex = Expert(region[tr], t2[tr], t4[tr], y[tr]); out, _ = ex.predict(region[te], t2[te], t4[te])
        P[te] = np.stack([out[e] for e in EXPERTS], 1); print('fold', k, flush=True)
    np.savez(MOE/'oof_train.npz', ids=np.array(ids), P=P, y=y, fold=fold)
    rows = [dict(kind='expert', name=e, cv_balanced=round(100*balanced(y, P[:, i].argmax(1)), 2), cv_accuracy=round(100*float((P[:, i].argmax(1) == y).mean()), 2))
            for i, e in enumerate(EXPERTS)]
    for g in GATES:  # second-level CV on the out-of-fold expert outputs, same folds
        pred = np.zeros(len(y), int)
        for k in range(5):
            f = gate_fit(g, P[fold != k], y[fold != k]); pred[fold == k] = f(P[fold == k]).argmax(1)
        rows.append(dict(kind='gate', name=g, cv_balanced=round(100*balanced(y, pred), 2), cv_accuracy=round(100*float((pred == y).mean()), 2)))
    for r in rows:
        print(json.dumps(r), flush=True)
    best = max((r for r in rows if r['kind'] == 'gate'), key=lambda r: r['cv_balanced'])
    (MOE/'selection.json').write_text(json.dumps(dict(protocol='5-fold WSI-grouped CV on official train regions; experts averaged over 3 seeds; '
                                                       'gates evaluated by a second CV on the out-of-fold expert outputs', rows=rows, selected=best), indent=2)+'\n')
    print('selected', best)


AGGS = ['mean_mix', 'logmean_mix', 'mean_then_geo', 'vote', 'learned']


def agg_features(E, M):  # tiles of one region: E [n, experts, 6], M [n, 6] -> region-level summary for the learned aggregator
    return np.concatenate([M.mean(0), M.max(0), np.quantile(M, 0.9, axis=0), np.bincount(M.argmax(1), minlength=6)/len(M), E.mean(0).reshape(-1)])


def aggregate(kind, E, M):
    if kind == 'mean_mix':
        return M.mean(0)
    if kind == 'logmean_mix':
        g = np.exp(np.log(np.clip(M, 1e-6, 1)).mean(0)); return g/g.sum()
    if kind == 'mean_then_geo':
        g = np.exp(np.log(np.clip(E.mean(0), 1e-6, 1)).mean(0)); return g/g.sum()
    if kind == 'vote':
        return np.bincount(M.argmax(1), minlength=6)/len(M)
    raise ValueError(kind)


def agg7():
    """Choose how 800 um tile opinions become a 7 mm answer, by CV on the 7 mm *training* regions: tile outputs come from experts
    trained without the region's WSI fold (out-of-fold), the learned aggregator is cross-validated over the same WSI folds."""
    region, t2, t4, y, wsi, _ = train_data(); fold = folds(wsi); fold_of = dict(zip(wsi, fold))
    gold = {g['id']: g for g in read_jsonl(GOLD)}; oof = np.load(MOE/'oof_train.npz')
    gate = gate_fit(json.loads((MOE/'selection.json').read_text())['selected']['name'], oof['P'], oof['y'])
    parts = [np.load(p) for p in sorted(HIRES.glob('u7_train.*.npz'))]; cat = lambda k: np.concatenate([p[k] for p in parts])
    qidx, rg, a2, a4 = cat('qidx'), cat('region').astype(np.float32), cat('t2').astype(np.float32), cat('t4').astype(np.float32)
    ids = [f'U7000:train:{q}' for q in qidx]; tf = np.array([fold_of[gold[i]['wsi']] for i in ids])
    E = np.zeros((len(qidx), len(EXPERTS), 6))
    for k in range(5):
        ex = Expert(region[fold != k], t2[fold != k], t4[fold != k], y[fold != k]); out, _ = ex.predict(rg[tf == k], a2[tf == k], a4[tf == k])
        E[tf == k] = np.stack([out[e] for e in EXPERTS], 1); print('u7 train fold', k, flush=True)
    M = gate(E); regs = sorted(set(qidx.tolist()))
    Y = np.array([LABELS.index(gold[f'U7000:train:{q}']['label']) for q in regs]); RF = np.array([fold_of[gold[f'U7000:train:{q}']['wsi']] for q in regs])
    per = {q: (E[qidx == q], M[qidx == q]) for q in regs}
    rows = []
    for kind in AGGS[:-1]:
        pred = np.array([aggregate(kind, *per[q]).argmax() for q in regs])
        rows.append(dict(aggregator=kind, cv_balanced=round(100*balanced(Y, pred), 2), cv_accuracy=round(100*float((pred == Y).mean()), 2)))
    X = np.stack([agg_features(*per[q]) for q in regs]); pred = np.zeros(len(regs), int)
    for k in range(5):
        if (RF == k).any():
            f = np.mean([probe(X[RF != k], Y[RF != k], 6, s, epochs=50, lr=0.01)(X[RF == k]) for s in range(3)], 0); pred[RF == k] = f.argmax(1)
    rows.append(dict(aggregator='learned', cv_balanced=round(100*balanced(Y, pred), 2), cv_accuracy=round(100*float((pred == Y).mean()), 2)))
    for r in rows:
        print(json.dumps(r), flush=True)
    best = max(rows, key=lambda r: r['cv_balanced'])
    t = np.load(MOE/'u7_test_tiles.npz'); probs = []
    learned = [probe(X, Y, 6, s, epochs=50, lr=0.01) for s in range(3)] if best['aggregator'] == 'learned' else None
    for i in range(len(t['ids'])):
        keep = (t['tissue'][i] >= 0.3) & ~np.isnan(t['mixture'][i][:, 0])
        keep = keep if keep.any() else ~np.isnan(t['mixture'][i][:, 0])
        Ei, Mi = t['experts'][i][keep], t['mixture'][i][keep]
        probs.append(np.mean([f(agg_features(Ei, Mi)[None])[0] for f in learned], 0) if learned else aggregate(best['aggregator'], Ei, Mi))
    np.savez(MOE/'u7_test_region.npz', ids=t['ids'], probs=np.stack(probs))
    (MOE/'agg7.json').write_text(json.dumps(dict(protocol='7 mm training regions (619), out-of-fold tile experts, same 5 WSI folds', n_regions=len(regs),
                                                 rows=rows, selected=best), indent=2)+'\n')
    print('selected', best, flush=True)


def u7_tiles():
    qs = read_jsonl(RUN/'inputs/queries_u7_test.jsonl')
    parts = [np.load(p) for p in sorted(HIRES.glob('u7_test.*.npz'))]
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return qs, cat('qidx'), cat('tidx'), cat('region').astype(np.float32), cat('t2').astype(np.float32), cat('t4').astype(np.float32)


def apply():
    sel = json.loads((MOE/'selection.json').read_text())['selected']['name']
    oof = np.load(MOE/'oof_train.npz'); gate = gate_fit(sel, oof['P'], oof['y'])
    region, t2, t4, y, wsi, ids = train_data(); ex = Expert(region, t2, t4, y)
    h = np.load(HIRES/'u8_test.npz'); rt = np.load(RUN/'linprobe/u8_test.npy').astype(np.float32)
    out, extra = ex.predict(rt, h['t2'].astype(np.float32), h['t4'].astype(np.float32)); P = np.stack([out[e] for e in EXPERTS], 1)
    np.savez(MOE/'u8_test.npz', ids=h['ids'], experts=P, mixture=gate(P), sub=extra['sub'], att=extra['att'], nn_idx=extra['nn_idx'], nn_sim=extra['nn_sim'],
             type=extra['type'], grade=extra['grade'], expert_names=np.array(EXPERTS), train_ids=np.array(ids))
    qs, qidx, tidx, rg, a2, a4 = u7_tiles(); n = len(qs)
    out, extra = ex.predict(rg, a2, a4); Pt = np.stack([out[e] for e in EXPERTS], 1); M = gate(Pt)
    caches = [np.load(RUN/'cache/u7_tiles'/f"{q['id'].replace(':', '_')}.npz") for q in qs]
    U = lambda shape: np.full((n, 81)+shape, np.nan)
    E, Mx, Sub, Ty, Gr = U((len(EXPERTS), 6)), U((6,)), U((16, 6)), U((4,)), U((2,))
    E[qidx, tidx], Mx[qidx, tidx], Sub[qidx, tidx], Ty[qidx, tidx], Gr[qidx, tidx] = Pt, M, extra['sub'], extra['type'], extra['grade']
    np.savez(MOE/'u7_test_tiles.npz', ids=np.array([q['id'] for q in qs]), experts=E, mixture=Mx, sub=Sub, type=Ty, grade=Gr,
             coords=np.stack([c['coords'] for c in caches]), tissue=np.stack([c['tissue'] for c in caches]), expert_names=np.array(EXPERTS))
    print('applied gate', sel, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['select', 'apply', 'agg7']); a = ap.parse_args()
    {'select': select, 'apply': apply, 'agg7': agg7}[a.mode]()
