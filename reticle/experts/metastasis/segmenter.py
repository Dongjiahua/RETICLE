"""The RETICLE metastasis expert for CAMELYON16-MS: MUSK-V fine-tuned with a per-token tumour head (24 x 24 tokens of 16 level-1 px,
~7.6 um, per 384 px tile) on the tumour masks of the expert-train slides, in 5 slide folds (the folds of tile_experts.py), so that model k
never sees fold-k slides. Round 2 adds hard negatives mined from the round-1 out-of-fold maps (recipe: configs/experts/metastasis_segmenter.json).
Region maps: 27 x 27 tiles of 384 px (stride 384, last tile flush) -> 640 x 640 token probabilities; trreg regions (cut from the
expert-train slides) use the out-of-fold model, test regions the mean of the 5 models.
The reader turns a map into the four options. Candidates are compared by slide-grouped 5-fold CV balanced accuracy on the 319 trreg
regions (out-of-fold maps): rule = largest 8-connected cluster >= ts -> bounding-box extent -> option; gated = the same inside tile-mixture
fields >= tt; gated_cluster = whole clusters that touch such a field; heads = 4-class probes on segmentation (and tile-map) statistics.
A rule reading becomes probabilities through its out-of-fold confusion matrix under a uniform class prior, then a temperature fitted on
the pool. The chosen reader writes the expert records (runs/contract_v1/c16seg<tag>_{calib,test}.jsonl) and the evidence the agent
reads (evidence_c16_test_seg<tag>.jsonl: text + the strongest clusters as regions).
The agent uses round 2 read with tag 3 (chosen reader: rule, ts 0.95).
  python -m reticle.experts.metastasis.segmenter tiles
  python -m reticle.experts.metastasis.segmenter train --round 1 --fold k                         (k = 0..4, one GPU each)
  python -m reticle.experts.metastasis.segmenter apply --round 1 --fold k --split trreg
  python -m reticle.experts.metastasis.segmenter mine
  python -m reticle.experts.metastasis.segmenter train --round 2 --fold k
  python -m reticle.experts.metastasis.segmenter apply --round 2 --fold k --split trreg|test [--shard i --nshards n]
  python -m reticle.experts.metastasis.segmenter read --round 2 --tag 3"""
import argparse
import json
import math
from multiprocessing import Pool
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from reticle.benchmarks.camelyon16_ms import GOLD, REGION, RUN, is_dev, open_slide
from reticle.experts.metastasis.features import FEATS, tissue
from reticle.experts.polyp.probes import to_input
from reticle.paths import CONFIGS

OUT = RUN/'seg'
TILE, TOK = 384, 16
POS = list(range(0, REGION-TILE+1, TILE))+[REGION-TILE]  # 27 positions, multiples of 16
CFG = json.loads((CONFIGS/'experts/metastasis_segmenter.json').read_text())


def slide_folds():
    ws = sorted(set(np.load(FEATS/'train.npz')['slide'].tolist()))
    return {w: i % 5 for i, w in enumerate(np.random.default_rng(0).permutation(ws))}


def tiles_for(s):
    """Training tiles of one expert-train slide (level-1 centres): tumour boundary (within ~100 um of the mask edge), every small deposit
    (<= 2 mm, jittered), tumour interior, normal tissue 0.2-1 mm from tumour, and far normal tissue. Returns images and token masks."""
    rng = random.Random(f'c16-seg-{s}'); z, m = open_slide(s); m5 = np.asarray(m[5]); H1, W1 = z[1].shape[:2]; t5 = m5 == 2; tis = m5 >= 1
    pts = []
    def pick(mask, n, tag):
        ys, xs = np.nonzero(mask)
        for i in rng.sample(range(len(ys)), min(n, len(ys))):
            pts.append((int(ys[i]*16+rng.randint(0, 15)), int(xs[i]*16+rng.randint(0, 15)), tag))
    if t5.any():
        edge = t5 ^ ndimage.binary_erosion(t5); pick(edge, 70, 'boundary'); pick(ndimage.binary_erosion(t5, iterations=2), 25, 'interior')
        lab, n = ndimage.label(t5, structure=np.ones((3, 3)))
        for o in ndimage.find_objects(lab)[:60]:
            if max(o[0].stop-o[0].start, o[1].stop-o[1].start)*16*0.5 <= 2000:
                pts.append((int((o[0].start+o[0].stop)*8+rng.randint(-200, 200)), int((o[1].start+o[1].stop)*8+rng.randint(-200, 200)), 'small'))
        near = ndimage.binary_dilation(t5, iterations=26) & ~ndimage.binary_dilation(t5, iterations=5) & tis; pick(near, 30, 'near')
        far = ~ndimage.binary_dilation(t5, iterations=130) & tis; pick(far, 40, 'far')
    else:
        pick(tis, 90, 'far')
    imgs, masks, meta = [], [], []
    for cy, cx, tag in pts:
        y0, x0 = cy-TILE//2, cx-TILE//2
        if not (0 <= y0 and y0+TILE <= H1 and 0 <= x0 and x0+TILE <= W1):
            continue
        a = np.asarray(z[1][y0:y0+TILE, x0:x0+TILE])[..., :3]; mk = np.asarray(m[1][y0:y0+TILE, x0:x0+TILE]) == 2
        if tissue(a[::4, ::4]) < 0.1 and not mk.any():
            continue
        imgs.append(a); masks.append(mk.reshape(TILE//TOK, TOK, TILE//TOK, TOK).mean((1, 3)).astype(np.float16)); meta.append(dict(slide=s, cy=cy, cx=cx, kind=tag))
    if not imgs:
        return s, np.zeros((0, TILE, TILE, 3), np.uint8), np.zeros((0, TILE//TOK, TILE//TOK), np.float16), meta
    return s, np.stack(imgs), np.stack(masks), meta


def tiles():
    OUT.mkdir(parents=True, exist_ok=True); fo = slide_folds(); sel = sorted(fo)
    assert not any(is_dev(s) or s.startswith('test') for s in sel)
    with Pool(16) as pool:
        for s, im, mk, meta in pool.imap_unordered(tiles_for, sel):
            np.savez(OUT/f'tiles_{s}.npz', img=im, mask=mk); (OUT/f'tiles_{s}.json').write_text(json.dumps(meta))
            print(s, len(im), {k: sum(x['kind'] == k for x in meta) for k in ('boundary', 'interior', 'small', 'near', 'far')}, flush=True)


def model():
    from reticle.encoders import musk_encoder
    musk = musk_encoder().model

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.musk = musk; self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 1))

        def forward(self, x):  # -> [B, 24, 24] token logits
            t = self.musk(image=x, return_global=False, with_head=False, out_norm=False)[0][:, 1:]
            return self.head(t).squeeze(-1).view(len(x), TILE//TOK, TILE//TOK)
    m = Net().cuda()
    for n, p in m.named_parameters():
        p.requires_grad = not n.startswith(('musk.vision_head', 'musk.language_head', 'musk.logit_scale'))
    return m


def fold_dir(k, rnd):
    return OUT/(f'f{k}' if rnd == 1 else f'r{rnd}f{k}')


def mine_slide(args):
    """Hard negatives of one expert-train slide: clusters of its trreg regions' round-1 out-of-fold map with probability >= 0.8 and no tumour
    in the official mask within ~40 um; the 20 strongest per slide, as 384 px level-1 tiles with their token masks."""
    s, regs = args; z, m = open_slide(s); H1, W1 = z[1].shape[:2]; cand = []; hn = CFG['hard_negatives']
    for g in regs:
        p = np.load(fold_dir(slide_folds()[s], 1)/'trreg'/f"{g['id']}.npz")['p'].astype(np.float32); n = p.shape[0]
        mk = np.asarray(m[1][g['y']:g['y']+REGION, g['x']:g['x']+REGION]) == 2
        tum = mk.reshape(n, TOK, n, TOK).max((1, 3)); near = ndimage.binary_dilation(tum, iterations=hn['tumour_margin_tokens'])
        lab, k = ndimage.label(p >= hn['min_probability'], structure=np.ones((3, 3)))
        for j, o in enumerate(ndimage.find_objects(lab), 1):
            if not (near[o] & (lab[o] == j)).any():
                cy, cx = ndimage.center_of_mass(lab == j); cand.append((float(p[o].max()), int(g['y']+cy*TOK), int(g['x']+cx*TOK)))
    imgs, masks, meta = [], [], []
    for v, cy, cx in sorted(cand, reverse=True)[:hn['per_slide']]:
        y0, x0 = min(max(cy-TILE//2, 0), H1-TILE), min(max(cx-TILE//2, 0), W1-TILE)
        imgs.append(np.asarray(z[1][y0:y0+TILE, x0:x0+TILE])[..., :3]); mk = np.asarray(m[1][y0:y0+TILE, x0:x0+TILE]) == 2
        masks.append(mk.reshape(TILE//TOK, TOK, TILE//TOK, TOK).mean((1, 3)).astype(np.float16)); meta.append(dict(slide=s, cy=cy, cx=cx, kind='mined', p=round(v, 4)))
    return s, imgs, masks, meta, len(cand)


def mine():
    gold = {g['id']: g for g in map(json.loads, open(GOLD))}; by = {}
    for q in queries('trreg'):
        by.setdefault(gold[q['id']]['slide'], []).append(gold[q['id']])
    with Pool(16) as pool:
        for s, im, mk, meta, n in pool.imap_unordered(mine_slide, sorted(by.items())):
            if im:
                np.savez(OUT/f'mined_{s}.npz', img=np.stack(im), mask=np.stack(mk)); (OUT/f'mined_{s}.json').write_text(json.dumps(meta))
            print(s, 'candidates', n, 'kept', len(im), flush=True)


def train(k, rnd):
    d = fold_dir(k, rnd); d.mkdir(parents=True, exist_ok=True); c = CFG['training']
    if (d/'best.pt').exists():
        return
    fo = slide_folds(); torch.manual_seed(k); rng = np.random.default_rng(k)
    X, Y, V, VY = [], [], [], []
    for s in sorted(fo):
        f = np.load(OUT/f'tiles_{s}.npz'); (X if fo[s] != k else V).append(f['img']); (Y if fo[s] != k else VY).append(f['mask'])
        if rnd >= 2 and fo[s] != k and (OUT/f'mined_{s}.npz').exists():  # mined negatives only for training, never for the held-out fold
            f = np.load(OUT/f'mined_{s}.npz'); X.append(f['img']); Y.append(f['mask'])
    X, Y, V, VY = np.concatenate(X), torch.tensor(np.concatenate(Y)).float(), np.concatenate(V), np.concatenate(VY).astype(np.float32)
    pos = float((Y >= 0.5).float().mean()); w1 = min(c['positive_weight_cap'], (1-pos)/max(pos, 1e-3))  # tumour tokens up-weighted (capped)
    m = model(); head = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith('head.')]
    body = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith('head.')]
    opt = torch.optim.AdamW([dict(params=body, lr=c['lr_backbone']), dict(params=head, lr=c['lr_head'])], weight_decay=c['weight_decay'])
    bs = c['batch_size']; steps = c['epochs']*(len(X)//bs); warm = steps//10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1, steps-warm))))
    log = (d/'train_log.jsonl').open('w')
    for epoch in range(c['epochs']):
        perm = rng.permutation(len(X)); m.train()
        for s in range(0, len(perm)-bs+1, bs):
            idx = np.sort(perm[s:s+bs]); r = int(rng.integers(4)); flip = rng.random() < .5
            x = np.rot90(X[idx], r, axes=(1, 2)); y = torch.rot90(Y[idx], r, dims=(1, 2))
            if flip:
                x, y = x[:, :, ::-1], torch.flip(y, dims=(2,))
            opt.zero_grad(set_to_none=True)
            for h in (slice(0, bs//2), slice(bs//2, bs)):  # two half-batches per step
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = m(to_input(np.ascontiguousarray(x[h]))).float()
                t = y[h].cuda(); loss = (F.binary_cross_entropy_with_logits(z, t, reduction='none')*(1+(w1-1)*(t >= 0.5))).mean()/2
                loss.backward()
            opt.step(); sched.step()
        m.eval(); P = predict(m, V); tv = VY >= 0.5; pv = P >= 0.5
        dice = 2*(tv & pv).sum()/max(1, tv.sum()+pv.sum())
        from reticle.experts.metastasis.tile_experts import auc
        sub = np.random.default_rng(0).choice(P.size, min(P.size, 400000), replace=False)
        rec = dict(epoch=epoch+1, fold=k, token_auc=round(auc(tv.reshape(-1)[sub].astype(int), P.reshape(-1)[sub]), 4), dice_at_05=round(float(dice), 4))
        log.write(json.dumps(rec)+'\n'); log.flush(); print(json.dumps(rec), flush=True)
    torch.save({n: t.detach().to(torch.bfloat16).cpu() for n, t in m.state_dict().items()}, d/'best.pt'); print('trained', k, flush=True)


def predict(m, x, bs=64):
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(x), bs):
            out.append(torch.sigmoid(m(to_input(np.ascontiguousarray(x[i:i+bs]))).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, TILE//TOK, TILE//TOK), np.float32)


class RegionTiles(torch.utils.data.Dataset):
    def __init__(self, native):
        self.a = np.load(native, mmap_mode='r'); thumb = np.asarray(Image.open(native.replace('/native/', '/images/').replace('.npy', '.png'))); f = 2048/REGION
        self.pos = [(y, x) for y in POS for x in POS]
        self.keep = [k for k, (y, x) in enumerate(self.pos) if tissue(thumb[int(y*f):int((y+TILE)*f), int(x*f):int((x+TILE)*f)]) >= 0.1]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        y, x = self.pos[self.keep[i]]; return torch.from_numpy(np.ascontiguousarray(self.a[y:y+TILE, x:x+TILE]))


def queries(split, k=None):
    qs = [json.loads(x) for x in open(RUN/f'inputs/queries_c16_{split}.jsonl')]
    if split == 'trreg' and k is not None:
        gold = {g['id']: g for g in map(json.loads, open(GOLD))}; fo = slide_folds(); qs = [q for q in qs if fo[gold[q['id']]['slide']] == k]
    return qs


def apply(k, split, shard, nshards, rnd):
    d = fold_dir(k, rnd); m = model(); m.load_state_dict({n: v.float() for n, v in torch.load(d/'best.pt', map_location='cpu').items()}); m.eval()
    (d/split).mkdir(parents=True, exist_ok=True); n = REGION//TOK
    for q in queries(split, k if split == 'trreg' else None)[shard::nshards]:
        out = d/split/f"{q['id']}.npz"
        if out.exists():
            continue
        ds = RegionTiles(q['native']); canvas, cnt = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
        P = np.concatenate([predict(m, x.numpy()) for x in torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8)]) if len(ds) else []
        for j, p in zip(ds.keep, P):
            y, x = ds.pos[j]; canvas[y//TOK:(y+TILE)//TOK, x//TOK:(x+TILE)//TOK] += p; cnt[y//TOK:(y+TILE)//TOK, x//TOK:(x+TILE)//TOK] += 1
        np.savez_compressed(out, p=(canvas/np.maximum(cnt, 1)).astype(np.float16), seen=cnt > 0)
    print('applied', k, split, shard, flush=True)


def seg_map(split, qid, rnd, slide=None):
    """Token map of a region: the out-of-fold model for trreg regions, the mean of the 5 fold models otherwise."""
    if split == 'trreg':
        return np.load(fold_dir(slide_folds()[slide], rnd)/'trreg'/f'{qid}.npz')['p'].astype(np.float32)
    return np.mean([np.load(fold_dir(k, rnd)/split/f'{qid}.npz')['p'].astype(np.float32) for k in range(5)], 0)


TAUS = (0.3, 0.5, 0.7, 0.9)


def extent(p, tau, mpp):
    """Largest 8-connected cluster of tokens >= tau: (bounding-box extent in um, area in um^2)."""
    t = p >= tau
    if not t.any():
        return 0.0, 0.0
    lab, n = ndimage.label(t, structure=np.ones((3, 3))); objs = ndimage.find_objects(lab)
    sizes = ndimage.sum(t, lab, range(1, n+1)); ext = [max(o[0].stop-o[0].start, o[1].stop-o[1].start) for o in objs]
    return float(max(ext)*TOK*mpp), float(sizes.max()*(TOK*mpp)**2)


def option(ext):
    return 0 if ext == 0 else 1 if ext <= 200 else 2 if ext <= 2000 else 3


def seg_features(p, mpp):
    f = [p.max(), np.sort(p.reshape(-1))[-20:].mean()]
    for tau in TAUS:
        e, a = extent(p, tau, mpp); f += [np.log1p(e), np.log1p(a), np.log1p((p >= tau).sum())]
    return np.array(f, np.float32)


TS = np.round(np.arange(0.3, 0.96, 0.05), 2)  # segmentation thresholds
TT = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)          # tile-mixture gate thresholds (0 = no gate)
FIELD = np.linspace(0, REGION-TILE, 27).round().astype(int)  # the tile mixture's 27 x 27 high-power fields


def gate_up(grid):
    """Tile-mixture field probabilities (27 x 27, NaN = no tissue) -> token canvas (max over the fields covering a token)."""
    n = REGION//TOK; up = np.zeros((n, n), np.float32); g = np.nan_to_num(grid, nan=0.0)
    for i, y in enumerate(FIELD):
        for j, x in enumerate(FIELD):
            sl = (slice(y//TOK, (y+TILE)//TOK), slice(x//TOK, (x+TILE)//TOK)); up[sl] = np.maximum(up[sl], g[i, j])
    return up


def ext_table(p, up, mpp):
    """Largest-cluster extent (um) of (seg >= ts) & (tile mixture >= tt) for every pair of thresholds -> [len(TS), len(TT)]."""
    return np.array([[extent(p*(up >= tt) if tt > 0 else p, ts, mpp)[0] for tt in TT] for ts in TS], np.float32)


def cluster_table(p, up, mpp):
    """Largest extent (um) among whole segmentation clusters (seg >= ts) that contain a token of a tile-mixture field >= tt
    (the cluster is kept or dropped as a whole, so a large deposit is not cut where single fields fall below tt) -> [len(TS), len(TT)]."""
    out = np.zeros((len(TS), len(TT)), np.float32)
    for a, ts in enumerate(TS):
        lab, n = ndimage.label(p >= ts, structure=np.ones((3, 3)))
        if not n:
            continue
        objs = ndimage.find_objects(lab); ext = np.array([max(o[0].stop-o[0].start, o[1].stop-o[1].start)*TOK*mpp for o in objs])
        sup = ndimage.maximum(up, lab, range(1, n+1))
        for b, tt in enumerate(TT):
            keep = sup >= tt; out[a, b] = ext[keep].max() if keep.any() else 0.0
    return out


def balanced_posterior(o, Y, k=4):
    """P(true class | reading) under a uniform class prior (the benchmark's balanced accuracy), from out-of-fold readings o:
    recall matrix P(reading | class) with add-one smoothing, normalised over classes for each reading."""
    conf = np.array([[(o[Y == c] == j).sum()+1 for j in range(k)] for c in range(k)], float); rec = conf/conf.sum(1, keepdims=True)
    return rec/rec.sum(0, keepdims=True)


def read(rnd, tag):
    """Choose the reader on the trreg pool (out-of-fold maps of round rnd), calibrate it, and write the test records and evidence with
    the suffix tag."""
    from reticle.experts.metastasis.tile_experts import Experts, gate_fit, region_maps, select
    from reticle.experts.metastasis.reader import map_features
    from reticle.experts.contract import apply_temperature, fit_temperature, record, report, write
    from reticle.experts.build_contracts import head_cv, logp
    gold = {g['id']: g for g in map(json.loads, open(GOLD))}; fo = slide_folds()
    tr = np.load(FEATS/'train.npz'); Fz = {k: tr[k].astype(np.float32) for k in ('high', 'mid', 'low')}; yz = tr['tumour'].astype(int)
    _, best, oof_P, _ = select(tr); tile_fold = np.array([fo[s] for s in tr['slide']])
    pool = queries('trreg'); Y = np.array(['ABCD'.index(gold[q['id']]['answer']) for q in pool]); G = [gold[q['id']]['slide'] for q in pool]
    M, grids = [None]*len(pool), [None]*len(pool)
    for k in range(5):  # tile-mixture maps, out of fold
        ex = Experts({n: v[tile_fold != k] for n, v in Fz.items()}, yz[tile_fold != k]); gate = gate_fit(best['name'], oof_P[tile_fold != k], yz[tile_fold != k])
        for j, q in enumerate(pool):
            if fo[gold[q['id']]['slide']] == k:
                grid, E, _ = region_maps(ex, gate, 'trreg', q); M[j] = map_features(grid, E, q['mpp']); grids[j] = grid
    M = np.stack(M); maps = [seg_map('trreg', q['id'], rnd, gold[q['id']]['slide']) for q in pool]
    TB = np.stack([ext_table(p, gate_up(g), q['mpp']) for p, g, q in zip(maps, grids, pool)])  # [N, TS, TT]
    TC = np.stack([cluster_table(p, gate_up(g), q['mpp']) for p, g, q in zip(maps, grids, pool)])  # whole-cluster gating
    feats = lambda p, tb, mpp: np.concatenate([seg_features(p, mpp), np.log1p(tb[[2, 4, 6]][:, 1:].reshape(-1))])
    S = np.stack([feats(p, tb, q['mpp']) for p, tb, q in zip(maps, TB, pool)])
    bal = lambda y, p: 100*float(np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()]))
    opt = np.vectorize(option)
    gs = sorted(set(G)); cv = {v: j % 5 for j, v in enumerate(np.random.default_rng(0).permutation(gs))}; cvf = np.array([cv[g] for g in G])

    tabs = {'rule': TB, 'gated': TB, 'gated_cluster': TC}

    def pick(idx, name):  # best (ts, tt) on the given regions
        pairs = [(a, b) for a in range(len(TS)) for b in range(len(TT)) if (b > 0) == (name != 'rule')]
        return max(pairs, key=lambda ab: bal(Y[idx], opt(tabs[name][idx, ab[0], ab[1]])))
    cand, picks = {}, {}
    for name in tabs:
        o = np.zeros(len(Y), int); picks[name] = []
        for f in range(5):
            a, b = pick(np.where(cvf != f)[0], name); picks[name].append((float(TS[a]), TT[b])); o[cvf == f] = opt(tabs[name][cvf == f, a, b])
        cand[name] = np.eye(4)[o]
    cand['head_seg'], h_seg = head_cv(S, Y, G, 4); cand['head_seg+tile'], h_st = head_cv(np.concatenate([S, M], 1), Y, G, 4); cand['head_tile'], _ = head_cv(M, Y, G, 4)
    res = {n: round(bal(Y, P.argmax(1)), 2) for n, P in cand.items()}; print('pool CV', res, 'picks', picks, flush=True)
    chosen = max([n for n in res if n != 'head_tile'], key=res.get)
    rules = ('rule', 'gated', 'gated_cluster'); a, b = pick(np.arange(len(Y)), chosen if chosen in rules else 'gated')  # thresholds also used for the evidence text
    ts, tt = float(TS[a]), TT[b]
    if chosen in rules:
        o = cand[chosen].argmax(1); post = balanced_posterior(o, Y); P_pool = post[:, o].T
    else:
        P_pool = cand[chosen]
    T = fit_temperature(logp(P_pool), Y); out = dict(pool_cv=res, cv_picks=picks, chosen=chosen, ts=ts, tt=tt, temperature=T, pool=report(apply_temperature(logp(P_pool), T), Y))
    sfx = tag
    write('c16seg'+sfx, 'calib', [record(q['id'], 'c16', 'c16_seg_reader', p) for q, p in zip(pool, apply_temperature(logp(P_pool), T))])
    split = 'test'
    mz = np.load(RUN/f'feats/maps_{split}.npz'); mt = {str(i): (g, e) for i, g, e in zip(mz['ids'], mz['grid'], mz['experts'])}; rows, Ys, Ps, ev = [], [], [], []
    for q in queries(split):
        p = seg_map(split, q['id'], rnd); grid, E = mt[q['id']]; up = gate_up(grid); tb = ext_table(p, up, q['mpp'])
        if chosen in rules:
            pr = post[:, option(cluster_table(p, up, q['mpp'])[a, b] if chosen == 'gated_cluster' else tb[a, b])]
        else:
            s = feats(p, tb, q['mpp']); pr = (h_seg(s[None]) if chosen == 'head_seg' else h_st(np.concatenate([s, map_features(grid, E, q['mpp'])])[None]))[0]
        pr = apply_temperature(logp(pr[None]), T)[0]
        if chosen == 'gated_cluster':  # whole clusters supported by a tile-mixture field >= tt
            lab, n = ndimage.label(p >= ts, structure=np.ones((3, 3))); keep = np.where(ndimage.maximum(up, lab, range(1, n+1)) >= tt)[0]+1 if n else []
            pg = p*np.isin(lab, keep)
        else:
            pg = p*(up >= tt) if tt > 0 else p  # clusters shown / cited: segmentation inside the tile-mixture gate
        lab, n = ndimage.label(pg >= ts, structure=np.ones((3, 3))); objs = ndimage.find_objects(lab) if n else []
        order = np.argsort([-max(o_[0].stop-o_[0].start, o_[1].stop-o_[1].start) for o_ in objs])[:5]; sz = p.shape[0]
        rois = [dict(box=[1000*objs[j][1].start/sz, 1000*objs[j][0].start/sz, 1000*objs[j][1].stop/sz, 1000*objs[j][0].stop/sz], score=float(p[objs[j]].max())) for j in order]
        rows.append(record(q['id'], 'c16', 'c16_seg_reader', pr, rois)); Ys.append('ABCD'.index(gold[q['id']]['answer'])); Ps.append(pr)
        kept = ' (whole clusters kept)' if chosen == 'gated_cluster' else ''; e, ar = extent(pg, ts, q['mpp']); top = np.argsort(-pr); shown = ', '.join(f"{[round(v) for v in r_['box']]} {r_['score']:.2f}" for r_ in rois) or 'none'
        text = (f'Tumour expert (mixture of MUSK models: tile classifiers at three magnifications find tumour fields; 5 MUSK models fine-tuned to outline '
                f'metastatic carcinoma at ~8 um resolution measure it; trained on other CAMELYON16 slides, reading calibrated on 319 regions of held-out slides; '
                f'it can miss isolated cells or mark look-alikes). Largest tumour cluster (segmentation >= {ts:.2f} inside fields the tile classifiers call tumour '
                f'with probability >= {tt:.2f}{kept}): extent {e:.0f} um, area {ar/1e6:.3f} mm2; {n} separate clusters. Strongest clusters (0-1000 coordinates, probability): {shown}.\n'
                f"Expert's calibrated reading: option {'ABCD'[top[0]]} ({100*pr[top[0]]:.0f}%), then {'ABCD'[top[1]]} ({100*pr[top[1]]:.0f}%).")
        bx = [[round(v) for v in r_['box']] for r_ in rois[:3]]; ev.append(dict(id=q['id'], text=text, coarse=bx, fine=bx, extent_um=round(e, 1)))
    write('c16seg'+sfx, split, rows); out[split] = report(np.array(Ps), np.array(Ys))
    (RUN/f'evidence_c16_{split}_seg{sfx}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ev))
    print(split, json.dumps(out[split]), flush=True)
    (OUT/f'read{sfx}.json').write_text(json.dumps(out, indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['tiles', 'mine', 'train', 'apply', 'read']); ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--split', default='test'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--tag', default='3', help='suffix of the records, evidence and rows written by read'); a = ap.parse_args()
    {'tiles': tiles, 'mine': mine, 'train': lambda: train(a.fold, a.round), 'apply': lambda: apply(a.fold, a.split, a.shard, a.nshards, a.round),
     'read': lambda: read(a.round, a.tag)}[a.mode]()
