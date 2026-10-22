"""The RETICLE mitosis expert for MIDOG-MS: cross-fitted fine-tuned MUSK mitosis detectors. Model k of round r is fine-tuned on the
expert-train cells of case folds != k (recipe: configs/experts/mitosis_detectors.json; model 0 of round 1 is finetune.py's detector).
Round 2 adds hard negatives mined from the round-1 out-of-fold maps and trains on random 192 px views up to +-64 px off the cell centre,
as on the inference grid; round 3 adds the non-breast MIDOG++ cases as training data. Training regions get the out-of-fold model's map,
dev / test regions the mean of the 5 models.
The reader maps a probability map to the four count options. Readers (count rule, Poisson-binomial count distribution, 4-class head on
map statistics), maps (fine-tuned ensemble, or its mean with the frozen mixture of counter.py) and rounds 1..max_round are compared by
case-grouped 5-fold CV balanced accuracy on the 360-region pool (dev + expert-train regions, out-of-fold maps). A rule reading becomes
probabilities through its out-of-fold confusion matrix under a uniform class prior, then a temperature fitted on the pool. Writes the
expert records (runs/contract_v1/mdg<max_round+1>_*.jsonl) and the evidence the agent reads (evidence_mdg_test_cv<max_round>.jsonl).
The benchmark uses max_round 3 (records mdg4, chosen reader: round-3 ensemble averaged with the frozen mixture, rule); the demo's live
tool uses max_round 2 (records mdg3 and cv_read2.json, chosen reader: round-2 ensemble, rule).
  python -m reticle.experts.mitosis.crossfit train --round 1 --fold k                (k = 1..4; fold 0 is finetune.py; one GPU each)
  python -m reticle.experts.mitosis.crossfit apply --round r --fold k --split train|dev|test [--shard i --nshards n]
  python -m reticle.experts.mitosis.crossfit mine                                    (after round 1 apply on train)
  python -m reticle.experts.mitosis.crossfit extra                                   (round-3 MIDOG++ cells)
  python -m reticle.experts.mitosis.crossfit train --round 2|3 --fold k
  python -m reticle.experts.mitosis.crossfit read --max-round 2|3"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from reticle.benchmarks.midog_ms import GOLD, RUN
from reticle.experts.metastasis.tile_experts import auc
from reticle.experts.mitosis.counter import L, bin_of, peaks
from reticle.experts.mitosis.features import FEATS, Grid
from reticle.experts.mitosis.finetune import Cells, folds, model, predict
from reticle.experts.polyp.probes import to_input
from reticle.paths import CONFIGS

CFG = json.loads((CONFIGS/'experts/mitosis_detectors.json').read_text())


def fold_dir(r, k):
    return RUN/'ft' if (r, k) == (1, 0) else RUN/f'cv{r}/f{k}'


PP = RUN.parent.parent/'datasets/midogpp'  # MIDOG++ (CC0): only its non-breast cases are used (round 3)


def cells(r):
    rows = [json.loads(x) for x in open(RUN/'cells_train.jsonl')]
    if r >= 2:
        rows += [json.loads(x) for x in open(RUN/'cv_mined.jsonl')]
    if r >= 3:  # other tumour types and species: always training data, never in a validation fold
        rows += [json.loads(x) for x in open(RUN/'cells_extra.jsonl')]
    return sorted(rows, key=lambda x: x['case'])


def prepare_extra():
    """Round 3 training cells from MIDOG++ cases 201-553 (canine lung cancer, lymphoma, mast cell tumour, soft tissue sarcoma; human
    neuroendocrine tumour, melanoma): every consensus mitotic figure (label 1) and imposter (label 0), plus random background points per
    case. The human breast cases 001-200 (MIDOG 2021, which MIDOG-MS is cut from) are excluded."""
    import random
    c = CFG['midogpp']; n_random = c['random_points_per_case']
    coco = json.loads((PP/'meta/MIDOG++.json').read_text()); ims = {i['id']: i for i in coco['images'] if i['id'] >= c['first_case'] and (PP/'images'/i['file_name']).exists()}
    rows = []
    for a in coco['annotations']:
        if a['image_id'] in ims:
            x0, y0, x1, y1 = a['bbox']; rows.append(dict(case=a['image_id'], cx=(x0+x1)/2, cy=(y0+y1)/2, label=int(a['category_id'] == 1), kind='annotated', src=str(PP/'images'/ims[a['image_id']]['file_name'])))
    for cid, im in sorted(ims.items()):
        rng = random.Random(f'midogpp-bg-{cid}')
        for _ in range(n_random):
            rows.append(dict(case=cid, cx=rng.uniform(200, im['width']-200), cy=rng.uniform(200, im['height']-200), label=0, kind='random', src=str(PP/'images'/im['file_name'])))
    (RUN/'cells_extra.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    print('extra cases', len(ims), 'cells', len(rows), 'mitoses', sum(x['label'] for x in rows), 'imposters', sum(1 for x in rows if x['kind'] == 'annotated' and not x['label']), flush=True)


def native_crops(rows, size=CFG['views']['crop']):
    """size x size native crops centred on the cells (edges clamped), read case by case."""
    import tifffile
    from reticle.benchmarks.midog_ms import SRC
    out, img, cur = [None]*len(rows), None, None
    for i in sorted(range(len(rows)), key=lambda j: rows[j]['case']):
        r = rows[i]
        if r['case'] != cur:
            img, cur = tifffile.imread(r.get('src') or SRC/f"images/{r['case']:03d}.tiff"), r['case']
            img = img[..., :3] if img.ndim == 3 else np.stack([img]*3, -1)
        y0 = int(min(max(r['cy']-size//2, 0), img.shape[0]-size)); x0 = int(min(max(r['cx']-size//2, 0), img.shape[1]-size))
        out[i] = np.ascontiguousarray(img[y0:y0+size, x0:x0+size])
    return out


def jitter(crops, rng, view=CFG['views']['view'], shift=CFG['views']['max_offset']):
    """Random 192 px cell views inside the 320 px crops (centre offset up to +-64 px), resized to 384 as in features.views."""
    from PIL import Image
    c = (crops.shape[1]-view)//2; out = []
    for a in crops:
        dy, dx = rng.integers(-shift, shift+1, 2); v = a[c+dy:c+dy+view, c+dx:c+dx+view]
        out.append(np.asarray(Image.fromarray(np.ascontiguousarray(v)).resize((384, 384), Image.BICUBIC)))
    return np.stack(out)


def train(r, k):
    off = CFG['seed_offset'].get(f'{r}/{k}', 0)  # a fold whose training diverged was retrained once with another seed
    out = fold_dir(r, k); out.mkdir(parents=True, exist_ok=True); torch.manual_seed(k+off); fo = folds(); c = CFG['training']
    if (out/'best.pt').exists():
        return
    rows = cells(r); rows_tr = [x for x in rows if fo.get(x['case'], -1) != k]; rows_va = [x for x in rows if fo.get(x['case'], -1) == k and x['kind'] == 'annotated']
    va = list(torch.utils.data.DataLoader(Cells(rows_va), batch_size=64, num_workers=0)); yv = np.array([x['label'] for x in rows_va])
    m = model(); head = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith('head.')]
    body = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith('head.')]
    opt = torch.optim.AdamW([dict(params=body, lr=c['lr_backbone']), dict(params=head, lr=c['lr_head'])], weight_decay=c['weight_decay'])
    y_all = np.array([x['label'] for x in rows_tr]); w = torch.tensor([len(y_all)/(2*max(1, (y_all == c).sum())) for c in (0, 1)], dtype=torch.float32).cuda()
    bs = c['batch_size']; steps = c['epochs']*(len(rows_tr)//bs); warm = steps//10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1, steps-warm))))
    rng = np.random.default_rng(k+off); log = (out/'train_log.jsonl').open('w'); data = Cells(rows_tr)
    if r == 1:
        X = torch.stack([data[i][0] for i in range(len(data))]); Y = torch.tensor([data[i][1] for i in range(len(data))])
    else:  # round 2: 320 px native crops, a random 192 px cell view (offset up to +-64 px, as on the inference grid) per step
        X = np.stack(native_crops(rows_tr)); Y = torch.tensor([x['label'] for x in rows_tr])
    for epoch in range(c['epochs']):
        perm = rng.permutation(len(X)); m.train()
        for s in range(0, len(perm)-bs+1, bs):
            idx = perm[s:s+bs]; x = X[idx].numpy() if r == 1 else jitter(X[idx], rng); x = np.rot90(x, rng.integers(4), axes=(1, 2))
            if rng.random() < .5:
                x = x[:, :, ::-1]
            opt.zero_grad(set_to_none=True)
            for h in (slice(0, bs//2), slice(bs//2, bs)):  # two half-batches per step
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(to_input(x[h])).float(), Y[idx][h].cuda(), weight=w)/2
                loss.backward()
            opt.step(); sched.step()
        m.eval(); p = np.concatenate([predict(m, xb.numpy()) for xb, _ in va])  # logged only; the epoch count is fixed
        log.write(json.dumps(dict(epoch=epoch+1, fold=k, auc_vs_lookalikes=round(auc(yv, p), 4)))+'\n'); log.flush(); print('fold', k, 'epoch', epoch+1, round(auc(yv, p), 4), flush=True)
    torch.save({n: t.detach().to(torch.bfloat16).cpu() for n, t in m.state_dict().items()}, out/'best.pt'); print('trained', r, k, flush=True)


def queries(split, k=None):
    qs = [json.loads(x) for x in open(RUN/f'inputs/queries_mdg_{split}.jsonl')]
    if split == 'train' and k is not None:
        gold = {g['id']: g for g in map(json.loads, open(GOLD))}; fo = folds(); qs = [q for q in qs if fo[gold[q['id']]['case']] == k]
    return qs


def apply(r, k, split, shard, nshards):
    d = fold_dir(r, k); m = model(); m.load_state_dict({n: v.float() for n, v in torch.load(d/'best.pt', map_location='cpu').items()}); m.eval()
    (d/split).mkdir(parents=True, exist_ok=True)
    for q in queries(split, k if split == 'train' else None)[shard::nshards]:
        out = d/split/f"{q['id']}.npz"
        if out.exists():
            continue
        ds = Grid(q['native']); dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8)
        p = np.concatenate([predict(m, x[:, 0].numpy()) for x in dl]) if len(ds) else np.zeros(0)
        np.savez(out, pts=np.array([ds.pts[j] for j in ds.keep]).reshape(-1, 2), p=p)
    print('applied', r, k, split, shard, flush=True)


def ft_map(r, split, qid, case=None):
    """Fine-tuned probability map: out-of-fold model for training regions, mean of the 5 models otherwise."""
    if split == 'train':
        f = np.load(fold_dir(r, folds()[case])/'train'/f'{qid}.npz'); return f['pts'], f['p']
    fs = [np.load(fold_dir(r, k)/split/f'{qid}.npz') for k in range(5)]
    return fs[0]['pts'], np.mean([f['p'] for f in fs], 0)


def mine():
    """Hard negatives from round-1 out-of-fold training-region maps: peaks >= 0.5 farther than 64 px from every annotated mitotic figure,
    strongest 40 per case, converted to case-image coordinates (the region's quadrant origin is found by matching its pixels)."""
    import tifffile
    from reticle.benchmarks.midog_ms import SRC
    gold = {g['id']: g for g in map(json.loads, open(GOLD))}; per_case = {}; hn = CFG['hard_negatives']
    for q in queries('train'):
        g = gold[q['id']]; pts, p = ft_map(1, 'train', q['id'], g['case']); mit = np.array(g['mitosis_xy']).reshape(-1, 2)
        for v, y, x in peaks(pts, p, hn['min_probability']):
            if not len(mit) or np.min(np.hypot(mit[:, 0]-x, mit[:, 1]-y)) > hn['min_distance_px']:
                per_case.setdefault(g['case'], []).append((v, q['id'], y, x))
    rows = []
    for case, cand in sorted(per_case.items()):
        img = tifffile.imread(SRC/f'images/{case:03d}.tiff')[..., :3]; Hc, Wc = img.shape[:2]; org = {}
        for v, qid, y, x in sorted(cand, reverse=True)[:hn['per_case']]:
            if qid not in org:
                a = np.load(RUN/'native'/f'{qid}.npy', mmap_mode='r')
                org[qid] = next((qy*Hc//2, qx*Wc//2) for qy in range(2) for qx in range(2) if np.array_equal(img[qy*Hc//2:qy*Hc//2+16, qx*Wc//2:qx*Wc//2+16], a[:16, :16]))
            rows.append(dict(case=case, cx=float(x+org[qid][1]), cy=float(y+org[qid][0]), label=0, kind='mined', p=round(float(v), 4)))
    (RUN/'cv_mined.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows)); print('mined', len(rows), 'cases', len(per_case), flush=True)


def match(pk, mit, radius=96):
    """Greedy matching of peaks (strongest first) to annotated mitoses within radius -> label per peak, number of unmatched mitoses."""
    used, lab = set(), []
    for v, y, x in pk:
        d = [(math.hypot(mx-x, my-y), j) for j, (mx, my) in enumerate(mit) if j not in used]; d = [t for t in d if t[0] <= radius]
        if d:
            used.add(min(d)[1]); lab.append(1)
        else:
            lab.append(0)
    return np.array(lab), len(mit)-len(used)


def pb_bins(q, mu):
    """Option probabilities of count = Poisson-binomial(q) + Poisson(mu), bins 0 / 1 / 2-3 / >= 4."""
    dist = np.zeros(64); dist[0] = 1.0
    for v in q:
        dist[1:] = dist[1:]*(1-v)+dist[:-1]*v; dist[0] *= (1-v)
    pois = np.array([math.exp(-mu)*mu**n/math.factorial(n) for n in range(64)]); c = np.convolve(dist, pois)[:64]
    return np.array([c[0], c[1], c[2]+c[3], max(0.0, 1-c[:4].sum())])


def map_feats(pts, p):
    pk = peaks(pts, p, 0.3); v = np.array([x[0] for x in pk]) if pk else np.zeros(1)
    return np.array([np.log1p(len(peaks(pts, p, t))) for t in (0.3, 0.5, 0.7, 0.9)]+[v.max(), np.sort(v)[-3:].mean(), np.log1p(v.sum()), len(pts)/1000], np.float32)


def read(max_round):
    """Choose the reader on the pool among rounds 1..max_round, calibrate it and write records, evidence and rows (suffix max_round)."""
    sfx = str(max_round)
    from reticle.experts.contract import apply_temperature, fit_temperature, record, report, write
    from reticle.experts.build_contracts import head_cv, logp
    from reticle.experts.mitosis.counter import EXPERTS, Experts, gate_fit, region_probs
    gold = {g['id']: g for g in map(json.loads, open(GOLD))}; fo = folds()
    tr = np.load(FEATS/'train.npz'); Fz = {'cell': tr['cell'].astype(np.float32), 'ctx': tr['ctx'].astype(np.float32)}; yz = tr['label'].astype(int)
    fz = np.array([fo[c] for c in tr['case']]); sel = json.loads((RUN/'expert_selection.json').read_text())['selected']['name']
    Pz = np.zeros((len(yz), len(EXPERTS)))
    for k in range(5):
        Pz[fz == k] = Experts({n: v[fz != k] for n, v in Fz.items()}, yz[fz != k]).predict({n: v[fz == k] for n, v in Fz.items()})
    frozen = {None: (Experts(Fz, yz), gate_fit(sel, Pz, yz))}
    for k in range(5):
        frozen[k] = (Experts({n: v[fz != k] for n, v in Fz.items()}, yz[fz != k]), gate_fit(sel, Pz[fz != k], yz[fz != k]))
    pool = [('dev', q['id']) for q in queries('dev')]+[('train', q['id']) for q in queries('train')]; test = [('test', q['id']) for q in queries('test')]
    rounds = [r for r in range(1, max_round+1) if all((fold_dir(r, k)/'train').exists() and len(list((fold_dir(r, k)/'test').glob('*.npz'))) == 240 for k in range(5))]
    maps = {}
    for r in rounds:
        for s, i in pool+test:
            case = gold[i]['case']; pts, pf = ft_map(r, s, i, case)
            ex, gate = frozen[fo[case] if s == 'train' else None]; pz_pts, pm, _, shape = region_probs(ex, gate, s, i)
            assert len(pz_pts) == len(pts)
            maps[(r, 'ens', s, i)] = (pts, pf, shape); maps[(r, 'ens+frozen', s, i)] = (pts, (pf+pm)/2 if len(pm) else pf, shape)
    Y = np.array(['ABCD'.index(gold[i]['answer']) for _, i in pool]); groups = np.array([gold[i]['case'] for _, i in pool])
    gs = sorted(set(groups)); fold_of = {v: j % 5 for j, v in enumerate(np.random.default_rng(0).permutation(gs))}; cvf = np.array([fold_of[v] for v in groups])
    bal = lambda y, p: 100*float(np.mean([(p[y == c] == c).mean() for c in range(4) if (y == c).any()]))

    def rule_fit(items, ys):
        sc = lambda t: bal(ys, np.array([bin_of(len(peaks(m[0], m[1], t))) for m in items])); return max(np.round(np.arange(0.3, 0.996, 0.01), 3), key=sc)

    def pb_fit(items, mits):
        V, Lb, miss, npts = [], [], 0, 0
        for m, mit in zip(items, mits):
            pk = peaks(m[0], m[1], 0.05); lab, mi = match(pk, mit); V += [x[0] for x in pk]; Lb += lab.tolist(); miss += mi; npts += len(m[0])
        V, Lb = np.array(V), np.array(Lb); z = np.log(np.clip(V, 1e-4, 1-1e-4)/np.clip(1-V, 1e-4, 1))
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=10.0).fit(z[:, None], Lb); rate = miss/max(1, npts)
        def prob(m):
            pk = peaks(m[0], m[1], 0.05); v = np.array([x[0] for x in pk])
            q = lr.predict_proba(np.log(np.clip(v, 1e-4, 1-1e-4)/np.clip(1-v, 1e-4, 1))[:, None])[:, 1] if len(v) else np.zeros(0)
            return pb_bins(q, rate*len(m[0]))
        return prob

    def readers(key, tr_idx, items_tr, items_te):
        ys = Y[tr_idx]; mits = [gold[pool[j][1]]['mitosis_xy'] for j in tr_idx]
        t = rule_fit(items_tr, ys); rule = np.eye(4)[[bin_of(len(peaks(m[0], m[1], t))) for m in items_te]]
        pbf = pb_fit(items_tr, mits); pb = np.stack([pbf(m) for m in items_te])
        return dict(rule=rule, pb=pb), dict(tau=float(t))

    res, oof = {}, {}
    for r in rounds:
        for mp_ in ('ens', 'ens+frozen'):
            items = [maps[(r, mp_, s, i)] for s, i in pool]; o = {'rule': np.zeros((len(Y), 4)), 'pb': np.zeros((len(Y), 4))}
            for f in range(5):
                tri, tei = np.where(cvf != f)[0], np.where(cvf == f)[0]
                out, _ = readers((r, mp_), tri, [items[j] for j in tri], [items[j] for j in tei])
                for n in out:
                    o[n][tei] = out[n]
            X = np.stack([map_feats(m[0], m[1]) for m in items]); o['head'], _ = head_cv(X, Y, groups.tolist(), 4)
            for n, P in o.items():
                res[f'r{r}/{mp_}/{n}'] = round(bal(Y, P.argmax(1)), 2); oof[f'r{r}/{mp_}/{n}'] = P
            print(r, mp_, {n: res[f'r{r}/{mp_}/{n}'] for n in o}, flush=True)
    best = max(res, key=res.get); r, mp_, n = best.split('/'); r = int(r[1:]); print('chosen', best, res[best], flush=True)
    items = [maps[(r, mp_, s, i)] for s, i in pool]; titems = [maps[(r, mp_, s, i)] for s, i in test]; Yt = np.array(['ABCD'.index(gold[i]['answer']) for _, i in test])
    if n == 'head':
        X = np.stack([map_feats(m[0], m[1]) for m in items]); _, head = head_cv(X, Y, groups.tolist(), 4); Pt = head(np.stack([map_feats(m[0], m[1]) for m in titems])); info = {}
    else:
        out, info = readers((r, mp_), np.arange(len(Y)), items, titems); Pt = out[n]
    if n == 'rule':  # a hard count has no spread: calibrated probabilities from the pool's out-of-fold confusion of the rule
        conf = np.array([[(oof[best][Y == c].argmax(1) == j).sum()+1 for j in range(4)] for c in range(4)], float)
        rec = conf/conf.sum(1, keepdims=True); post = rec/rec.sum(0, keepdims=True)  # uniform class prior
        P_pool = post[:, oof[best].argmax(1)].T; Pt = post[:, Pt.argmax(1)].T
    else:
        P_pool = oof[best]
    T = fit_temperature(logp(P_pool), Y); P_poolc, Ptc = apply_temperature(logp(P_pool), T), apply_temperature(logp(Pt), T)
    write(f'mdg{max_round+1}', 'calib', [record(i, 'mdg', 'mdg_cv_reader', p) for (_, i), p in zip(pool, P_poolc)])
    rows, ev = [], []
    for (s, i), p, m in zip(test, Ptc, titems):
        pts, pmap, shape = m; H, W = int(shape[0]), int(shape[1]); pk = peaks(pts, pmap, 0.3)
        box = lambda y_, x_: [round(1000*(x_-96)/W, 1), round(1000*(y_-96)/H, 1), round(1000*(x_+96)/W, 1), round(1000*(y_+96)/H, 1)]
        rows.append(record(i, 'mdg', 'mdg_cv_reader', p, [dict(box=box(y_, x_), score=v) for v, y_, x_ in pk[:10]]))
        cand = pk[:6] if len(pk) >= 6 else peaks(pts, pmap, 0.1)[:6]; top = np.argsort(-p)
        text = (f'Mitosis detector (ensemble of 5 MUSK models fine-tuned on mitotic figures and look-alikes of other MIDOG cases, each trained without a different fifth '
                f'of the training cases; its reading was calibrated on 360 held-out regions and it can miss or over-count) scanned {len(pts)} tissue points of this region.\n'
                f'Candidates with probability >= 0.30 after removing neighbours: {len(pk)}. '
                f'Strongest candidates (0-1000 coordinates, probability): {", ".join(f"{[round(v) for v in box(yy, xx)]} {v:.2f}" for v, yy, xx in cand) or "none"}.\n'
                f"Detector's calibrated reading: option {L[top[0]]} ({100*p[top[0]]:.0f}%), then {L[top[1]]} ({100*p[top[1]]:.0f}%).")
        ev.append(dict(id=i, text=text, coarse=[], fine=[box(yy, xx) for _, yy, xx in cand], count=len(pk)))
    write(f'mdg{max_round+1}', 'test', rows)
    (RUN/f'evidence_mdg_test_cv{sfx}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ev))
    rep = dict(rounds=rounds, pool_balanced_cv=res, chosen=best, info=info, temperature=T, pool=report(P_poolc, Y), test=report(Ptc, Yt),
               test_balanced_uncalibrated_argmax=round(bal(Yt, Pt.argmax(1)), 2))
    (RUN/f'cv_read{sfx}.json').write_text(json.dumps(rep, indent=2)+'\n'); print(json.dumps(rep), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train', 'apply', 'mine', 'extra', 'read']); ap.add_argument('--round', type=int, default=1)
    ap.add_argument('--fold', type=int, default=0); ap.add_argument('--split', default='test'); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--max-round', type=int, default=2, choices=[2, 3]); a = ap.parse_args()
    {'train': lambda: train(a.round, a.fold), 'apply': lambda: apply(a.round, a.fold, a.split, a.shard, a.nshards), 'mine': mine, 'extra': prepare_extra, 'read': lambda: read(a.max_round)}[a.mode]()
