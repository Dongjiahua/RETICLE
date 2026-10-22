"""Standard MUSK baselines: the foundation model in its published usage (not an ablation of RETICLE).
  probe    the MUSK paper's linear-probe protocol (MUSK-V features at 384 px with multiscale augmentation, a linear layer, 10 seeds) for
           u8 / u7 / c16 / mdg; u8 reads the 800 um region features of reticle/experts/polyp/probes.py, the other sets the whole question
           image. Seed 0 is also written as the arm '<set>_musk', scored by reticle/evaluation/score_regions.py next to RETICLE
  pathmmu  zero-shot: the option whose MUSK text vector is closest to the MUSK vector of the question image (reticle/library/encode.py);
           arm 'musk', scored by reticle/evaluation/score_pathmmu.py
  mcdx     MultiCaRe-Dx retrieval baseline: the most similar published case figure (MUSK image similarity) names the diagnosis; judged
           like the consultation arms (arm 'musk')
  python -m reticle.evaluation.baselines.musk probe [u8 u7 c16 mdg]          (one GPU)
  python -m reticle.evaluation.baselines.musk pathmmu --split test           (one GPU)
  python -m reticle.evaluation.baselines.musk mcdx --ports ...               (one GPU + a vLLM server for the judge)"""
import argparse
import json

import numpy as np
from PIL import Image, ImageOps
import torch

from reticle.paths import DATA

BASE = DATA
OUT = BASE/'runs/musk_baselines'
SETS = {  # set: (run, gold, train splits, test split, labels)
    'u7': ('unitopatho_ms_v1', 'unitopatho_ms_v1', ['u7_train'], 'u7_test', 'ABCDEF'),
    'c16': ('camelyon16_ms_v1', 'camelyon16_ms_v1', ['c16_trreg', 'c16_dev', 'c16_dev2'], 'c16_test', 'ABCD'),
    'mdg': ('midog_ms_v1', 'midog_ms_v1', ['mdg_train', 'mdg_dev'], 'mdg_test', 'ABCD'),
    'u8': ('unitopatho_ms_v1', 'unitopatho_ms_v1', ['u8_train'], 'u8_test', 'ABCDEF')}


def musk():
    from reticle.encoders import musk_encoder
    return musk_encoder().model.cuda().eval()


def features(model, paths, bs=64):
    out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(paths), bs):
            ims = []
            for p in paths[i:i+bs]:
                im = Image.open(p).convert('RGB'); im = ImageOps.pad(im, (max(im.size),)*2, color=(255, 255, 255)).resize((384, 384), Image.BICUBIC)
                ims.append(np.asarray(im))
            x = torch.from_numpy(np.stack(ims)).cuda().permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)
            out.append(model(image=x, with_head=False, out_norm=True, ms_aug=True)[0].float().cpu().numpy())
    return np.concatenate(out)


def probe_set(model, s):
    run, gd, trains, test, L = SETS[s]; R = BASE/'runs'/run
    gold = {g['id']: g for g in map(json.loads, open(BASE/f'scoring/{gd}/gold.jsonl'))}
    def load(split):
        q = [json.loads(x) for x in open(R/f"inputs/queries_{split.split('_', 1)[0]}_{split.split('_', 1)[1]}.jsonl")]
        return [r['id'] for r in q], [r['image_path'] for r in q]
    tr_ids, tr_paths = [], []
    for sp in trains:
        i, p = load(sp); tr_ids += i; tr_paths += p
    te_ids, te_paths = load(test)
    f = OUT/f'{s}_features.npz'
    if s == 'u8':  # the cached 800 um region features (same MUSK-V input), in query order
        from reticle.experts.polyp.probes import region_features
        xtr, xte = region_features('train'), region_features('test')
    elif f.exists():
        d = np.load(f); xtr, xte = d['train'], d['test']
    else:
        xtr, xte = features(model, tr_paths), features(model, te_paths); np.savez(f, train=xtr, test=xte)
    ytr = torch.tensor([L.index(gold[i]['answer']) for i in tr_ids]); yte = np.array([L.index(gold[i]['answer']) for i in te_ids])
    xtr, xte = torch.tensor(xtr), torch.tensor(xte); res = []
    for seed in range(10):  # the MUSK paper's linear-probe protocol
        torch.manual_seed(seed); np.random.seed(seed)
        lin = torch.nn.Linear(xtr.shape[1], len(L)).cuda(); torch.nn.init.xavier_uniform_(lin.weight); torch.nn.init.zeros_(lin.bias)
        opt = torch.optim.AdamW(lin.parameters(), lr=0.1, weight_decay=0); n = len(xtr); total = 10*((n+255)//256); step = 0
        for epoch in range(10):
            perm = torch.randperm(n)
            for k in range(0, n, 256):
                for g in opt.param_groups:
                    g['lr'] = 0.5*(1+np.cos(np.pi*step/total))*0.1
                idx = perm[k:k+256]
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = torch.nn.functional.cross_entropy(lin(xtr[idx].cuda()), ytr[idx].cuda())
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
        with torch.no_grad():
            pred = lin(xte.cuda()).argmax(1).cpu().numpy()
        res.append(dict(seed=seed, accuracy=round(100*float((pred == yte).mean()), 2),
                        balanced=round(100*float(np.mean([(pred[yte == c] == c).mean() for c in np.unique(yte)])), 2)))
        if seed == 0:
            (R/'qa').mkdir(exist_ok=True)
            (R/'qa'/f'{s}_musk.test.0.jsonl').write_text(''.join(json.dumps(dict(id=i, arm=f'{s}_musk', choice=L[p], finish='stop', output_tokens=0))+'\n' for i, p in zip(te_ids, pred)))
    b = np.array([r['balanced'] for r in res]); a = np.array([r['accuracy'] for r in res])
    out = dict(set=s, train=len(tr_ids), test=len(te_ids), balanced_mean=round(float(b.mean()), 2), balanced_sd=round(float(b.std()), 2),
               accuracy_mean=round(float(a.mean()), 2), seeds=res)
    print(s, json.dumps({k: v for k, v in out.items() if k != 'seeds'}), flush=True)
    return out


def probe(sets):
    OUT.mkdir(parents=True, exist_ok=True); model = musk(); rep = {}
    f = OUT/'probe.json'
    if f.exists():
        rep = json.loads(f.read_text())
    for s in sets:
        rep[s] = probe_set(model, s); f.write_text(json.dumps(rep, indent=2)+'\n')


def pathmmu(split):
    """MUSK zero-shot on PathMMU: the option text closest to the question image (option letters stripped)."""
    from reticle.encoders import musk_encoder
    from reticle.library.encode import encode_texts
    R = BASE/'runs/pathmmu_v3'
    rows = [json.loads(x) for x in open(R/'inputs/pathmmu_images.jsonl')]; img = np.load(R/'musk/pathmmu_img.npy').astype(np.float32)
    at = {r['image_sha256']: n for n, r in enumerate(rows)}
    qs = [json.loads(x) for x in open(R/f'inputs/queries_{split}.jsonl')]
    opts = [o[3:].strip() if len(o) > 2 and o[1] == ')' else o for q in qs for o in q['options']]
    txt = np.asarray(encode_texts(musk_encoder(), opts), dtype=np.float32); out, k = [], 0
    for q in qs:
        s = txt[k:k+len(q['options'])] @ img[at[q['image_sha256']]]; k += len(q['options'])
        out.append(dict(id=q['id'], arm='musk', choice=q['options'][int(np.argmax(s))][0], finish='stop', output_tokens=0))
    (R/'qa').mkdir(exist_ok=True); (R/'qa'/f'musk.{split}.0.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in out))
    print('pathmmu', split, len(out), flush=True)


def mcdx(ports):
    """MUSK retrieval baseline on MultiCaRe-Dx test: the nearest published case (image only) names the diagnosis."""
    import itertools
    from concurrent.futures import ThreadPoolExecutor
    from reticle.agent.consult import Library
    from reticle.evaluation import consultation as M
    lib = Library('cuda')
    items = [json.loads(x) for x in open(M.RUN/'inputs/items.jsonl')]; gold = {g['id']: g for g in map(json.loads, open(M.GOLD))}
    rows = []
    for it in items:
        g = gold[it['id']]; q = lib.image_vector(Image.open(it['image_path']).convert('RGB'))
        cands = lib.cases_by_image(q, 3, {g['article_id']}, 0.90)
        top = cands[0] if cands else None
        ans = dict(answer=('Most similar published case: '+top['title']+'. Figure caption: '+(top.get('caption') or '')[:300]) if top else 'No similar case found.',
                   differential=[dict(diagnosis=c['title'], likelihood='medium', reasoning='nearest published case (MUSK image similarity %.2f)' % c['similarity']) for c in cands])
        rows.append(dict(id=it['id'], arm='musk', turns=[dict(question='t1', answer=ans)]))
    (M.RUN/'answers_musk.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cyc = itertools.cycle(ports)
    def one(r):
        g = gold[r['id']]
        ref = f"Gold diagnosis: {g['final_diagnosis']} (aliases: {', '.join(g['diagnosis_aliases'])}). Published title: {g['title']}. Figure caption: {g['caption'][:600]}"
        t1 = M.chat(next(cyc), ref+'\n\nCandidate answer:\n'+M.render(r['turns'][0]['answer'])+
                    '\n\nanswer_top = the single diagnosis the candidate commits to (here: the diagnosis named by the most similar published case). top1_match = true only if it is the same disease entity as the gold '
                    '(synonyms and more specific subtypes count; a different entity, a broad category such as "tumor" or "inflammation", or no committed '
                    'diagnosis count as false). in_differential = true if the gold entity is the committed diagnosis or one of the listed differential diagnoses.', M.JUDGE_T1)
        return dict(id=r['id'], arm='musk', t1=t1)
    with ThreadPoolExecutor(8*len(ports)) as ex:
        judged = list(ex.map(one, rows))
    (M.RUN/'judged_musk.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in judged))
    t1 = 100*np.mean([j['t1']['top1_match'] for j in judged]); t3 = 100*np.mean([j['t1']['top1_match'] or j['t1']['in_differential'] for j in judged])
    rep = dict(n=len(judged), t1_top1=round(float(t1), 2), t1_top3=round(float(t3), 2))
    (OUT/'mcdx.json').write_text(json.dumps(rep, indent=2)+'\n'); print('mcdx', json.dumps(rep), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['probe', 'pathmmu', 'mcdx']); ap.add_argument('sets', nargs='*', default=['u8', 'u7', 'c16', 'mdg'])
    ap.add_argument('--split', default='test'); ap.add_argument('--ports', default='8931'); a = ap.parse_args()
    {'probe': lambda: probe(a.sets), 'pathmmu': lambda: pathmmu(a.split), 'mcdx': lambda: mcdx([int(p) for p in a.ports.split(',')])}[a.mode]()
