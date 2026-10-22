"""Caption evidence for PathMMU. Removes library pairs that duplicate PathMMU images or text (decontamination, configs/pathmmu.json), then
lists the 20 most similar library captions for every question (one caption per source) by reciprocal-rank fusion of image->image and
image->caption similarity. Needs the embeddings of reticle.library.encode and the inputs of reticle.benchmarks.pathmmu; writes
runs/pathmmu_v3/evidence_fused_{val,test}.jsonl.
  python3 -m reticle.library.retrieve_captions        (one GPU)"""
import json
from pathlib import Path

import numpy as np
import torch

from reticle.paths import CONFIGS, DATA

BASE = DATA
RUN = BASE/'runs/pathmmu_v3'
CORPUS = BASE/'library/captions/corpus.jsonl'
CFG = json.loads((CONFIGS/'pathmmu.json').read_text())


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().split('\n') if x]


def source_of(row):
    """Coarsest known source: YouTube video, tweet, PubMed figure (subfigure suffix stripped), TCGA patient."""
    name = row['image_path'].rsplit('/', 1)[-1].rsplit('.', 1)[0]
    if row.get('upstream_subset') == 'quilt' and '_image_' in name:
        return 'video:'+name.split('_image_')[0]
    if row.get('upstream_subset') == 'pubmed':
        return 'figure:'+name.split('_')[0]
    return row['source_group']


def topk(query, matrix, k, chunk=200000):
    """Cosine top-k of each query row against a large fp16 matrix on GPU."""
    q = torch.from_numpy(query).cuda().half()
    best_s, best_i = None, None
    for start in range(0, len(matrix), chunk):
        m = torch.from_numpy(matrix[start:start+chunk]).cuda().half()
        s = (q @ m.T).float()
        v, i = s.topk(min(k, s.shape[1]), dim=1)
        i += start
        if best_s is None:
            best_s, best_i = v, i
        else:
            cat_s, cat_i = torch.cat([best_s, v], 1), torch.cat([best_i, i], 1)
            best_s, order = cat_s.topk(k, dim=1)
            best_i = cat_i.gather(1, order)
    return best_s.cpu().numpy(), best_i.cpu().numpy()


def main():
    rows = read_jsonl(CORPUS)
    blocks = sorted((RUN/'musk').glob('block_*.npz'))
    assert len(blocks) == (len(rows)+9999)//10000, 'encoding incomplete'
    parts = [np.load(b) for b in blocks]
    img = np.concatenate([p['img'] for p in parts]); txt = np.concatenate([p['txt'] for p in parts])
    ok = np.concatenate([p['ok'] for p in parts]); ids = np.concatenate([p['image_id'] for p in parts])
    assert len(img) == len(rows) and all(ids[i] == rows[i]['image_id'] for i in range(0, len(rows), 997))

    pm_rows = read_jsonl(RUN/'inputs/pathmmu_images.jsonl'); pm = np.load(RUN/'musk/pathmmu_img.npy')
    # Visual decontamination: max cosine of every library image to any available PathMMU image.
    s, _ = topk(img, pm, 1, chunk=len(pm))  # here the "query" is the library, the matrix is PathMMU
    max_cos = s[:, 0]
    thr = CFG['decontamination']['visual_threshold']
    text_bad = set((RUN/'inputs/text_contaminated_image_ids.txt').read_text().split())
    # Source-level decontamination: a strong duplicate anywhere in a video/figure/tweet removes that whole source.
    sources = [source_of(r) for r in rows]
    bad_sources = {sources[j] for j in np.flatnonzero(max_cos >= CFG['decontamination']['source_threshold'])}
    in_bad_source = np.array([src in bad_sources for src in sources])
    drop = (max_cos >= thr) | in_bad_source | ~ok | np.array([r['image_id'] in text_bad for r in rows])
    hist = {f'>={t}': int((max_cos >= t).sum()) for t in [0.8, 0.85, 0.9, 0.95, 0.98]}

    keep = np.flatnonzero(~drop)
    img_k, txt_k = img[keep], txt[keep]
    pm_index = {r['image_sha256']: n for n, r in enumerate(pm_rows)}
    summary = dict(library=len(rows), kept=int(len(keep)), dropped_visual=int((max_cos >= thr).sum()),
                   contaminated_sources=len(bad_sources), dropped_by_source=int((in_bad_source & (max_cos < thr)).sum()),
                   dropped_decode=int((~ok).sum()), dropped_text=len(text_bad), visual_threshold=thr, max_cos_hist=hist)
    for split in ['val', 'test']:
        queries = read_jsonl(RUN/f'inputs/queries_{split}.jsonl')
        qv = pm[[pm_index[q['image_sha256']] for q in queries]]
        si, ii = topk(qv, img_k, 200)
        st, it = topk(qv, txt_k, 200)
        out_fused = []
        for n, q in enumerate(queries):
            def entry(local, score_img, channel):
                r = rows[keep[local]]
                return dict(image_id=r['image_id'], source_group=sources[keep[local]], dataset=r['dataset'],
                            subset=r.get('upstream_subset'), caption=r['caption'], score=round(float(score_img), 4), channel=channel)
            def diverse(cands, limit=20):
                seen, caps, out = set(), set(), []
                for c in cands:
                    key = c['caption'].strip().lower()[:200]
                    if c['source_group'] in seen or key in caps:
                        continue
                    seen.add(c['source_group']); caps.add(key); out.append(c)
                    if len(out) == limit:
                        break
                return out
            # Reciprocal-rank fusion of image->image and image->caption; displayed score stays the visual cosine.
            visual_cos = {int(l): float(s_) for l, s_ in zip(ii[n], si[n])}
            fused = {}
            for ranking in (ii[n], it[n]):
                for rank, l in enumerate(ranking, 1):
                    fused[int(l)] = fused.get(int(l), 0)+1/(60+rank)
            order = sorted(fused, key=lambda l: -fused[l])
            vis = img_k[order].astype(np.float32) @ qv[n].astype(np.float32)
            out_fused.append(dict(id=q['id'], evidence=diverse([entry(l, visual_cos.get(l, v), 'fused') for l, v in zip(order, vis)])))
        (RUN/f'evidence_fused_{split}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in out_fused))
        summary[f'{split}_queries'] = len(queries)
        summary[f'{split}_top1_cos_median'] = float(np.median(si[:, 0]))
    (RUN/'retrieve_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # no options; --help shows the docstring
    main()
