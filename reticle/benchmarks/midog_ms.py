"""MIDOG-MS: mitotic-figure counting in breast-cancer high-power regions, built from the MIDOG 2021 training set (CC BY 4.0; 150
labelled cases, one ~2 mm^2 region per case cut by a pathologist, 3 scanners: Hamamatsu XR / S360 at 0.23 um/px, Aperio CS2 at
0.25 um/px; consensus labels for mitotic figures and look-alikes). Each case image is split into 2 x 2 quadrants (~0.8 x 0.6 mm,
~3,600 x 2,700 px); question: how many mitotic figures are in the quadrant, 4 options (0 / 1 / 2-3 / 4 or more; bins set on the
training cases only, where they split 84 / 61 / 70 / 85). Mitotic figures are ~10 um, i.e. ~12 px on a 1,024 px overview: counting
needs full-resolution fields. Split per scanner by case hash: 20 test, 5 dev, 25 expert-train cases (test 240 regions).
Query ids are neutral (no case or label information).
Outputs (under RETICLE_DATA): runs/midog_ms_v1/{inputs/queries_mdg_<split>.jsonl, images/<id>.png (2,048 px), native/<id>.npy,
cells_train.jsonl}, scoring/midog_ms_v1/gold.jsonl.
  python -m reticle.benchmarks.midog_ms           (WSI container, CPU)"""
import hashlib
import json
from multiprocessing import Pool
import random

import numpy as np
from PIL import Image

from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
BASE = DATA
SRC = BASE/'datasets/midog21'
RUN = BASE/'runs/midog_ms_v1'
GOLD = BASE/'scoring/midog_ms_v1/gold.jsonl'
MPP = [0.23, 0.23, 0.25]  # scanner of cases 1-50, 51-100, 101-150
OPTIONS = ['A) No mitotic figures', 'B) 1 mitotic figure', 'C) 2 or 3 mitotic figures', 'D) 4 or more mitotic figures']
QUESTION = ('This H&E image shows a {w:.2f} x {h:.2f} mm region of an invasive breast carcinoma (scanned at {m:.2f} um/px). '
            'How many mitotic figures does it contain?')


def bin_of(n):
    return 0 if n == 0 else 1 if n == 1 else 2 if n <= 3 else 3


def split_of(cid, ids):
    sc = (cid-1)//50; same = sorted([c for c in ids if (c-1)//50 == sc], key=lambda c: hashlib.sha256(f'midog-{c}'.encode()).hexdigest()); r = same.index(cid)
    return 'test' if r < 20 else 'dev' if r < 25 else 'train'


def cut(args):
    cid, quads = args
    import tifffile
    a = tifffile.imread(SRC/f'images/{cid:03d}.tiff')
    a = a[..., :3] if a.ndim == 3 else np.stack([a]*3, -1)
    for q in quads:
        y0, y1, x0, x1 = q['box']; r = np.ascontiguousarray(a[y0:y1, x0:x1])
        np.save(RUN/'native'/f"{q['id']}.npy", r)
        im = Image.fromarray(r); s = 2048/max(im.size); im.resize((round(im.width*s), round(im.height*s)), Image.LANCZOS).save(RUN/'images'/f"{q['id']}.png")
    return cid


def main():
    for d in ['inputs', 'images', 'native']:
        (RUN/d).mkdir(parents=True, exist_ok=True)
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    coco = json.loads((SRC/'meta/MIDOG.json').read_text())
    ims = {i['id']: i for i in coco['images'] if i['id'] <= 150 and (SRC/f"images/{i['file_name']}").exists()}
    assert len(ims) == 150, len(ims)
    ann = {}
    for a in coco['annotations']:
        ann.setdefault(a['image_id'], []).append(a)
    regs, cells = {'test': [], 'dev': [], 'train': []}, []
    for cid, im in sorted(ims.items()):
        sp = split_of(cid, ims); W, H = im['width'], im['height']; mpp = MPP[(cid-1)//50]
        for qy in range(2):
            for qx in range(2):
                x0, x1, y0, y1 = qx*W//2, (qx+1)*W//2, qy*H//2, (qy+1)*H//2
                inside = [a for a in ann.get(cid, []) if x0 <= (a['bbox'][0]+a['bbox'][2])/2 < x1 and y0 <= (a['bbox'][1]+a['bbox'][3])/2 < y1]
                mit = [((a['bbox'][0]+a['bbox'][2])/2-x0, (a['bbox'][1]+a['bbox'][3])/2-y0) for a in inside if a['category_id'] == 1]
                look = [((a['bbox'][0]+a['bbox'][2])/2-x0, (a['bbox'][1]+a['bbox'][3])/2-y0) for a in inside if a['category_id'] == 2]
                regs[sp].append(dict(case=cid, scanner=(cid-1)//50, mpp=mpp, box=[y0, y1, x0, x1], mitoses=mit, lookalikes=look))
        if sp == 'train':
            for a in ann.get(cid, []):
                cells.append(dict(case=cid, cx=(a['bbox'][0]+a['bbox'][2])/2, cy=(a['bbox'][1]+a['bbox'][3])/2, label=int(a['category_id'] == 1), kind='annotated'))
            rng = random.Random(f'midog-bg-{cid}')
            for _ in range(60):  # random background points (tissue is checked when the features are extracted)
                cells.append(dict(case=cid, cx=rng.uniform(200, W-200), cy=rng.uniform(200, H-200), label=0, kind='random'))
    gold, jobs = [], {}
    rng = random.Random(20261019)
    for sp, rs in regs.items():
        rng.shuffle(rs); qs = []
        for k, r in enumerate(rs):
            qid = f'MDG_{sp}_{k:03d}'; r['id'] = qid; h, w = r['box'][1]-r['box'][0], r['box'][3]-r['box'][2]
            qs.append(dict(id=qid, source='MIDOG-MS', split=sp, question=QUESTION.format(w=w*r['mpp']/1000, h=h*r['mpp']/1000, m=r['mpp']), options=OPTIONS,
                           image_path=str(RUN/'images'/f'{qid}.png'), native=str(RUN/'native'/f'{qid}.npy'), mpp=r['mpp']))
            n = len(r['mitoses']); gold.append(dict(id=qid, split=f'mdg_{sp}', label=str(bin_of(n)), answer='ABCD'[bin_of(n)], wsi=f"case{r['case']:03d}", case=r['case'],
                                                    scanner=r['scanner'], mitoses=n, mitosis_xy=r['mitoses'], lookalike_xy=r['lookalikes']))
            jobs.setdefault(r['case'], []).append(r)
        (RUN/f'inputs/queries_mdg_{sp}.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
    with Pool(16) as pool:
        for k, _ in enumerate(pool.imap_unordered(cut, list(jobs.items()))):
            if k % 30 == 0:
                print('cut', k, '/', len(jobs), flush=True)
    GOLD.write_text(''.join(json.dumps(g)+'\n' for g in gold))
    (RUN/'cells_train.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cells))
    for sp in regs:
        print(sp, len(regs[sp]), [sum(g['answer'] == L for g in gold if g['split'] == f'mdg_{sp}') for L in 'ABCD'], flush=True)
    print('train cells', len(cells), 'mitoses', sum(c['label'] for c in cells))


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # no options; --help shows the docstring
    main()
