"""CAMELYON16-MS: lymph-node metastasis sizing in large regions, built from CAMELYON16 (CC0; sentinel lymph nodes of breast-cancer
patients with exhaustive tumour annotations).

Each region is 10,240 px at level 1 (4.6-5.0 mm, 0.45-0.49 um/px; the model sees a 2,048 px overview). The question asks for the
largest metastatic deposit, with four options: none / isolated tumour cells (<= 0.2 mm) / micrometastasis (0.2-2 mm) /
macrometastasis (> 2 mm). The label is the bounding-box extent of the largest connected deposit of the official mask (level 3).

Splits: test = official test slides (129); the training slides (50 tumour + 30 normal, meta/selected.txt) are split by slide hash
into expert-train slides (tiles for the experts) and dev slides (regions for prompt checks and reader calibration). Regions are
sampled with fixed seeds before any model output: around the largest deposit and up to two small deposits (<= 2 mm) of each tumour
slide, and at random on tissue for negative slides. Query ids are neutral (no slide or label information).
Extra region sets of the training slides (`python -m reticle.benchmarks.camelyon16_ms <set>`): trreg = regions of the expert-train
slides (reader calibration), dev2 = more dev regions.

Outputs (under RETICLE_DATA): runs/camelyon16_ms_v1/{inputs/queries_c16_<split>.jsonl, images/<id>.png, native/<id>.npy,
tiles_train.jsonl}, scoring/camelyon16_ms_v1/gold.jsonl.
  python -m reticle.benchmarks.camelyon16_ms            (WSI container, CPU)"""
import csv
import hashlib
import json
from multiprocessing import Pool
import random

import numpy as np
from PIL import Image
from scipy import ndimage

from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
BASE = DATA
SRC = BASE/'datasets/camelyon16'
RUN = BASE/'runs/camelyon16_ms_v1'
GOLD = BASE/'scoring/camelyon16_ms_v1/gold.jsonl'
REGION = 10240  # level-1 pixels
OPTIONS = ['A) No metastatic carcinoma', 'B) Isolated tumour cells only (largest deposit 0.2 mm or less)',
           'C) Micrometastasis (largest deposit more than 0.2 mm, up to 2 mm)', 'D) Macrometastasis (largest deposit more than 2 mm)']
LABELS = ['none', 'itc', 'micro', 'macro']
QUESTION = ('This H&E image shows a {w:.1f} mm wide region of a sentinel lymph node section from a patient with breast cancer '
            '(scanned at {m:.2f} um/px). What is the largest metastatic tumour deposit in this region?')


def is_dev(slide):
    return int(hashlib.sha256(slide.encode()).hexdigest(), 16) % 100 < 20


def open_slide(s):
    import tifffile
    import zarr
    z = zarr.open(tifffile.TiffFile(SRC/f'images/{s}.tif').aszarr(), mode='r')
    m = zarr.open(tifffile.TiffFile(SRC/f'masks/{s}_mask.tif').aszarr(), mode='r')
    return z, m


def mpp_of(s):
    import openslide
    return float(openslide.OpenSlide(str(SRC/f'images/{s}.tif')).properties['openslide.mpp-x'])


def largest_deposit(mask3, mpp3):  # mask at level 3 (8x) cut to the region -> (largest extent um, tumour area um2)
    t = mask3 == 2
    if not t.any():
        return 0.0, 0.0
    lab, n = ndimage.label(t, structure=np.ones((3, 3)))
    ext = max(max(sl[0].stop-sl[0].start, sl[1].stop-sl[1].start) for sl in ndimage.find_objects(lab))
    return float(ext*mpp3), float(t.sum()*mpp3*mpp3)


def label_of(ext):
    return 0 if ext == 0 else 1 if ext <= 200 else 2 if ext <= 2000 else 3


def regions_for(args):
    s, split, tumour = args
    rng = random.Random(f'c16-{s}' if split in ('test', 'dev') else f'c16{split}-{s}')
    z, m = open_slide(s); mpp1 = 2*mpp_of(s); n_pos = 6 if split in ('dev2', 'trreg') else 3
    m5 = np.asarray(m[5]); H1, W1 = z[1].shape[:2]; R5 = REGION//16
    tissue = m5 >= 1; out = []

    def place(cy5, cx5):  # region (level-1 top-left) with the point inside its middle 60%
        y = int((cy5 - rng.uniform(0.2, 0.8)*R5)*16); x = int((cx5 - rng.uniform(0.2, 0.8)*R5)*16)
        return min(max(y, 0), H1-REGION), min(max(x, 0), W1-REGION)
    taken = []
    if tumour:
        lab, n = ndimage.label(m5 == 2, structure=np.ones((3, 3))); objs = ndimage.find_objects(lab)
        ext = [max(o[0].stop-o[0].start, o[1].stop-o[1].start)*16*mpp1 for o in objs]
        cen = [((o[0].start+o[0].stop)/2, (o[1].start+o[1].stop)/2) for o in objs]
        order = [int(np.argmax(ext))] if objs else []
        small = [i for i in range(n) if ext[i] <= 2000]; rng.shuffle(small)
        for i in order+small:
            cy, cx = cen[i]
            if any(abs(cy*16-(ty+REGION/2)) < REGION/2 and abs(cx*16-(tx+REGION/2)) < REGION/2 for ty, tx in taken):
                continue  # already inside a chosen region
            taken.append(place(cy, cx))
            if len(taken) == n_pos:
                break
    want_neg = {'test': 0 if tumour else 1, 'dev': 1 if tumour else 2, 'dev2': 3 if tumour else 5, 'trreg': 3 if tumour else 5}[split]
    far = ndimage.binary_dilation(m5 == 2, iterations=int(1000/(16*mpp1))+1) if tumour else np.zeros_like(tissue)
    ys, xs = np.nonzero(tissue & ~far); neg = 0
    for _ in range(200):
        if neg == want_neg or len(ys) == 0:
            break
        k = rng.randrange(len(ys)); y, x = place(ys[k], xs[k]); w5 = m5[y//16:y//16+R5, x//16:x//16+R5]
        if (w5 >= 1).mean() >= 0.25 and not (w5 == 2).any() and not any(abs(y-ty) < REGION/2 and abs(x-tx) < REGION/2 for ty, tx in taken):
            taken.append((y, x)); neg += 1
    for k, (y, x) in enumerate(taken):
        m3 = np.asarray(m[3][y//4:y//4+REGION//4, x//4:x//4+REGION//4]); ext, area = largest_deposit(m3, 4*mpp1)
        out.append(dict(slide=s, split=split, y=y, x=x, mpp=mpp1, largest_um=round(ext, 1), tumour_um2=round(area), label=LABELS[label_of(ext)],
                        tissue=round(float((m5[y//16:y//16+R5, x//16:x//16+R5] >= 1).mean()), 3)))
    return out


def extract(args):
    r, qid = args
    npy, png = RUN/'native'/f'{qid}.npy', RUN/'images'/f'{qid}.png'
    if npy.exists() and png.exists():
        return qid
    z, _ = open_slide(r['slide']); a = np.asarray(z[1][r['y']:r['y']+REGION, r['x']:r['x']+REGION])[..., :3]
    np.save(npy, a); Image.fromarray(a).reduce(5).save(png)  # 10,240 / 5 = 2,048 px overview
    return qid


def tiles_for(s):  # expert-train tiles: level-1 centres of tumour tiles (>=50% tumour in 384 px) and normal tiles (>=1 mm from tumour)
    rng = random.Random(f'c16-tiles-{s}'); z, m = open_slide(s); mpp1 = 2*mpp_of(s); m5 = np.asarray(m[5]); H1, W1 = z[1].shape[:2]
    tum = ndimage.uniform_filter((m5 == 2).astype(np.float32), 24) >= 0.5
    far = ~ndimage.binary_dilation(m5 == 2, iterations=int(1000/(16*mpp1))+1) & (ndimage.uniform_filter((m5 >= 1).astype(np.float32), 24) >= 0.5)
    out = []
    for lab, ok, n in ((1, tum, 80), (0, far, 80 if s.startswith('tumor') else 120)):
        ys, xs = np.nonzero(ok); idx = rng.sample(range(len(ys)), min(n, len(ys)))
        for i in idx:
            cy, cx = int(ys[i]*16+8), int(xs[i]*16+8)
            if 768 <= cy < H1-768 and 768 <= cx < W1-768:
                out.append(dict(slide=s, cy=cy, cx=cx, tumour=lab, mpp=mpp1))
    return out


def main_dev2(split='dev2'):  # an extra region set: dev2 (dev slides) or trreg (expert-train slides)
    sel = [s for s in (SRC/'meta/selected.txt').read_text().split() if not s.startswith('test') and (is_dev(s) if split == 'dev2' else not is_dev(s))]
    ref = {r['image'][:-4]: r for r in csv.DictReader(open(SRC/'meta/reference.csv'))}
    with Pool(16) as pool:
        regs = [r for rs in pool.map(regions_for, [(s, split, s.startswith('tumor')) for s in sel]) for r in rs]
    rng = random.Random(20261020); rng.shuffle(regs); queries, gold = [], []
    for k, r in enumerate(regs):
        qid = f'C16_{split}_{k:03d}'; r['id'] = qid
        queries.append(dict(id=qid, source='CAMELYON16-MS', split=split, question=QUESTION.format(w=REGION*r['mpp']/1000, m=r['mpp']), options=OPTIONS,
                            image_path=str(RUN/'images'/f'{qid}.png'), native=str(RUN/'native'/f'{qid}.npy'), mpp=r['mpp']))
        gold.append(dict(id=qid, split=f'c16_{split}', label=r['label'], answer='ABCD'[LABELS.index(r['label'])], wsi=r['slide'], slide=r['slide'],
                         center=ref[r['slide']]['center'], largest_um=r['largest_um'], tumour_um2=r['tumour_um2'], tissue=r['tissue'], y=r['y'], x=r['x']))
    with Pool(16) as pool:
        list(pool.imap_unordered(extract, [(r, r['id']) for r in regs]))
    (RUN/f'inputs/queries_c16_{split}.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
    old = [g for g in map(json.loads, open(GOLD)) if g['split'] != f'c16_{split}']
    GOLD.write_text(''.join(json.dumps(g)+'\n' for g in old+gold))
    print(split, {l: sum(g['label'] == l for g in gold) for l in LABELS}, 'slides', len(sel), flush=True)


def main():
    for d in ['inputs', 'images', 'native']:
        (RUN/d).mkdir(parents=True, exist_ok=True)
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    ref = {r['image'][:-4]: r for r in csv.DictReader(open(SRC/'meta/reference.csv'))}
    sel = [s for s in (SRC/'meta/selected.txt').read_text().split() if (SRC/f'images/{s}.tif').exists() and (SRC/f'masks/{s}_mask.tif').exists()]
    test = [s for s in sel if s.startswith('test')]; train = [s for s in sel if not s.startswith('test')]
    dev = [s for s in train if is_dev(s)]; fit = [s for s in train if not is_dev(s)]
    print('slides: test', len(test), 'dev', len(dev), 'expert-train', len(fit), flush=True)
    jobs = [(s, 'test', ref[s]['type'] == 'tumor') for s in test]+[(s, 'dev', s.startswith('tumor')) for s in dev]
    with Pool(24) as pool:
        regs = [r for rs in pool.map(regions_for, jobs) for r in rs]
        tiles = [t for ts in pool.map(tiles_for, fit) for t in ts]
    gold, queries = [], {'test': [], 'dev': []}
    rng = random.Random(20261019)
    for split in ('test', 'dev'):
        rs = [r for r in regs if r['split'] == split]; rng.shuffle(rs)  # neutral order: ids carry no slide or label information
        for k, r in enumerate(rs):
            qid = f'C16_{split}_{k:03d}'; r['id'] = qid
            queries[split].append(dict(id=qid, source='CAMELYON16-MS', split=split, question=QUESTION.format(w=REGION*r['mpp']/1000, m=r['mpp']), options=OPTIONS,
                                       image_path=str(RUN/'images'/f'{qid}.png'), native=str(RUN/'native'/f'{qid}.npy'), mpp=r['mpp']))
            gold.append(dict(id=qid, split=f'c16_{split}', label=r['label'], answer='ABCD'[LABELS.index(r['label'])], wsi=r['slide'], slide=r['slide'],
                             center=ref[r['slide']]['center'], largest_um=r['largest_um'], tumour_um2=r['tumour_um2'], tissue=r['tissue'], y=r['y'], x=r['x']))
    with Pool(16) as pool:
        for k, _ in enumerate(pool.imap_unordered(extract, [(r, r['id']) for r in regs])):
            if k % 25 == 0:
                print('extracted', k, '/', len(regs), flush=True)
    for split, qs in queries.items():
        (RUN/f'inputs/queries_c16_{split}.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs))
    GOLD.write_text(''.join(json.dumps(g)+'\n' for g in gold))
    (RUN/'tiles_train.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tiles))
    for split in ('test', 'dev'):
        c = {l: sum(g['label'] == l for g in gold if g['split'] == f'c16_{split}') for l in LABELS}
        print(split, c, flush=True)
    print('expert-train tiles', len(tiles), 'tumour', sum(t['tumour'] for t in tiles))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build CAMELYON16-MS regions, questions and labels.')
    ap.add_argument('set', nargs='?', default='main', choices=['main', 'trreg', 'dev2'],
                    help='main = the test and dev regions and the expert-train tiles; trreg / dev2 = the extra region sets')
    a = ap.parse_args()
    main() if a.set == 'main' else main_dev2(a.set)
