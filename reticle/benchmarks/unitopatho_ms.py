"""UniToPatho-MS: colorectal polyp type and dysplasia grade in large regions, built from UniToPatho (6 classes: normal mucosa,
hyperplastic polyp, tubular / tubulovillous adenoma with low- / high-grade dysplasia), on the official split by whole-slide image
(204 train / 88 test WSIs). Two scales: 800 um regions (1,812 px at 0.44 um/px; 2,399 test regions) and 7 mm regions (15,855 px;
248 test regions), where nuclei are lost on the overview and grading needs full-resolution views.
Query ids are neutral (the original file names contain the label).
Outputs (under RETICLE_DATA, runs/unitopatho_ms_v1): inputs/queries_{u8,u7}_{train,test}.jsonl, cache/u8_<split>.npy (384 px),
cache/u7_tiles/<id>.npz (9 x 9 grid of 800 um tiles at 384 px with coordinates and tissue fraction), images/u7/<id>_overview.png
(2,048 px), native/<id>.npy (test only, for views); scoring/unitopatho_ms_v1/gold.jsonl (answer, label, wsi = bootstrap group).
  python -m reticle.benchmarks.unitopatho_ms      (CPU)"""
import csv
import json
from multiprocessing import Pool

import numpy as np
from PIL import Image

from reticle.paths import DATA

Image.MAX_IMAGE_PIXELS = None
BASE = DATA
SRC = BASE/'datasets/unitopatho/v1'
EXT = SRC/'extracted/unitopath-public'
RUN = BASE/'runs/unitopatho_ms_v1'
GOLD = BASE/'scoring/unitopatho_ms_v1/gold.jsonl'
LABELS = ['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']
OPTIONS = ['A) Normal colonic mucosa', 'B) Hyperplastic polyp', 'C) Tubular adenoma, low-grade dysplasia', 'D) Tubular adenoma, high-grade dysplasia',
           'E) Tubulovillous adenoma, low-grade dysplasia', 'F) Tubulovillous adenoma, high-grade dysplasia']
QUESTION = {'u8': 'This H&E image shows an 800 um wide region of a colorectal polyp biopsy (0.44 um/px). Which category best describes it?',
            'u7': 'This H&E image shows a 7 mm wide region of a colorectal polyp biopsy, downsampled from 0.44 um/px. Which category best describes it?'}
GRID = 9
TILE = 1812


def tissue_fraction(a):
    g = a.astype(np.int16)
    return float(((g.mean(-1) < 215) & ((g.max(-1)-g.min(-1)) > 15)).mean())


def small(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB').resize((384, 384), Image.BICUBIC), dtype=np.uint8)


def big(args):
    path, qid, keep_native = args
    out = RUN/'images/u7'/f'{qid}_overview.png'
    tiles_f = RUN/'cache/u7_tiles'/f'{qid}.npz'
    if out.exists() and tiles_f.exists() and (not keep_native or (RUN/'native'/f'{qid}.npy').exists()):
        return qid
    with Image.open(path) as im:
        a = np.asarray(im.convert('RGB'), dtype=np.uint8)
    Image.fromarray(a).resize((2048, 2048), Image.LANCZOS).save(out)
    pos = np.linspace(0, a.shape[0]-TILE, GRID).round().astype(int)
    tiles, coords, tissue = [], [], []
    for y in pos:
        for x in pos:
            t = a[y:y+TILE, x:x+TILE]
            tiles.append(np.asarray(Image.fromarray(t).resize((384, 384), Image.BICUBIC))); coords.append((x, y)); tissue.append(tissue_fraction(t[::8, ::8]))
    np.savez(tiles_f, tiles=np.stack(tiles), coords=np.array(coords), tissue=np.array(tissue), size=a.shape[0])
    if keep_native:
        np.save(RUN/'native'/f'{qid}.npy', a)
    return qid


def main():
    for d in ['inputs', 'cache/u7_tiles', 'images/u7', 'native']:
        (RUN/d).mkdir(parents=True, exist_ok=True)
    GOLD.parent.mkdir(parents=True, exist_ok=True)
    gold, jobs = [], []
    for scale, key in [('800', 'u8'), ('7000', 'u7')]:
        for split in ['train', 'test']:
            rows = list(csv.DictReader(open(SRC/f'metadata/original/unitopath-public/{scale}/{split}.csv')))
            queries = []
            for n, r in enumerate(rows):
                qid = f'U{scale}:{split}:{n}'
                path = EXT/scale/r['top_label_name']/r['image_id']
                assert path.exists(), path
                q = dict(id=qid, source=f'UniToPatho-{scale}', split=split, question=QUESTION[key], options=OPTIONS)
                queries.append(dict(q, path=str(path)))
                gold.append(dict(id=qid, source=f'UniToPatho-{scale}', split=f'{key}_{split}', image_sha256=r['wsi'], wsi=r['wsi'],
                                 label=r['top_label_name'], answer='ABCDEF'[LABELS.index(r['top_label_name'])]))
            if key == 'u8':
                arr = np.stack(Pool(32).map(small, [q['path'] for q in queries], chunksize=16))
                np.save(RUN/f'cache/u8_{split}.npy', arr)
                (RUN/'images/u8').mkdir(parents=True, exist_ok=True)
                for q in queries:  # neutral file name (the original name holds the label)
                    link = RUN/'images/u8'/f"{q['id'].replace(':', '_')}.png"
                    if not link.exists():
                        link.symlink_to(q['path'])
                    q['image_path'] = str(link)
            else:
                jobs += [(q['path'], q['id'].replace(':', '_'), split == 'test') for q in queries]
                for q in queries:
                    q['image_path'] = str(RUN/'images/u7'/f"{q['id'].replace(':', '_')}_overview.png")
                    q['native'] = str(RUN/'native'/f"{q['id'].replace(':', '_')}.npy") if split == 'test' else None
            for q in queries:
                del q['path']  # the original path holds the label; model-side code reads image_path / native only
            (RUN/f'inputs/queries_{key}_{split}.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in queries))
            print(key, split, len(queries), flush=True)
    GOLD.write_text(''.join(json.dumps(g)+'\n' for g in gold))
    with Pool(24) as pool:
        for k, qid in enumerate(pool.imap_unordered(big, jobs)):
            if k % 100 == 0:
                print('u7 regions', k, '/', len(jobs), flush=True)
    print('done')


if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()   # no options; --help shows the docstring
    main()
