"""Frozen MUSK-V features for CAMELYON16-MS at three magnifications (384 px input, out_norm, no ms_aug), the input of the tile
experts (tile_experts.py):
  high = 384 px at level 1 (~0.45 um/px, ~180 um), mid = 768 px -> 384 (~0.9 um/px), low = 1,536 px -> 384 (~1.8 um/px).
train    crops centred at the expert-train tile centres (runs/camelyon16_ms_v1/tiles_train.jsonl) -> feats/train.npz
regions  grids over each region (high 27 x 27, mid 14 x 14, low 7 x 7 positions), tissue tiles only -> feats/<split>/<id>.npz
  python -m reticle.experts.metastasis.features train
  python -m reticle.experts.metastasis.features regions --split trreg|test [--shard i --nshards n]"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reticle.benchmarks.camelyon16_ms import REGION, RUN, SRC
from reticle.experts.polyp.subfields import encoder, encode

SCALES = {'high': (384, 27), 'mid': (768, 14), 'low': (1536, 7)}  # crop size in level-1 px, grid positions per side
FEATS = RUN/'feats'


def to384(a):
    return a if a.shape[0] == 384 else np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((384, 384), Image.BICUBIC))


class TrainTiles(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows, self.open = rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import tifffile
        import zarr
        r = self.rows[i]
        if r['slide'] not in self.open:
            self.open[r['slide']] = zarr.open(tifffile.TiffFile(SRC/f"images/{r['slide']}.tif").aszarr(), mode='r')[1]
        z = self.open[r['slide']]; big = np.asarray(z[r['cy']-768:r['cy']+768, r['cx']-768:r['cx']+768])[..., :3]
        return torch.from_numpy(np.stack([to384(big[768-192:768+192, 768-192:768+192]), to384(big[384:1152, 384:1152]), to384(big)]))


def tissue(a):
    g = a.astype(np.int16)
    return float(((g.mean(-1) < 215) & ((g.max(-1)-g.min(-1)) > 15)).mean()) if g.size else 0.0


class RegionTiles(torch.utils.data.Dataset):
    def __init__(self, native, scale):
        self.a = np.load(native, mmap_mode='r'); size, n = SCALES[scale]; self.size = size
        pos = np.linspace(0, REGION-size, n).round().astype(int); self.pos = [(y, x) for y in pos for x in pos]
        thumb = np.asarray(Image.open(Path(native.replace('/native/', '/images/').replace('.npy', '.png'))))  # 2,048 px overview for tissue
        f = 2048/REGION
        self.keep = [k for k, (y, x) in enumerate(self.pos) if tissue(thumb[int(y*f):int((y+size)*f), int(x*f):int((x+size)*f)]) >= 0.1]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        y, x = self.pos[self.keep[i]]
        return torch.from_numpy(to384(np.asarray(self.a[y:y+self.size, x:x+self.size])))


def train():
    FEATS.mkdir(exist_ok=True); out = FEATS/'train.npz'
    if out.exists():
        return
    rows = [json.loads(x) for x in open(RUN/'tiles_train.jsonl')]; model = encoder(); feats = []
    dl = torch.utils.data.DataLoader(TrainTiles(rows), batch_size=32, num_workers=16)
    for k, x in enumerate(dl):
        feats.append(encode(model, x.flatten(0, 1)).view(len(x), 3, -1))
        if k % 50 == 0:
            print('train batch', k, '/', len(dl), flush=True)
    z = torch.cat(feats).numpy()
    np.savez(out, slide=np.array([r['slide'] for r in rows]), tumour=np.array([r['tumour'] for r in rows]), high=z[:, 0], mid=z[:, 1], low=z[:, 2])
    print('train', z.shape, flush=True)


def regions(split, shard, nshards):
    (FEATS/split).mkdir(parents=True, exist_ok=True); model = encoder()
    qs = [json.loads(x) for x in open(RUN/f'inputs/queries_c16_{split}.jsonl')][shard::nshards]
    for q in qs:
        out = FEATS/split/f"{q['id']}.npz"
        if out.exists():
            continue
        res = {}
        for s in SCALES:
            ds = RegionTiles(q['native'], s); dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=12)
            f = torch.cat([encode(model, x) for x in dl]).numpy() if len(ds) else np.zeros((0, 1024), np.float16)
            res[s] = f; res[s+'_idx'] = np.array(ds.keep, dtype=int)
        np.savez(out, **res)
        print(split, q['id'], {s: len(res[s]) for s in SCALES}, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train', 'regions']); ap.add_argument('--split', default='test')
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); a = ap.parse_args()
    train() if a.mode == 'train' else regions(a.split, a.shard, a.nshards)
