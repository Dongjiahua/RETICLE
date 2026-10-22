"""Frozen MUSK-V features for MIDOG-MS at two scales around a point: 'cell' = 192 native px (~45 um, the cell and its neighbours)
upsampled to 384, 'ctx' = 384 native px (~90 um). Also the point grid (every 128 px, tissue only) shared by every mitosis reader.
train    at the expert-train cells (mitotic figures, look-alikes, random tissue points; cells_train.jsonl) -> feats/train.npz
regions  on the point grid of each region -> feats/<split>/<id>.npz with the point coordinates
  python -m reticle.experts.mitosis.features train
  python -m reticle.experts.mitosis.features regions --split train|dev|test [--shard i --nshards n]"""
import argparse
import json

import numpy as np
import torch
from PIL import Image

from reticle.experts.metastasis.features import tissue
from reticle.benchmarks.midog_ms import RUN, SRC
from reticle.experts.polyp.subfields import encode, encoder

FEATS = RUN/'feats'
STEP = 128


def views(a, cy, cx):  # two crops centred at (cy, cx) of an HxWx3 array, edges clamped -> 2 x 384 x 384 x 3
    out = []
    for size in (192, 384):
        y0 = int(min(max(cy-size//2, 0), a.shape[0]-size)); x0 = int(min(max(cx-size//2, 0), a.shape[1]-size))
        c = np.ascontiguousarray(a[y0:y0+size, x0:x0+size]); out.append(c if size == 384 else np.asarray(Image.fromarray(c).resize((384, 384), Image.BICUBIC)))
    return np.stack(out)


class Cells(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows, self.cache = rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import tifffile
        r = self.rows[i]
        if r['case'] not in self.cache:
            self.cache = {r['case']: tifffile.imread(SRC/f"images/{r['case']:03d}.tiff")[..., :3]}
        v = views(self.cache[r['case']], r['cy'], r['cx'])
        return torch.from_numpy(v), tissue(v[1][::4, ::4])


class Grid(torch.utils.data.Dataset):
    def __init__(self, native):
        self.a = np.load(native, mmap_mode='r'); H, W = self.a.shape[:2]
        self.pts = [(y, x) for y in range(STEP//2, H, STEP) for x in range(STEP//2, W, STEP)]
        small = np.asarray(self.a[::16, ::16]); self.keep = [k for k, (y, x) in enumerate(self.pts) if tissue(small[max(0, y//16-6):y//16+6, max(0, x//16-6):x//16+6]) >= 0.2]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        y, x = self.pts[self.keep[i]]
        return torch.from_numpy(views(self.a, y, x))


def train():
    FEATS.mkdir(parents=True, exist_ok=True); out = FEATS/'train.npz'
    if out.exists():
        return
    rows = sorted([json.loads(x) for x in open(RUN/'cells_train.jsonl')], key=lambda r: r['case']); model = encoder(); feats, tis = [], []
    for x, t in torch.utils.data.DataLoader(Cells(rows), batch_size=32, num_workers=0):
        feats.append(encode(model, x.flatten(0, 1)).view(len(x), 2, -1)); tis.append(t)
    z = torch.cat(feats).numpy(); t = torch.cat(tis).numpy()
    keep = np.array([r['kind'] == 'annotated' or tt >= 0.3 for r, tt in zip(rows, t)])
    np.savez(out, case=np.array([r['case'] for r in rows])[keep], label=np.array([r['label'] for r in rows])[keep], kind=np.array([r['kind'] for r in rows])[keep],
             cell=z[keep, 0], ctx=z[keep, 1])
    print('train cells', int(keep.sum()), 'mitoses', int(np.array([r['label'] for r in rows])[keep].sum()), flush=True)


def regions(split, shard, nshards):
    (FEATS/split).mkdir(parents=True, exist_ok=True); model = encoder()
    qs = [json.loads(x) for x in open(RUN/f'inputs/queries_mdg_{split}.jsonl')][shard::nshards]
    for q in qs:
        out = FEATS/split/f"{q['id']}.npz"
        if out.exists():
            continue
        ds = Grid(q['native']); dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8)
        z = torch.cat([encode(model, x.flatten(0, 1)).view(len(x), 2, -1) for x in dl]).numpy() if len(ds) else np.zeros((0, 2, 1024), np.float16)
        np.savez(out, pts=np.array([ds.pts[k] for k in ds.keep]).reshape(-1, 2), cell=z[:, 0], ctx=z[:, 1], shape=np.array(ds.a.shape[:2]))
        print(split, q['id'], len(ds), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train', 'regions']); ap.add_argument('--split', default='test')
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); a = ap.parse_args()
    train() if a.mode == 'train' else regions(a.split, a.shard, a.nshards)
