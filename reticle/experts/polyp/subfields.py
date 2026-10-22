"""Sub-field MUSK features of the RETICLE polyp expert (UniToPatho-MS) and the MUSK feature helpers shared by the other experts.
A whole 800 um region at 384 px (~2.1 um/px) is too coarse for the nuclear detail that separates low- from high-grade dysplasia, so
each region is also encoded as 2 x 2 sub-tiles (906 px -> 384, ~1.0 um/px) and 4 x 4 sub-tiles (453 px -> 384, ~0.52 um/px) with
frozen MUSK-V (no ms_aug). The 7 mm regions are read as 800 um tiles (cached in cache/u7_tiles; tissue >= 0.3).
  python -m reticle.experts.polyp.subfields features                                  (800 um train and test regions)
  python -m reticle.experts.polyp.subfields features_u7 --shard i --nshards n          (tiles of the 7 mm test regions)
  python -m reticle.experts.polyp.subfields features_u7train --shard i --nshards n     (tiles of the 7 mm training regions)"""
import argparse

import numpy as np
import torch

from reticle.experts.polyp.probes import RUN, read_jsonl

OUT = RUN/'hires'


def subtiles(a):  # uint8 HxWx3 (an 800 um region) -> 20 x 384 x 384 x 3: 2x2 then 4x4 grid
    from PIL import Image
    out = []
    for k in (2, 4):
        s = a.shape[0]//k
        for i in range(k):
            for j in range(k):
                out.append(np.asarray(Image.fromarray(a[i*s:(i+1)*s, j*s:(j+1)*s]).resize((384, 384), Image.BICUBIC)))
    return np.stack(out)


class Regions(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return torch.from_numpy(subtiles(np.asarray(im.convert('RGB'))))


class Tiles(torch.utils.data.Dataset):  # 800 um tiles cut from the native 7 mm arrays
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, x, y = self.items[i]
        return torch.from_numpy(subtiles(np.asarray(np.load(path, mmap_mode='r')[y:y+1812, x:x+1812])))


def encoder():
    from reticle.encoders import musk_encoder
    return musk_encoder().model.cuda().eval()


def encode(model, x, ms_aug=False):  # uint8 NHWC tensor -> float16 features
    x = x.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        return model(image=x, with_head=False, out_norm=True, ms_aug=ms_aug)[0].half().cpu()


def features():
    OUT.mkdir(exist_ok=True); model = encoder()
    for split in ('train', 'test'):
        f = OUT/f'u8_{split}.npz'
        if f.exists():
            continue
        qs = read_jsonl(RUN/f'inputs/queries_u8_{split}.jsonl'); feats = []
        dl = torch.utils.data.DataLoader(Regions([q['image_path'] for q in qs]), batch_size=8, num_workers=16)
        for x in dl:
            feats.append(encode(model, x.flatten(0, 1)).view(len(x), 20, -1))
        z = torch.cat(feats).numpy()
        np.savez(f, ids=np.array([q['id'] for q in qs]), t2=z[:, :4], t4=z[:, 4:])
        print(split, z.shape, flush=True)


def features_u7(shard, nshards):  # tissue tiles (>= 0.3) of the 7 mm test regions, region feature (ms_aug) and sub-tiles
    OUT.mkdir(exist_ok=True); f = OUT/f'u7_test.{shard}.npz'
    if f.exists():
        return
    qs = read_jsonl(RUN/'inputs/queries_u7_test.jsonl'); items, cached = [], []
    for qi, q in enumerate(qs):
        c = np.load(RUN/'cache/u7_tiles'/f"{q['id'].replace(':', '_')}.npz"); tiles = c['tiles']
        for ti, ((x, y), t) in enumerate(zip(c['coords'], c['tissue'])):
            if t >= 0.3:
                items.append((qi, ti, q['native'], x, y)); cached.append(tiles[ti])
    mine = list(range(shard, len(items), nshards)); model = encoder()
    region = torch.cat([encode(model, torch.from_numpy(np.stack([cached[i] for i in mine[k:k+64]])), ms_aug=True) for k in range(0, len(mine), 64)])
    dl = torch.utils.data.DataLoader(Tiles([items[i][2:] for i in mine]), batch_size=8, num_workers=16)
    z = torch.cat([encode(model, x.flatten(0, 1)).view(len(x), 20, -1) for x in dl]).numpy()
    np.savez(f, qidx=np.array([items[i][0] for i in mine]), tidx=np.array([items[i][1] for i in mine]), region=region.numpy(), t2=z[:, :4], t4=z[:, 4:])
    print('u7 test shard', shard, z.shape, flush=True)


class TrainRegions(torch.utils.data.Dataset):  # 7 mm training regions: decode the source PNG once, yield the sub-tiles of its tissue tiles
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        qi, path, tiles = self.items[i]
        with Image.open(path) as im:
            a = np.asarray(im.convert('RGB'))
        return qi, np.array([t for t, _, _ in tiles]), torch.from_numpy(np.stack([subtiles(a[y:y+1812, x:x+1812]) for _, x, y in tiles]))


def features_u7train(shard, nshards):  # the same features for the 7 mm training regions (used to fit the 7 mm aggregation)
    import csv
    from reticle.benchmarks.unitopatho_ms import EXT, SRC
    OUT.mkdir(exist_ok=True); f = OUT/f'u7_train.{shard}.npz'
    if f.exists():
        return
    rows = list(csv.DictReader(open(SRC/'metadata/original/unitopath-public/7000/train.csv'))); items, cached = [], {}
    for qi, r in enumerate(rows):
        if qi % nshards != shard:
            continue
        c = np.load(RUN/'cache/u7_tiles'/f'U7000_train_{qi}.npz'); keep = [ti for ti, t in enumerate(c['tissue']) if t >= 0.3]
        items.append((qi, str(EXT/'7000'/r['top_label_name']/r['image_id']), [(ti, *c['coords'][ti]) for ti in keep])); cached[qi] = c['tiles'][keep]
    model = encoder(); out = dict(qidx=[], tidx=[], region=[], t2=[], t4=[])
    dl = torch.utils.data.DataLoader(TrainRegions(items), batch_size=None, num_workers=8, prefetch_factor=1)
    for k, (qi, tidx, x) in enumerate(dl):
        if len(tidx) == 0:
            continue
        z = torch.cat([encode(model, x[j:j+8].flatten(0, 1)).view(-1, 20, 1024) for j in range(0, len(x), 8)]).numpy()
        out['region'].append(encode(model, torch.from_numpy(cached[int(qi)]), ms_aug=True).numpy()); out['qidx'].append(np.full(len(tidx), int(qi)))
        out['tidx'].append(tidx.numpy() if hasattr(tidx, 'numpy') else tidx); out['t2'].append(z[:, :4]); out['t4'].append(z[:, 4:])
        if k % 20 == 0:
            print('u7 train shard', shard, k, '/', len(items), flush=True)
    np.savez(f, **{k: np.concatenate(v) for k, v in out.items()})
    print('u7 train shard', shard, 'done', flush=True)


def balanced(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['features', 'features_u7', 'features_u7train'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    features() if a.mode == 'features' else (features_u7 if a.mode == 'features_u7' else features_u7train)(a.shard, a.nshards)
