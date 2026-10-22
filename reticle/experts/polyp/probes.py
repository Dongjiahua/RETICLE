"""Shared definitions of the RETICLE polyp expert for UniToPatho-MS (labels, paths, MUSK input) and the whole-region MUSK features.
Region features: MUSK-V (with_head=False, out_norm=True, ms_aug=True -> 2048-d) of each 800 um region at 384 px (the cached
cache/u8_<split>.npy arrays), the input of the 'low' probe and of several other experts of the mixture (mixture.py).
  python -m reticle.experts.polyp.probes features      (one GPU; writes linprobe/u8_train.npy and linprobe/u8_test.npy)"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reticle.paths import DATA

RUN = DATA/'runs/unitopatho_ms_v1'
GOLD = DATA/'scoring/unitopatho_ms_v1/gold.jsonl'
LABELS = ['NORM', 'HP', 'TA.LG', 'TA.HG', 'TVA.LG', 'TVA.HG']
FEATS = RUN/'linprobe'


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().split('\n') if x]


def to_input(a):  # uint8 NHWC -> normalised NCHW on the GPU
    return torch.from_numpy(np.ascontiguousarray(a)).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)


def region_features(split):
    f = FEATS/f'u8_{split}.npy'
    if f.exists():
        return np.load(f)
    from reticle.encoders import musk_encoder
    FEATS.mkdir(parents=True, exist_ok=True); model = musk_encoder().model.cuda().eval()
    arr = np.load(RUN/f'cache/u8_{split}.npy', mmap_mode='r'); out = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        for i in range(0, len(arr), 128):
            x = torch.from_numpy(np.ascontiguousarray(arr[i:i+128])).cuda().permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)
            out.append(model(image=x, with_head=False, out_norm=True, ms_aug=True)[0].float().cpu().numpy())
    x = np.concatenate(out); np.save(f, x)
    return x


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', nargs='?', default='features', choices=['features']); ap.parse_args()
    for split in ('train', 'test'):
        print(split, region_features(split).shape, flush=True)
