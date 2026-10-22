"""MUSK image and caption embeddings of the teaching caption library (and of the PathMMU question images), in resumable 10k-row blocks
(block_*.npz with img, txt, ok, image_id). Read by reticle.agent.library and reticle.library.retrieve_captions. The library manifest
(CORPUS) lists image_id, image_path and caption per pair. Several ranks split the blocks:
  python3 -m reticle.library.encode --rank r --world w [--out DIR]       (one GPU per rank)"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from reticle.encoders import musk_encoder
from reticle.image_io import configure_image_decoder
from reticle.paths import DATA

BASE = DATA
CORPUS = BASE/'library/captions/corpus.jsonl'  # reticle.library.captions
BLOCK = 10000
TRANSFORM = T.Compose([T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize((.5,)*3, (.5,)*3)])


def letterbox(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
    side = max(im.size)
    return ImageOps.pad(im, (side, side), color=(255, 255, 255))


class Images(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        configure_image_decoder()
        try:
            return TRANSFORM(letterbox(self.paths[i])), True
        except Exception:
            return torch.zeros(3, 384, 384), False


@torch.inference_mode()
def encode_images(enc, paths, workers):
    loader = torch.utils.data.DataLoader(Images(paths), batch_size=96, num_workers=workers, pin_memory=True)
    vecs, ok = [], []
    for x, good in loader:
        with torch.autocast('cuda', dtype=torch.float16):
            v = enc.model(image=x.cuda(non_blocking=True), return_global=True, with_head=True, out_norm=False, ms_aug=False)[0]
        vecs.append(F.normalize(v.float(), dim=-1).cpu()); ok += good.tolist()
    return torch.cat(vecs).numpy().astype(np.float16), ok


@torch.inference_mode()
def encode_texts(enc, texts):
    out = []
    for i in range(0, len(texts), 512):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(enc.encode_text(texts[i:i+512]).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--out', default=str(BASE/'runs/pathmmu_v3/musk'))
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    enc = musk_encoder()
    status = out/f'status_{a.rank}.json'

    if a.rank == 0 and not (out/'pathmmu_img.npy').exists():
        rows = [json.loads(x) for x in (BASE/'runs/pathmmu_v3/inputs/pathmmu_images.jsonl').read_text().split('\n') if x]
        vec, ok = encode_images(enc, [r['image_path'] for r in rows], a.workers)
        assert all(ok), 'PathMMU image failed to decode'
        np.save(out/'pathmmu_img.npy', vec)

    rows = [json.loads(x) for x in CORPUS.read_text().split('\n') if x]
    blocks = [b for b in range(0, (len(rows)+BLOCK-1)//BLOCK) if b % a.world == a.rank]
    started = time.monotonic()
    for n, b in enumerate(blocks):
        name = out/f'block_{b:03d}.npz'
        if name.exists():
            continue
        part = rows[b*BLOCK:(b+1)*BLOCK]
        img, ok = encode_images(enc, [r['image_path'] for r in part], a.workers)
        txt = encode_texts(enc, [r['caption'] for r in part])
        tmp = out/f'block_{b:03d}.tmp.npz'
        np.savez(tmp, img=img, txt=txt, ok=np.array(ok), image_id=np.array([r['image_id'] for r in part]))
        tmp.rename(name)
        status.write_text(json.dumps(dict(rank=a.rank, done=n+1, blocks=len(blocks), last=b,
                                          seconds=round(time.monotonic()-started, 1), bad=int(len(ok)-sum(ok)))))
    status.write_text(json.dumps(dict(rank=a.rank, done=len(blocks), blocks=len(blocks), complete=True,
                                      seconds=round(time.monotonic()-started, 1))))


if __name__ == '__main__':
    main()
