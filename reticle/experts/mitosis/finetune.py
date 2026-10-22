"""Round-1 fold-0 mitosis detector (crossfit.py uses it as model 0 of round 1, directory runs/midog_ms_v1/ft) and the shared pieces
of every fine-tuned mitosis detector: the network (MUSK-V + a 2-class head on the global token), the case folds, the cell dataset and
inference. MUSK-V is fine-tuned on the ~45 um 'cell' view (192 native px -> 384) of the expert-train cells of case folds 1-4,
class-balanced loss, flips / 90-degree rotations (recipe: configs/experts/mitosis_finetune.json); the epoch with the best fold-0 AUC
against look-alikes is kept. Its region maps are written by crossfit.py like those of the other detectors.
  python -m reticle.experts.mitosis.finetune train
  python -m reticle.experts.mitosis.crossfit apply --round 1 --fold 0 --split train|dev|test      (train: fold-0 regions only)"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reticle.benchmarks.midog_ms import RUN, SRC
from reticle.experts.metastasis.tile_experts import auc
from reticle.experts.mitosis.features import FEATS, views
from reticle.experts.polyp.probes import to_input
from reticle.paths import CONFIGS

OUT = RUN/'ft'
CFG = json.loads((CONFIGS/'experts/mitosis_finetune.json').read_text())


def model():
    from reticle.encoders import musk_encoder
    musk = musk_encoder().model

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.musk = musk; self.head = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 2))

        def forward(self, x):
            return self.head(self.musk(image=x, return_global=True, with_head=False, out_norm=False)[0])
    m = Net().cuda()
    for n, p in m.named_parameters():
        p.requires_grad = not n.startswith(('musk.vision_head', 'musk.language_head', 'musk.logit_scale'))
    return m


def folds():
    tr = np.load(FEATS/'train.npz'); cases = sorted(set(tr['case'].tolist()))
    return {c: i % 5 for i, c in enumerate(np.random.default_rng(0).permutation(cases))}


class Cells(torch.utils.data.Dataset):  # cell views of expert-train cells (cached per case image)
    def __init__(self, rows):
        self.rows, self.cache = rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import tifffile
        r = self.rows[i]
        if r['case'] not in self.cache:
            self.cache = {r['case']: tifffile.imread(SRC/f"images/{r['case']:03d}.tiff")[..., :3]}
        return torch.from_numpy(views(self.cache[r['case']], r['cy'], r['cx'])[0]), r['label']


def predict(m, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        return torch.softmax(m(to_input(x)).float(), -1)[:, 1].cpu().numpy()


def train():
    OUT.mkdir(parents=True, exist_ok=True); torch.manual_seed(0); fo = folds(); c = CFG['training']; v0 = CFG['validation_fold']
    rows = [json.loads(x) for x in open(RUN/'cells_train.jsonl')]
    rows = sorted(rows, key=lambda r: r['case']); rows_tr = [r for r in rows if fo.get(r['case'], -1) != v0]; rows_va = [r for r in rows if fo.get(r['case']) == v0]
    va = list(torch.utils.data.DataLoader(Cells(rows_va), batch_size=64, num_workers=0))
    yv = np.array([r['label'] for r in rows_va]); hard = np.array([r['kind'] == 'annotated' for r in rows_va])
    m = model(); head = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith('head.')]
    body = [p for n, p in m.named_parameters() if p.requires_grad and not n.startswith('head.')]
    opt = torch.optim.AdamW([dict(params=body, lr=c['lr_backbone']), dict(params=head, lr=c['lr_head'])], weight_decay=c['weight_decay'])
    y_all = np.array([r['label'] for r in rows_tr]); w = torch.tensor([len(y_all)/(2*max(1, (y_all == c).sum())) for c in (0, 1)], dtype=torch.float32).cuda()
    epochs, bs = c['epochs'], c['batch_size']; steps = epochs*(len(rows_tr)//bs); warm = steps//10
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/warm if s < warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1, steps-warm))))
    rng = np.random.default_rng(0); best = (-1, 0); log = (OUT/'train_log.jsonl').open('w')
    order = np.argsort([r['case'] for r in rows_tr]); data = Cells([rows_tr[i] for i in order])
    X = torch.stack([data[i][0] for i in range(len(data))]); Y = torch.tensor([data[i][1] for i in range(len(data))])  # ~5k x 384^2 x 3 uint8, fits in RAM
    for epoch in range(epochs):
        perm = rng.permutation(len(X)); m.train()
        for k in range(0, len(perm)-bs+1, bs):
            idx = perm[k:k+bs]; x = X[idx].numpy(); x = np.rot90(x, rng.integers(4), axes=(1, 2))
            if rng.random() < .5:
                x = x[:, :, ::-1]
            opt.zero_grad(set_to_none=True)
            for h in (slice(0, bs//2), slice(bs//2, bs)):  # two half-batches per step
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(to_input(x[h])).float(), Y[idx][h].cuda(), weight=w)/2
                loss.backward()
            opt.step(); sched.step()
        m.eval(); p = np.concatenate([predict(m, xb.numpy()) for xb, _ in va])
        a, ah = auc(yv, p), auc(yv[hard], p[hard]); log.write(json.dumps(dict(epoch=epoch+1, fold0_auc=round(a, 4), fold0_auc_vs_lookalikes=round(ah, 4)))+'\n'); log.flush()
        print('epoch', epoch+1, round(a, 4), round(ah, 4), flush=True)
        if ah > best[0]:
            best = (ah, epoch+1); torch.save({k_: t.detach().to(torch.bfloat16).cpu() for k_, t in m.state_dict().items()}, OUT/'best.pt')
    print('trained; best epoch', best[1], round(best[0], 4), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['train']); ap.parse_args()
    train()
