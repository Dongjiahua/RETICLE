"""Live RETICLE experts for the demo: the benchmark experts run on a new image instead of reading precomputed outputs.
  mitosis  the cross-fitted mitosis detectors (reticle/experts/mitosis/crossfit.py, round 2: mean map of the 5 fold models, count of map
           peaks >= tau) on a breast-carcinoma high-power region resampled to 0.25 um/px. tau and the calibrated probabilities of each count
           bin are exported from the round-2 read (cv_read2.json, records mdg3), so the live tool reproduces the benchmark expert
           ('verify' checks this on MIDOG-MS test regions).
  polyp    the polyp mixture (reticle/experts/polyp/mixture.py: the seven experts refit on the cached training features, the selected gate
           fitted on their out-of-fold outputs, as in 'apply') on an 800 um region resampled to 0.44 um/px; the text follows
           reticle/experts/polyp/evidence.py.
  python -m reticle.experts.live export                       (CPU: writes runs/demo_v4/expert_tools.json)
  python -m reticle.experts.live verify [n]                   (one GPU: live mitosis tool vs the stored test reading)
  python -m reticle.experts.live verify_polyp [n]             (one GPU: live polyp tool vs the stored test mixture)
  python -m reticle.experts.live mitosis <image> <um_per_px>  (one GPU: prints the expert text)"""
import argparse
import json

import numpy as np
from PIL import Image
import torch

from reticle.paths import DATA

BASE = DATA
CFG = BASE/'runs/demo_v4/expert_tools.json'
MPP = 0.25          # MIDOG-MS native resolution
MAX_UM = 1600       # largest region side read live (um); larger images are centre-cropped
L = 'ABCD'
LABELS = ['0 mitotic figures', '1 mitotic figure', '2-3 mitotic figures', '>= 4 mitotic figures']
L6 = 'ABCDEF'


def export():
    """tau of the round-2 rule reader and the calibrated probabilities of each count bin (from the mdg3 test records)."""
    from reticle.experts.mitosis.crossfit import ft_map, queries
    from reticle.experts.mitosis.counter import bin_of, peaks
    from reticle.benchmarks.midog_ms import RUN
    rd = json.loads((RUN/'cv_read2.json').read_text()); assert rd['chosen'] == 'r2/ens/rule', rd['chosen']
    tau = rd['info']['tau']; rec = {r['id']: r['probs'] for r in map(json.loads, open(BASE/'runs/contract_v1/mdg3_test.jsonl'))}
    by_bin = {}
    for q in queries('test'):
        pts, p = ft_map(2, 'test', q['id']); b = bin_of(len(peaks(pts, p, tau)))
        by_bin.setdefault(b, set()).add(tuple(rec[q['id']]))
    assert all(len(v) == 1 for v in by_bin.values()), {k: len(v) for k, v in by_bin.items()}
    out = dict(mitosis=dict(reader='r2/ens/rule', tau=tau, mpp=MPP, models=[str(RUN/f'cv2/f{k}/best.pt') for k in range(5)],
                            bin_probs={str(b): list(next(iter(v))) for b, v in sorted(by_bin.items())}, source='reticle.experts.mitosis.crossfit read --max-round 2 (cv_read2.json, records mdg3)'))
    CFG.parent.mkdir(parents=True, exist_ok=True); CFG.write_text(json.dumps(out, indent=1)+'\n'); print(json.dumps(out, indent=1))


class MitosisTool:
    def __init__(self):
        self.cfg = json.loads(CFG.read_text())['mitosis']; self.models = None

    def _load(self):
        from reticle.experts.mitosis.finetune import model
        self.models = []
        for f in self.cfg['models']:
            m = model(); m.load_state_dict({n: v.float() for n, v in torch.load(f, map_location='cpu').items()}); self.models.append(m.eval())

    def scan(self, a):
        """a: HxWx3 uint8 at 0.25 um/px -> grid points, mean probability of the 5 fold models (as reticle/experts/mitosis/crossfit.py apply / ft_map)."""
        from reticle.experts.metastasis.features import tissue
        from reticle.experts.mitosis.features import STEP, views
        from reticle.experts.mitosis.finetune import predict
        if self.models is None:
            self._load()
        H, W = a.shape[:2]; pts = [(y, x) for y in range(STEP//2, H, STEP) for x in range(STEP//2, W, STEP)]
        small = np.asarray(a[::16, ::16]); keep = [k for k, (y, x) in enumerate(pts) if tissue(small[max(0, y//16-6):y//16+6, max(0, x//16-6):x//16+6]) >= 0.2]
        pts = [pts[k] for k in keep]
        if not pts:
            return np.zeros((0, 2), int), np.zeros(0)
        P = []
        for m in self.models:
            out = []
            for i in range(0, len(pts), 64):
                out.append(predict(m, np.stack([views(a, y, x)[0] for y, x in pts[i:i+64]])))
            P.append(np.concatenate(out))
        return np.array(pts), np.mean(P, 0)

    def read(self, a):
        from reticle.experts.mitosis.counter import bin_of, peaks
        pts, p = self.scan(a); H, W = a.shape[:2]; pk = peaks(pts, p, 0.3); n = len(peaks(pts, p, self.cfg['tau']))
        probs = np.array(self.cfg['bin_probs'][str(bin_of(n))])
        box = lambda y_, x_: [round(1000*(x_-96)/W, 1), round(1000*(y_-96)/H, 1), round(1000*(x_+96)/W, 1), round(1000*(y_+96)/H, 1)]
        cand = pk[:6] if len(pk) >= 6 else peaks(pts, p, 0.1)[:6]; top = np.argsort(-probs)
        text = (f'Mitosis detector (ensemble of 5 MUSK models fine-tuned on mitotic figures and look-alikes of MIDOG cases; its reading was calibrated on '
                f'360 held-out regions and it can miss or over-count; run live on this image, resampled to {MPP} um/px, {W*MPP:.0f} x {H*MPP:.0f} um) scanned '
                f'{len(pts)} tissue points.\nCandidates with probability >= 0.30 after removing neighbours: {len(pk)}; at the reader threshold '
                f'{self.cfg["tau"]:.2f}: {n}. Strongest candidates (0-1000 coordinates, probability): '
                f'{", ".join(f"{[round(v) for v in box(yy, xx)]} {v:.2f}" for v, yy, xx in cand) or "none"}.\n'
                f"Detector's calibrated reading: option {L[top[0]]} ({100*probs[top[0]]:.0f}%), then {L[top[1]]} ({100*probs[top[1]]:.0f}%).")
        return dict(text=text, probs=probs.tolist(), count=n, candidates=len(pk), boxes=[box(yy, xx) for _, yy, xx in cand[:3]],
                    centres=[(yy, xx) for _, yy, xx in cand[:3]], options=[f'{L[k]}) {LABELS[k]}' for k in range(4)])

    def run(self, image, um_per_px):
        """PIL image at um_per_px -> expert reading, overview (<= 1,024 px) and up to 3 crops (192 px around the strongest candidates, shown at 768 px)."""
        im = image.convert('RGB'); s = um_per_px/MPP
        if abs(s-1) > 0.02:
            im = im.resize((max(1, round(im.width*s)), max(1, round(im.height*s))), Image.LANCZOS)
        side = int(MAX_UM/MPP)
        if im.width > side or im.height > side:
            l, t = max(0, (im.width-side)//2), max(0, (im.height-side)//2); im = im.crop((l, t, min(im.width, l+side), min(im.height, t+side)))
        a = np.asarray(im); r = self.read(a); crops = []
        for y, x in r.pop('centres'):
            y0 = int(min(max(y-96, 0), max(0, a.shape[0]-192))); x0 = int(min(max(x-96, 0), max(0, a.shape[1]-192)))
            crops.append(Image.fromarray(np.ascontiguousarray(a[y0:y0+192, x0:x0+192])).resize((768, 768), Image.LANCZOS))
        ov = im.copy(); ov.thumbnail((1024, 1024))
        return dict(r, overview=ov, crops=crops)


class PolypTool:
    MPP, SIDE = 0.44, 1812  # UniToPatho-MS 800 um regions

    def __init__(self):
        self.ready = False

    def _load(self):
        from reticle.experts.polyp.subfields import encoder
        from reticle.experts.polyp.mixture import MOE, Expert, gate_fit, train_data
        sel = json.loads((MOE/'selection.json').read_text())['selected']; oof = np.load(MOE/'oof_train.npz')
        self.gate_name, self.cv = sel['name'], sel['cv_balanced']; self.gate = gate_fit(sel['name'], oof['P'], oof['y'])
        region, t2, t4, y, _, ids = train_data(); self.ex = Expert(region, t2, t4, y); self.enc = encoder()
        gold = {g['id']: g['answer'] for g in map(json.loads, open(BASE/'scoring/unitopatho_ms_v1/gold.jsonl'))}; self.nn_label = [gold[i] for i in ids]
        self.ready = True

    def features(self, a):
        """a: square HxWx3 uint8 region -> whole-region feature (384 px, ms_aug, as reticle/experts/polyp/probes.py) and the 2x2 / 4x4
        sub-field features (reticle/experts/polyp/subfields.py)."""
        from reticle.experts.polyp.subfields import encode, subtiles
        x = np.asarray(Image.fromarray(a).resize((384, 384), Image.BICUBIC))[None]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            t = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float().div(255).sub(.5).div(.5)
            region = self.enc(image=t, with_head=False, out_norm=True, ms_aug=True)[0].float().cpu().numpy()
        z = encode(self.enc, torch.from_numpy(subtiles(a))).float().numpy()
        return region.astype(np.float32), z[None, :4].astype(np.float32), z[None, 4:].astype(np.float32)

    def read(self, a):
        from reticle.experts.polyp.evidence import fmt, opinions
        from reticle.experts.polyp.mixture import EXPERTS
        if not self.ready:
            self._load()
        out, extra = self.ex.predict(*self.features(a)); E = np.stack([out[e] for e in EXPERTS], 1)[0]; m = self.gate(E[None])[0]
        nn = ', '.join(f"{self.nn_label[j]} {s:.2f}" for j, s in zip(extra['nn_idx'][0][:5], extra['nn_sim'][0][:5]))
        sub = extra['sub'][0]; cls = int(m.argmax()); key = sub[:, 3]+sub[:, 5] if cls in (2, 3, 4, 5) else sub[:, cls]
        fine = [[int(j % 4)*250, int(j//4)*250, int(j % 4+1)*250, int(j//4+1)*250] for j in np.argsort(-key)[:3]]
        who = (f'Mixture of MUSK experts (7 classifiers on frozen MUSK features at three magnifications, combined by {self.gate_name} mean; balanced accuracy '
               f'{self.cv:.0f}% in cross-validation over training slides; experts can be wrong, especially on dysplasia grade; run live on this image, '
               f'{a.shape[1]*self.MPP:.0f} um at {self.MPP} um/px)')
        text = (f'{who}.\nMixture, probability per option: {fmt(m)}.\nIndividual experts (top option): {opinions(E, EXPERTS, extra["type"][0], extra["grade"][0])}.\n'
                f'Retrieval: the 5 most similar training regions (label, cosine): {nn}.')
        return dict(text=text, probs=m.tolist(), boxes=fine, options=['A) Normal colonic mucosa', 'B) Hyperplastic polyp', 'C) Tubular adenoma, low-grade dysplasia',
                    'D) Tubular adenoma, high-grade dysplasia', 'E) Tubulovillous adenoma, low-grade dysplasia', 'F) Tubulovillous adenoma, high-grade dysplasia'])

    def run(self, image, um_per_px):
        """PIL image at um_per_px -> resampled to 0.44 um/px, centre square of at most 800 um, expert reading, overview and 3 crops (200 um sub-fields)."""
        im = image.convert('RGB'); s = um_per_px/self.MPP
        if abs(s-1) > 0.02:
            im = im.resize((max(1, round(im.width*s)), max(1, round(im.height*s))), Image.LANCZOS)
        side = min(im.width, im.height, self.SIDE); l, t = (im.width-side)//2, (im.height-side)//2; im = im.crop((l, t, l+side, t+side))
        a = np.asarray(im); r = self.read(a); q = side//4
        crops = [im.crop((b[0]*side//1000, b[1]*side//1000, b[0]*side//1000+q, b[1]*side//1000+q)).resize((768, 768), Image.LANCZOS) for b in r['boxes']]
        ov = im.copy(); ov.thumbnail((1024, 1024))
        return dict(r, overview=ov, crops=crops)


def verify_polyp(n=12):
    """Live polyp tool on UniToPatho-MS 800 um test regions vs the stored mixture (mixture.py apply; probes refit with the same seeds)."""
    from reticle.experts.polyp.mixture import MOE
    tool = PolypTool(); d = np.load(MOE/'u8_test.npz'); idx = {str(i): k for k, i in enumerate(d['ids'])}
    qs = [json.loads(x) for x in open(BASE/'runs/unitopatho_ms_v1/inputs/queries_u8_test.jsonl')][:n]; agree, diffs = 0, []
    for q in qs:
        a = np.asarray(Image.open(q['image_path']).convert('RGB')); r = tool.read(a); ref = d['mixture'][idx[q['id']]]
        diffs.append(float(np.abs(np.array(r['probs'])-ref).max())); agree += int(np.argmax(r['probs']) == np.argmax(ref))
        print(q['id'], a.shape, 'live', L6[int(np.argmax(r['probs']))], 'stored', L6[int(np.argmax(ref))], 'max |dp|', round(diffs[-1], 4), flush=True)
    print('verified', agree, '/', len(qs), 'max |dp|', round(max(diffs), 4), flush=True)


def verify(n=12):
    """Live tool on MIDOG-MS test regions vs the stored round-2 reading (same count bin and probabilities expected)."""
    from reticle.experts.mitosis.crossfit import ft_map, queries
    from reticle.experts.mitosis.counter import bin_of, peaks
    tool = MitosisTool(); rec = {r['id']: r['probs'] for r in map(json.loads, open(BASE/'runs/contract_v1/mdg3_test.jsonl'))}; ok = 0
    for q in queries('test')[:n]:
        a = np.load(q['native']); r = tool.read(a); pts, p = ft_map(2, 'test', q['id'])
        same = np.allclose(r['probs'], rec[q['id']], atol=1e-4); ok += same
        print(q['id'], 'live count', r['count'], 'stored count', len(peaks(pts, p, tool.cfg['tau'])), 'probs match', same, flush=True)
    print('verified', ok, '/', n, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['export', 'verify', 'verify_polyp', 'mitosis'])
    ap.add_argument('args', nargs='*', help='verify / verify_polyp: number of test regions (default 12); mitosis: <image> <um_per_px>'); a = ap.parse_args()
    if a.mode == 'export':
        export()
    elif a.mode == 'verify':
        verify(int(a.args[0]) if a.args else 12)
    elif a.mode == 'verify_polyp':
        verify_polyp(int(a.args[0]) if a.args else 12)
    else:
        print(MitosisTool().run(Image.open(a.args[0]), float(a.args[1]))['text'])
