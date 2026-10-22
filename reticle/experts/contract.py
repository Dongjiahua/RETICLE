"""The expert record every RETICLE expert writes (docs/expert_interface.md): schema, temperature calibration, reports and validation.
A record is {id, task, expert, probs (calibrated, over the question's options), confidence (= max probs), rois [{box, score}]}, with boxes
in the 0-1000 coordinates of the image overview. Records go to $RETICLE_DATA/runs/contract_v1/<name>_<split>.jsonl."""
import json

import numpy as np

from reticle.paths import DATA

OUT = DATA/'runs/contract_v1'


def fit_temperature(logp, y, grid=np.exp(np.linspace(np.log(0.05), np.log(20), 200))):
    """Temperature T minimising the NLL of softmax(logp / T) on a calibration pool (logp: [N, K] log-probabilities)."""
    def nll(t):
        z = logp/t; z = z-z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)))
    return float(min(grid, key=nll))


def apply_temperature(logp, t):
    z = logp/t; z = z-z.max(1, keepdims=True); p = np.exp(z)
    return p/p.sum(1, keepdims=True)


def ece(conf, correct, bins=10):
    """Expected calibration error of 'confidence that the top option is right', equal-width bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float); e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf >= lo) & (conf < lo+1/bins) if lo < 1-1/bins else (conf >= lo)
        if m.any():
            e += m.mean()*abs(conf[m].mean()-correct[m].mean())
    return float(e)


def report(probs, y):
    """Accuracy, balanced accuracy, mean confidence, ECE, accuracy of the more / less confident half and NLL, in percent."""
    probs = np.asarray(probs); y = np.asarray(y); top = probs.argmax(1); ok = top == y; conf = probs.max(1)
    order = np.argsort(-conf); half = order[:len(order)//2]
    return dict(n=int(len(y)), accuracy=round(100*float(ok.mean()), 2), balanced=round(100*float(np.mean([ok[y == k].mean() for k in np.unique(y)])), 2),
                mean_confidence=round(100*float(conf.mean()), 2), ece=round(100*ece(conf, ok), 2),
                accuracy_top_half_confidence=round(100*float(ok[half].mean()), 2), accuracy_bottom_half=round(100*float(ok[order[len(order)//2:]].mean()), 2),
                nll=round(float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)))), 4))


def record(qid, task, expert, probs, rois=()):
    """One record; probabilities are renormalised and boxes clipped to the image."""
    p = np.asarray(probs, float); p = p/p.sum()
    return dict(id=str(qid), task=task, expert=expert, probs=[round(float(x), 5) for x in p], confidence=round(float(p.max()), 5),
                rois=[dict(box=[round(min(1000.0, max(0.0, float(v))), 1) for v in r['box']], score=round(float(r['score']), 4)) for r in rois])


def validate(path):
    rows = [json.loads(x) for x in open(path)]
    for r in rows:
        assert set(r) >= {'id', 'task', 'expert', 'probs', 'confidence', 'rois'}, r.keys()
        assert abs(sum(r['probs'])-1) < 1e-3 and all(p >= 0 for p in r['probs']), r['id']
        assert abs(r['confidence']-max(r['probs'])) < 1e-4, r['id']
        for roi in r['rois']:
            assert len(roi['box']) == 4 and all(-1 <= v <= 1001 for v in roi['box']) and roi['box'][2] > roi['box'][0] and roi['box'][3] > roi['box'][1], (r['id'], roi)
    return len(rows)


def write(task, split, rows):
    """Write and validate <task>_<split>.jsonl; returns the number of records."""
    OUT.mkdir(parents=True, exist_ok=True); p = OUT/f'{task}_{split}.jsonl'
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows)); return validate(p)
