"""Expert records and temperature calibration (reticle.experts.contract) on synthetic data."""
import numpy as np

from reticle.experts.contract import apply_temperature, ece, fit_temperature, record


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0); n, k, t_true = 4000, 4, 2.5
    logits = rng.normal(0, 1, (n, k)); p = np.exp(logits)/np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=row) for row in p])
    over = np.log(p)*t_true   # an over-confident expert: logits multiplied by the true temperature
    t = fit_temperature(over, y)
    assert abs(t-t_true) < 0.3
    cal = apply_temperature(over, t)   # calibrated probabilities
    before = np.exp(over)/np.exp(over).sum(1, keepdims=True)
    assert ece(cal.max(1), cal.argmax(1) == y) < ece(before.max(1), before.argmax(1) == y)


def test_record_normalises_probabilities_and_clips_boxes():
    r = record('q1', 'demo', 'expert', [2, 1, 1, 0], rois=[dict(box=[-5, 10, 1200, 50], score=0.9)])
    assert abs(sum(r['probs'])-1) < 1e-6 and abs(r['confidence']-0.5) < 1e-6
    box = r['rois'][0]['box']; assert min(box) >= 0 and max(box) <= 1000
