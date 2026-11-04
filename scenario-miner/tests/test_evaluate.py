import numpy as np

from scenario_miner.evaluate import _bootstrap_weights, average_precision, prf


def test_prf_known_values():
    p, r, f = prf(8, 2, 4)
    assert (float(p), float(r)) == (0.8, 8 / 12)
    assert abs(float(f) - 2 * 0.8 * (2 / 3) / (0.8 + 2 / 3)) < 1e-12
    p, r, f = prf(0, 0, 3)  # no positive predictions: precision undefined, recall 0
    assert np.isnan(p) and float(r) == 0.0 and float(f) == 0.0


def test_average_precision():
    gt = np.array([1, 1, 0, 0, 1], dtype=bool)
    assert average_precision(np.array([0.9, 0.8, 0.1, 0.2, 0.7]), gt) == 1.0
    worst = average_precision(np.array([0.1, 0.2, 0.9, 0.8, 0.3]), gt)
    assert abs(worst - (1 / 3 + 2 / 4 + 3 / 5) / 3) < 1e-12
    assert np.isnan(average_precision(np.zeros(3), np.zeros(3, dtype=bool)))


def test_bootstrap_weights_resample_segments():
    W = _bootstrap_weights(7, 500, np.random.default_rng(0))
    assert W.shape == (500, 7) and (W.sum(axis=1) == 7).all()
    assert abs(W.mean() - 1.0) < 0.05
