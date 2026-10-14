import math

import numpy as np
import pytest

from vpt import RobustScale, conformal_threshold


def test_conformal_threshold_order_statistic():
    s = list(range(1, 100))  # n = 99
    # ceil(100 * 0.9) = 90 -> 90th smallest
    assert conformal_threshold(s, 0.1) == 90


def test_conformal_threshold_infinite_when_alpha_too_small():
    assert math.isinf(conformal_threshold([0.1, 0.2, 0.3], 0.01))


def test_conformal_fpr_is_controlled():
    rng = np.random.default_rng(0)
    alpha, hits, trials = 0.1, 0, 2000
    for _ in range(trials):
        cal = rng.normal(size=200)
        t = conformal_threshold(cal, alpha)
        hits += rng.normal() > t
    assert hits / trials <= alpha + 0.02


def test_robust_scale_floor():
    sc = RobustScale.fit([0.0] * 50)
    assert sc.scale == pytest.approx(0.05)
    assert sc(0.5) == pytest.approx(10.0)
