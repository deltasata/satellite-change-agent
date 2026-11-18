"""Milestone B2: the radar water rules and the scores."""

import numpy as np

from experiments import b2_common as B
from experiments import b2_rules as R


def test_otsu_splits_two_groups():
    rng = np.random.default_rng(0)
    v = np.concatenate([rng.normal(-25, 1.5, 4000), rng.normal(-14, 1.5, 6000)])   # water and land in dB
    assert -22 < R.otsu(v) < -17


def test_threshold_counts_from_histograms_equal_direct_counts():
    rng = np.random.default_rng(1)
    vh = np.round(rng.uniform(-35, -5, (50, 50)), 1).astype(np.float32) + 0.05     # inside the 0.1 dB bins
    y = (rng.random((50, 50)) < 0.3).astype(np.int8)
    m = np.ones_like(y, bool)
    d = {"vh": [np.histogram(vh[y == cls], R.EDGES)[0] for cls in (1, 0)]}
    for t in (-25.0, -20.3, -12.0):
        assert R.counts_at(d, "vh", t) == B.counts(vh < t, y, m)


def test_miou_is_mean_of_water_and_dry_iou():
    tp, fp, fn, tn = 30, 10, 20, 940
    assert abs(B.miou([tp, fp, fn, tn]) - (30 / 60 + 940 / 970) / 2) < 1e-12


def test_radar_db_conversion():
    from core.data.radar import to_db
    out = to_db(np.array([1.0, 0.01, 0.0, -1.0], np.float32))
    assert np.allclose(out[:2], [0.0, -20.0]) and np.isnan(out[2:]).all()
