"""Milestone B5: windows around the fire date, which patches are used, scoring arithmetic, the bootstrap over
fires, Prithvi's 30 m averaging (no data, no network files)."""

from datetime import date

import numpy as np
import pytest

from core.data import impactmesh_es as D


def test_windows_around_the_fire_date():
    (b0, b1), (a0, a1) = D._windows(date(2021, 8, 13), "normal")
    assert (b0, b1, a0, a1) == (date(2021, 6, 14), date(2021, 8, 10), date(2021, 8, 20), date(2021, 10, 12))
    (b0, _), (_, a1) = D._windows(date(2021, 8, 13), "wide")
    assert b0 == date(2021, 4, 15) and a1 == date(2021, 11, 11)


def test_fires_at_exam_places_are_not_used_for_training():
    p = dict(part="train", fire="EMSR207")
    assert not D.used(p)
    assert D.used(dict(part="train", fire="EMSR209"))
    assert D.used(dict(part="test_holdout", fire="EMSR387"))
    assert not D.used(dict(part="test_seen_events", fire="EMSR209"))


def test_unlabelled_and_unusable_pixels_are_not_scored():
    from experiments import b5_common as C
    z = {"label": np.array([[1, 1, 0, -1]], np.int8), "usable": np.array([[1, 0, 1, 1]], bool)}
    pred = np.array([[1, 1, 1, 1]], bool)
    assert C.confusion(pred, z) == [1, 1, 0, 0]


def test_bootstrap_resamples_fires():
    from experiments import b5_common as C
    a = {"EMSR1_a": [1, 1, 1, 10], "EMSR1_b": [1, 1, 1, 10], "EMSR2_a": [1, 1, 1, 10]}
    b = {"EMSR1_a": [2, 0, 0, 12], "EMSR1_b": [2, 0, 0, 12], "EMSR2_a": [2, 0, 0, 12]}
    d, lo, hi, n = C.bootstrap_difference(a, b, n=200)
    assert n == 2 and d == pytest.approx(1 - 1 / 3) and lo == pytest.approx(d) and hi == pytest.approx(d)


def test_dnbr_is_positive_where_vegetation_burnt():
    from experiments import b5_common as C
    before = np.full((7, 2, 2), 0.05, np.float32)
    before[3], before[6] = 0.30, 0.10            # green vegetation: high B08, low B12
    after = before.copy()
    after[3, 0, 0], after[6, 0, 0] = 0.15, 0.20  # burnt: B08 down, B12 up
    d = C.dnbr({"before": before, "after": after})
    assert d[0, 0] > 0.5 and abs(d[1, 1]) < 1e-6


def test_prithvi_input_is_averaged_to_30_m():
    pytest.importorskip("torch")
    from experiments.b5_prithvi import to30
    x = np.arange(256 * 256, dtype=np.float32).reshape(1, 256, 256)
    y = to30(x)
    assert y.shape == (1, 86, 86)
    assert y[0, 0, 0] == pytest.approx(x[0, :3, :3].mean())


def test_vegetation_detector_is_a_known_burn_network():
    from core.measure import VEGETATION_DETECTOR
    from core.models.change import NAMES
    assert VEGETATION_DETECTOR in NAMES and VEGETATION_DETECTOR.startswith("burn-")
