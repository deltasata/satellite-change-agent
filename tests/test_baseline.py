"""Change-model tests on synthetic data: inject a known change, check it is recovered."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from core.models.baseline import (
    binary_metrics,
    change_score_pair,
    detect_change_timeseries,
    normalized_difference,
    otsu_threshold,
)

BANDS = ("blue", "green", "red", "nir", "swir16")


def test_metrics_known_counts():
    pred = np.array([1, 1, 0, 0, 1], bool)
    truth = np.array([1, 0, 1, 0, 1], bool)
    m = binary_metrics(pred, truth)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["iou"] == pytest.approx(0.5)


def test_normalized_difference_handles_zero():
    out = normalized_difference(np.array([0.3, 0.0]), np.array([0.1, 0.0]))
    assert out[0] == pytest.approx(0.5) and np.isnan(out[1])


def test_pair_injection_recovery():
    rng = np.random.default_rng(0)
    before = 0.1 + 0.01 * rng.standard_normal((5, 80, 80)).astype(np.float32)
    after = before + 0.01 * rng.standard_normal((5, 80, 80)).astype(np.float32)
    after *= 1.3  # global illumination change: must NOT count as change
    after[:, 20:40, 20:40] += 0.15  # the injected change
    score = change_score_pair(before, after)
    pred = score > otsu_threshold(score)
    truth = np.zeros((80, 80), bool)
    truth[20:40, 20:40] = True
    assert binary_metrics(pred, truth)["f1"] > 0.9


def _cube(n_before=8, n_after=8, change=0.0, cloud_all_before=False, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.to_datetime([f"2024-05-{d:02d}" for d in range(1, 1 + n_before)]
                       + [f"2024-07-{d:02d}" for d in range(1, 1 + n_after)])
    ny = nx = 40
    data = {}
    for k, b in enumerate(BANDS):
        x = 0.05 + 0.04 * k + 0.005 * rng.standard_normal((len(t), ny, nx))
        x[n_before:, 10:20, 10:20] += change
        if cloud_all_before:
            x[:n_before] = np.nan
        data[b] = (("time", "y", "x"), x.astype(np.float32))
    ds = xr.Dataset(data, coords={"time": t, "y": np.arange(ny) * -10.0, "x": np.arange(nx) * 10.0})
    ds["valid_fraction"] = ("time", np.isfinite(ds["red"]).mean(("y", "x")).values)
    return ds


WIN = dict(before=(date(2024, 5, 1), date(2024, 5, 31)), after=(date(2024, 7, 1), date(2024, 7, 31)))


def test_timeseries_recovers_injected_change():
    r = detect_change_timeseries(_cube(change=0.2), **WIN)
    assert r["status"] == "ok"
    block = np.zeros((40, 40), bool)
    block[10:20, 10:20] = True
    assert r["_changed"][block].mean() > 0.9  # found
    assert r["_changed"][~block].mean() < 0.01  # no false alarms outside


def test_timeseries_null_is_quiet():
    r = detect_change_timeseries(_cube(change=0.0), **WIN)
    assert r["status"] == "ok" and r["changed_fraction"] < 0.005


def test_timeseries_abstains_without_data():
    r = detect_change_timeseries(_cube(cloud_all_before=True), **WIN)
    assert r["status"] == "insufficient_data" and "reason" in r
