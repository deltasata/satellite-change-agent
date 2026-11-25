"""Milestone B10: a stored cube reduced to chosen dates (core.tools.subset_cube), and the fire rule's settings."""

from datetime import date

import numpy as np
import xarray as xr


def test_subset_cube_keeps_only_the_chosen_dates(tmp_path, monkeypatch):
    monkeypatch.setenv("EO_DATA_CACHE", str(tmp_path))
    from core import tools
    t = np.array(["2025-01-01", "2025-01-06", "2025-01-11"], dtype="datetime64[ns]")
    ds = xr.Dataset({"red": (("time", "y", "x"), np.arange(12, dtype=np.float32).reshape(3, 2, 2)),
                     "valid_fraction": (("time",), np.array([0.05, 0.9, 0.5], np.float32))},
                    coords={"time": t, "y": [0.0, 10.0], "x": [0.0, 10.0]}, attrs={"bbox": [0, 0, 1, 1]})
    p = tools._cube_path("parent")
    p.parent.mkdir(parents=True)
    ds.to_netcdf(p, engine="h5netcdf")
    assert [round(v, 2) for _, v in tools.cube_dates("parent")] == [0.05, 0.9, 0.5]
    s = tools.subset_cube("parent", [date(2025, 1, 11), date(2025, 1, 6)])
    assert s.n_dates == 2 and s.n_usable_dates == 1 and s.start == date(2025, 1, 6)
    sub = tools._load_cube(s.cube_id)
    assert sub["red"].shape == (2, 2, 2) and float(sub["red"][0, 0, 0]) == 4.0


def test_fire_rule_is_the_adopted_one():
    from core import measure
    assert measure.FIRE_SELECTION == "nearest8" and measure.NEAREST_N == 8
    assert measure.FIRE_PERIODS["wide"] == ((-120, -3), (7, 90))


def test_construction_falls_back_to_the_rule_without_the_network(tmp_path, monkeypatch):
    """A public copy has no construction network (its training labels may not be shared)."""
    from core.models import change
    monkeypatch.setattr(change, "MODELS", tmp_path)
    assert change.available("rule") and not change.available("terramind+water")
