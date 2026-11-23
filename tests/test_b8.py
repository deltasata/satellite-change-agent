"""Milestone B8: which CEMS map is used, which radar and TerraMind dates are picked, and the 12 input channels
(no data, no network files)."""

from datetime import date

import numpy as np
import pytest


def _product(t, mon, status="F", url="x_v1.zip"):
    return {"type": t, "monitoringNumber": mon, "downloadPath": url, "version": {"statusCode": status}}


def test_map_choice_follows_impactmesh():
    from core.data.cems import choose
    aoi = {"products": [_product("DEL", 0), _product("DEL", 3), _product("GRA", 0), _product("DEL", 5, "N", "")]}
    assert choose(aoi)["type"] == "GRA"                                  # grading map first
    aoi = {"products": [_product("DEL", 0), _product("DEL", 3), _product("DEL", 5, "N", "")]}
    assert choose(aoi)["monitoringNumber"] == 3                          # else the last finished delineation
    assert choose({"products": [_product("FEP", 0)]}) is None


def test_radar_dates_most_covered_then_closest_to_the_fire():
    from core.data.burn_radar import select
    d = date(2021, 7, 1)
    dates = [f"2021-07-{k:02d}" for k in range(2, 30)]
    cover = np.ones(len(dates))
    cover[0] = 0.5
    idx = select(dates, cover, (date(2021, 7, 2), date(2021, 7, 29)), d)
    assert len(idx) == 8 and idx == sorted(idx)
    assert [dates[k] for k in idx] == [f"2021-07-{k:02d}" for k in range(3, 11)]   # not the half-covered day


def test_terramind_dates():
    from core.data.tmfire_inputs import pick_optical, pick_radar
    d = date(2021, 8, 1)
    days = [date(2021, 6, 1), date(2021, 6, 11), date(2021, 7, 1), date(2021, 7, 6), date(2021, 8, 2),
            date(2021, 8, 3), date(2021, 8, 20), date(2021, 9, 5)]
    clear = np.array([1.0, 1.0, 0.2, 0.9, 0.1, 1.0, 0.95, 1.0])
    o = pick_optical(days, clear, d)
    assert o[0] == 1                 # June 11 (d-51) wins a tie on clearness against June 1 (d-61)
    assert o[1] == 3                 # the clearer of July 1 and July 6
    assert o[2] == 4                 # the event step: closest to d, however cloudy
    assert o[3] == 7                 # the clearest after the fire
    assert pick_radar(days, d) == [1, 3, 4, 6]


def test_twelve_channels_and_time_order():
    pytest.importorskip("torch")
    from experiments.b8_train import inputs
    xo = np.full((2, 7, 4, 4), 0.1, np.float32)
    xo[1, :, 0, 0] = np.nan
    xr = np.full((1, 2, 4, 4), -12.0, np.float32)
    xr[0, :, 1, 1] = np.nan
    dem = np.full((4, 4), 456.555, np.float32)
    order = [(-10, 0, 0), (-3, 1, 0), (5, 0, 1)]          # optical, radar, optical
    x = inputs((xo, xr, dem, order), size=4)
    assert x.shape == (3, 12, 4, 4)
    assert x[0, 7].all() and not x[0, 8:11].any()          # optical date: clear flag, no radar channels
    assert not x[1, :8].any() and x[1, 10, 0, 0] == 1 and x[1, 10, 1, 1] == 0 and x[1, 8, 1, 1] == 0
    assert x[2, 7, 0, 0] == 0 and x[2, 0, 0, 0] == 0       # missing optical pixel: 0 and not clear
    assert np.allclose(x[:, 11], 0.0)                      # elevation at the mean -> 0 on every date


def test_pipeline_dates_follow_the_training_copy():
    from core.data.impactmesh_ts import select as select_ts
    from core.models.change import select
    rng = np.random.default_rng(0)
    clear = rng.uniform(0, 1, 30)
    clear[[3, 7]] = 0.005                                                # below 1 % clear: never kept
    dates = [f"2021-07-{k + 1:02d}" for k in range(30)]
    assert select(clear) == select_ts(dates, clear, (date(2021, 7, 1), date(2021, 7, 30)))
    assert len(select(clear)) == 12 and 3 not in select(clear[:10]) and select(clear[:10]) == sorted(select(clear[:10]))
    from core.measure import VEGETATION_DETECTOR
    from core.models.change import NAMES, SERIES
    assert VEGETATION_DETECTOR in NAMES and VEGETATION_DETECTOR in SERIES


def test_pipeline_burn_detector_takes_no_radar():
    """The radar network was dropped after B8: the pipeline's burn detector reads the 7 optical bands and a
    clear flag, nothing else."""
    import inspect
    from core.measure import VEGETATION_DETECTOR
    from core.models import change
    assert VEGETATION_DETECTOR == "burn-utae"
    assert "radar" not in change.NAMES[VEGETATION_DETECTOR].lower()
    assert change.MEAN.shape[0] == 7                                     # 7 optical bands
    src = inspect.getsource(change.probability_series)
    assert "burn_radar" not in src and "dem" not in src.lower()


def test_construction_detector_removes_water():
    """B9 avenue 1: the construction detector is B4's network minus the water of the after image; the suffix is
    stripped before the network files are looked up."""
    from core.measure import CHANGE_DETECTOR
    from core.models.change import NAMES, WATER_SUFFIX, _parse
    assert CHANGE_DETECTOR == "terramind" + WATER_SUFFIX and CHANGE_DETECTOR in NAMES
    assert _parse((CHANGE_DETECTOR + "-without-paris_bercy").replace(WATER_SUFFIX, "")) == ("terramind", "paris_bercy")
