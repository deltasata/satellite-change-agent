from datetime import date

import numpy as np
import pytest

from core.data.cloud_mask import SCL_CLASSES, valid_mask
from core.data.timeseries import to_reflectance


def test_every_scl_class_has_a_decision():
    scl = np.arange(12)
    m = valid_mask(scl)
    assert set(np.flatnonzero(~m)) == {0, 1, 3, 8, 9, 10}
    assert set(np.flatnonzero(m)) == {2, 4, 5, 6, 7, 11}
    assert len(SCL_CLASSES) == 12


def test_mask_keeps_shape_and_rejects_nan():
    scl = np.array([[[4.0, 9.0], [np.nan, 6.0]]])
    assert valid_mask(scl).tolist() == [[[True, False], [False, True]]]


def test_reflectance_calibration():
    dn = np.array([0, 1000, 3000], dtype=np.uint16)
    # old processing: no offset; new processing: -0.1 offset (DN shifted by +1000)
    old = to_reflectance(dn, 1e-4, 0.0)
    new = to_reflectance(dn, 1e-4, -0.1)
    assert np.isnan(old[0]) and np.isnan(new[0])  # DN 0 means "no data", not "black"
    np.testing.assert_allclose(old[1:], [0.1, 0.3], rtol=1e-6)
    np.testing.assert_allclose(new[1:], [0.0, 0.2], atol=1e-6)


@pytest.mark.network
def test_live_timeseries_small_area():
    """End to end on a 1 km x 1 km box for one month."""
    from core.data.stac_search import search_imagery
    from core.data.timeseries import load_timeseries
    from core.schemas import BBox, SearchRequest

    bbox = BBox(west=11.570, south=48.130, east=11.584, north=48.139)
    res = search_imagery(SearchRequest(bbox=bbox, start=date(2024, 7, 1), end=date(2024, 7, 31)))
    ds = load_timeseries(res.scenes, bbox)
    assert ds.sizes["time"] == len({s.datetime.date() for s in res.scenes})
    assert 80 <= ds.sizes["x"] <= 130 and 80 <= ds.sizes["y"] <= 130  # ~1 km at 10 m
    vf = ds["valid_fraction"].values
    assert ((0 <= vf) & (vf <= 1)).all()
    # Physics checks on clear pixels (these caught a double-applied calibration offset):
    red, nir, scl = ds["red"].values, ds["nir"].values, ds["scl"].values
    clear = red[np.isfinite(red)]
    assert clear.size > 0
    # reflectance lives in [0, ~1]; a few specular glints (glass roofs) may exceed 1
    assert np.mean((clear < -0.01) | (clear > 1.0)) < 0.01
    # vegetation (SCL 4): dark in red (chlorophyll absorbs), bright in near-infrared
    veg = (scl == 4) & np.isfinite(red) & np.isfinite(nir)
    assert veg.sum() > 100
    assert 0.0 < np.median(red[veg]) < 0.08
    assert np.median(nir[veg]) > 3 * np.median(red[veg])
    # where ESA's class map says cloud/shadow/no-data, the data must be NaN
    assert np.isnan(red[~valid_mask(ds["scl"].values)]).all()
