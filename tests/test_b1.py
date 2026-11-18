"""B1: the water detectors in the pipeline (no GPU; needs the ONNX files from experiments/b1_export.py)."""

import numpy as np
import pytest

from core.models import water
from experiments import b1_common as C
from experiments.b1_rules import counts_at, EDGES

needs_models = pytest.mark.skipif(not (water.MODELS / "water_unet_ALL.onnx").exists(),
                                  reason="ONNX water models not made yet")


def test_iou_of_counts():
    assert C.iou([3, 1, 1, 10]) == pytest.approx(0.6)          # TP / (TP + FP + FN)
    assert C.iou(np.array([[3, 1, 1, 10], [1, 0, 0, 5]])) == pytest.approx(4 / 6)   # pooled, not averaged


def test_threshold_counts_from_histograms():
    hw = np.histogram([0.3, 0.1, -0.2], EDGES)[0]       # water pixels' index values
    hd = np.histogram([-0.5, -0.1, 0.2], EDGES)[0]      # dry pixels' index values
    assert counts_at(hw, hd, 0.0) == [2, 1, 1, 2]


def test_ndwi_water_is_the_old_rule():
    c = np.zeros((7, 1, 2), np.float32)
    c[1] = [[0.10, 0.05]]   # green
    c[3] = [[0.05, 0.30]]   # B08
    assert water.ndwi_water(c).tolist() == [[True, False]]


@needs_models
def test_unet_runs_on_small_and_odd_sized_images():
    rng = np.random.default_rng(0)
    for shape in ((7, 50, 50), (7, 500, 503)):
        c = rng.uniform(0.02, 0.4, shape).astype(np.float32)
        c[:, :5, :5] = np.nan                       # no clear image there
        p = water.unet_probability(c)
        assert p.shape == shape[1:] and np.isfinite(p).all() and (0 <= p).all() and (p <= 1).all()


@needs_models
def test_unet_marks_open_water_and_vegetation_correctly():
    """Textbook spectra: water is dark in near and short-wave infrared, vegetation bright in near infrared."""
    c = np.empty((7, 256, 256), np.float32)
    water_px = [0.06, 0.07, 0.05, 0.03, 0.03, 0.02, 0.01]
    veg_px = [0.04, 0.07, 0.04, 0.35, 0.36, 0.20, 0.10]
    c[:, :, :128] = np.array(water_px)[:, None, None]
    c[:, :, 128:] = np.array(veg_px)[:, None, None]
    m = water.water_mask(c, "unet")
    assert m[:, 20:108].mean() > 0.95 and m[:, 148:236].mean() < 0.05


def test_unknown_detector_is_refused():
    with pytest.raises(ValueError):
        water.water_mask(np.zeros((7, 4, 4), np.float32), "magic")
