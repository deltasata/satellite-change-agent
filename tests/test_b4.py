"""Milestone B4: scoring arithmetic, the groups of neighbouring cities, and the network's windows (no data, no network files)."""

import numpy as np
import pytest

from core.data import oscd_pc
from core.models import change


def test_neighbouring_cities_form_one_group():
    assert oscd_pc.group("saclay_e") == oscd_pc.group("saclay_w")
    assert oscd_pc.group("paris") == oscd_pc.group("bercy")
    assert oscd_pc.group("beirut") == "beirut"


def test_confusion_and_iou():
    from experiments import b4_common as C
    label = np.array([[1, 1, 0, 0]], bool)
    pred = np.array([[1, 0, 1, 0]], bool)
    usable = np.array([[1, 1, 1, 0]], bool)       # the last pixel is not scored
    cm = C.confusion(pred, {"label": label, "usable": usable})
    assert cm == [1, 1, 1, 0]
    assert C.iou(cm) == pytest.approx(1 / 3)


def test_bootstrap_resamples_groups_not_cities():
    from experiments import b4_common as C
    a = {"saclay_e": [1, 1, 1, 10], "saclay_w": [1, 1, 1, 10], "beirut": [1, 1, 1, 10]}
    b = {"saclay_e": [2, 0, 0, 12], "saclay_w": [2, 0, 0, 12], "beirut": [2, 0, 0, 12]}
    d, lo, hi, n = C.bootstrap_difference(a, b, n=200)
    assert n == 2 and d == pytest.approx(1 - 1 / 3) and lo == pytest.approx(d) and hi == pytest.approx(d)


def test_model_names():
    assert change._parse("fsd") == ("fsd", "ALL")
    assert change._parse("terramind-without-paris_bercy") == ("terramind", "paris_bercy")


class _Fake:
    """Stands in for an ONNX session: 'changed' logit = after minus before, band 0 (scaled)."""

    def run(self, _, feed):
        d = feed["after"][:, 0] - feed["before"][:, 0]
        return [np.stack([np.zeros_like(d), d], 1)]


def test_small_area_is_mirrored_and_cut_back(monkeypatch):
    monkeypatch.setattr(change, "_sessions", lambda name: [_Fake(), _Fake(), _Fake()])
    before = np.full((7, 200, 180), 0.1, np.float32)
    after = before.copy()
    after[0, 50:60, 70:90] = 0.6                 # a bright new patch
    p = change.probability(before, after, "fsd")
    assert p.shape == (200, 180)
    m = change.changed(before, after, "fsd")
    assert m[50:60, 70:90].all() and m.sum() == 200


def test_pipeline_uses_a_known_change_detector():
    from core import measure
    assert measure.CHANGE_DETECTOR in change.NAMES
