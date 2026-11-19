"""Milestone B3: the cloud scoring and the window procedure of the OmniCloudMask port.

The port's agreement with the library itself (same classes, pixel for pixel, on 7 awkward shapes; 99.996 %
on the 953 test images made on the GPU) is checked by experiments/b3_check.py, which needs the library
and the weights."""

import numpy as np

from core.models import cloud
from experiments import b3_common as C


def test_scl_rule_classes_and_widening():
    s = np.array([[3, 8, 9, 10, 4, 7]])
    assert C.scl_classes(s).tolist() == [[3, 1, 1, 2, 0, 0]]               # shadow, thick, thick, thin, clear
    assert C.scl_classes(s, (3, 7, 8, 9, 10)).tolist() == [[1, 1, 1, 1, 0, 1]]
    one = np.zeros((7, 7), np.uint8)
    one[3, 3] = 9
    assert C.scl_classes(one, (3, 7, 8, 9, 10), widen=2).sum() == 25          # a 5 x 5 square


def test_unusable_iou_from_the_4_by_4_table():
    y = np.array([0, 0, 1, 2, 3, 3])
    pred = np.array([0, 1, 1, 0, 3, 2])          # 1 good pixel thrown away, 1 cloud let through
    cm = C.confusion(pred, y, np.ones(6, bool))
    tp, fp, fn, tn = C.binary(cm)
    assert (tp, fp, fn, tn) == (3, 1, 1, 1)
    assert C.iou_unusable(cm) == 3 / 5            # a shadow called thin cloud still counts as unusable


def test_windows_cover_the_image_and_shrink_for_small_images():
    size, overlap = cloud._patch_size(np.ones((3, 500, 502), np.float32))
    assert (size, overlap) == (500, 250)          # as the library: window = smaller side, overlap = half
    idx = cloud._indexes(500, 502, size, overlap)
    covered = np.zeros((500, 502), bool)
    for t, b, l, r in idx:
        covered[t:b, l:r] = True
    assert covered.all() and all(b - t == 500 and r - l == 500 for t, b, l, r in idx)


def test_blending_weight_rises_linearly_over_the_overlap():
    g = cloud._gradient(10, 3)
    assert g.shape == (10, 10) and np.isclose(g[5, 5], 1.0)
    assert np.allclose(g[5, :3], [1 / 3, 2 / 3, 1.0]) and np.allclose(g, np.rot90(g, 2))


def test_window_moves_inward_away_from_a_no_data_edge():
    x = np.random.default_rng(0).random((3, 100, 100)).astype(np.float32) + 0.1
    x[:, :, :10] = np.nan
    p, idx = cloud._get_patch(x, (0, 60, 0, 60))
    assert idx == (0, 60, 10, 70)                 # moved 10 pixels right, to the first column with data
    assert np.allclose(p.reshape(3, -1).mean(1), 0, atol=1e-5) and np.allclose(p.reshape(3, -1).std(1), 1, atol=1e-4)
