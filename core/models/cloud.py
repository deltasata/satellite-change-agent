"""Which pixels are cloud or cloud shadow: OmniCloudMask, run from ONNX files on the CPU (milestone B3).

B3 (docs/METHODS.md section 6) compared cloud detectors on 953 expert-labelled CloudSEN12+ test images
(195 places, none of them in the detectors' training data), in Earth Search's Level-2A images, the kind
the pipeline reads. IoU of "unusable" (thick cloud, thin cloud or shadow), pooled:
    ESA's scene classification, SCL 3/8/9/10 (the rule until B3)     0.644
    the best SCL variant (classes 3/7/8/9/10, widened by 5 pixels)    0.681
    U-Net trained from nothing (3 seeds)                              0.844
    TerraMind-1.0-base fine-tuned (3 seeds)                           0.878
    OmniCloudMask, as published                                       0.885  (+0.241, interval +0.221..+0.261)

OmniCloudMask (Wright et al. 2025, MIT licence; model version 4): two U-Nets (encoders regnety_004 and
edgenext_small) read red (B04), green (B03) and narrow near infrared (B8A); their outputs are averaged;
each pixel gets the most probable of 4 classes, 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow.

The networks run from ONNX files (experiments/b3_export.py) so the pipeline needs no PyTorch. Everything
around them is the library's own procedure (omnicloudmask 1.8.1, cloud_mask.py, raster_utils.py,
model_utils.py), rewritten in NumPy below: windows of up to 1000 pixels overlapping by 300 (smaller for
small images), each window scaled band by band to mean 0 and spread 1 over its valid pixels, windows moved
inward away from no-data edges, results blended with a weight that rises linearly over the overlap, and
no-data pixels set to class 0. Checked against the library: tests/test_b3.py and experiments/b3_check.py.

    Portions adapted from OmniCloudMask, Copyright (c) 2025 DPIRD-DMA, MIT licence:
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions: The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
    WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "data_cache" / "models"
NETS = ("regnety_004", "edgenext_small")
PATCH, OVERLAP, HARD_MIN = 1000, 300, 32
NAME = "OmniCloudMask 1.8.1, model version 4 (milestone B3)"


@lru_cache(maxsize=2)
def _session(net: str):
    import onnxruntime as ort
    path = MODELS / f"ocm_v4_{net}.onnx"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; make it with: .venv-ml/bin/python -m experiments.b3_export")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _nodata(a: np.ndarray) -> np.ndarray:
    return np.isnan(a) | (a == 0)


def _channel_norm(patch: np.ndarray) -> np.ndarray:
    out = np.zeros(patch.shape, np.float32)
    for k, band in enumerate(patch):
        ok = ~_nodata(band)
        if ok.any():
            v = band[ok]
            mean, std = v.mean(), v.std()
            out[k][ok] = (v - mean) / (std if std != 0 else 1)
    return out


def _patch_size(x: np.ndarray, size: int = PATCH, overlap: int = OVERLAP) -> tuple[int, int]:
    """The library's check_patch_size: smaller windows for small images or images with much no data."""
    h, w = x.shape[1:]
    if np.count_nonzero(_nodata(x)) / x.size > 0.3 and size > min(h, w) / 2:
        size = max(min(h, w) // 2, HARD_MIN)
        overlap = min(overlap, size // 2)
    if size > min(h, w):
        size = max(min(h, w), HARD_MIN)
        overlap = min(overlap, size // 2)
    if overlap >= size or size < HARD_MIN:
        raise ValueError(f"image {h} x {w} too small for OmniCloudMask")
    return size, overlap


def _indexes(h: int, w: int, size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride, out = size - overlap, []
    for top in range(0, h, stride):
        top = min(top, h - size)
        for left in range(0, w, stride):
            left = min(left, w - size)
            out.append((top, top + size, left, left + size))
    return out


def _get_patch(x: np.ndarray, index):
    """The library's get_patch: None if all no data; otherwise move the window inward while a whole edge
    row / column is no data; return the scaled window and its final position."""
    top, bottom, left, right = index
    p = x[:, top:bottom, left:right]
    nd = _nodata(p)
    if nd.all():
        return None, None
    if nd.any():
        H, W = x.shape[1:]
        moved = False
        if bottom < H and _nodata(p[:, 0, :]).all():
            while bottom < H and _nodata(p[:, 0, :]).all():
                top, bottom = top + 1, bottom + 1
                p = x[:, top:bottom, left:right]
            moved = True
        if not moved and top > 0 and _nodata(p[:, -1, :]).all():
            while top > 0 and _nodata(p[:, -1, :]).all():
                top, bottom = top - 1, bottom - 1
                p = x[:, top:bottom, left:right]
        moved = False
        if right < W and _nodata(p[:, :, 0]).all():
            while right < W and _nodata(p[:, :, 0]).all():
                left, right = left + 1, right + 1
                p = x[:, top:bottom, left:right]
            moved = True
        if not moved and left > 0 and _nodata(p[:, :, -1]).all():
            while left > 0 and _nodata(p[:, :, -1]).all():
                left, right = left - 1, right - 1
                p = x[:, top:bottom, left:right]
    return _channel_norm(p.astype(np.float32)), (top, top + p.shape[1], left, left + p.shape[2])


def _gradient(size: int, overlap: int) -> np.ndarray:
    if overlap <= 0:
        return np.ones((size, size), np.float32)
    overlap = min(overlap, size // 2)
    g = np.full((size, size), overlap, np.float32)
    g[:, :overlap] = np.arange(1, overlap + 1)
    g[:, -overlap:] = np.arange(overlap, 0, -1)
    g /= overlap
    return (np.rot90(g) * g).astype(np.float32)


def classes(x3: np.ndarray) -> np.ndarray:
    """x3: (3, H, W) red, green, narrow near infrared (B04, B03, B8A) as reflectance, NaN (or 0) where no
    data. Returns (H, W) int8 classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow; 0 where no
    band has data (as the library does)."""
    x = np.asarray(x3, np.float32)
    size, overlap = _patch_size(x)
    grad = _gradient(size, overlap)
    tracker = np.zeros((4, *x.shape[1:]), np.float32)
    seen = set()
    for idx in _indexes(*x.shape[1:], size, overlap):
        p, new = _get_patch(x, idx)
        if p is None or new in seen:
            continue
        seen.add(new)
        logits = np.mean([_session(n).run(None, {"x": p[None]})[0][0] for n in NETS], axis=0)
        tracker[:, new[0]:new[1], new[2]:new[3]] += logits * grad
    out = tracker.argmax(0).astype(np.int8)
    out[_nodata(x).all(axis=0)] = 0
    return out


def unusable(x3: np.ndarray) -> np.ndarray:
    """True where OmniCloudMask finds thick cloud, thin cloud or cloud shadow."""
    return classes(x3) > 0
