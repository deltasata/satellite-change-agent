"""B2 shared parts: radar and optical inputs of Sen1Floods11, the scored pixels, the official parts.

Kinds of image:
  rtc   our radar copy from Planetary Computer (gamma naught, terrain-flattened), VV and VH in dB.
        This is what the pipeline reads. Part A (decision) uses it, leaving one event out.
  grd   the data set's own radar (S1Hand, Earth Engine sigma naught), VV and VH in dB. Part B.
  l1c   the data set's own optical image (S2Hand), all 13 bands as reflectance (0-1). Part B.

Inputs to a network ("inputs"): "s1" (2 channels), "s2" (13), "s1s2" (15, radar first).
Every channel is scaled with TerraMind's pretraining mean and spread for that kind of image, so the
U-Net and TerraMind see the same numbers. Missing values become 0 after scaling (the mean).

A pixel is scored when the hand label is 0 or 1 and every input image has a value there. The counts,
IoU and event bootstrap are those of B1 (experiments/b1_common.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.data import sen1floods11 as s1f
from experiments.b1_common import bootstrap_difference, counts, iou, label, pooled_iou  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "b2"

# TerraMind-1.0 pretraining statistics (terratorch terramind_register.py, "untok_..." entries)
STATS = {
    "rtc": ([-10.93, -17.329], [4.391, 4.459]),
    "grd": ([-12.599, -20.293], [5.195, 5.890]),
    "l1c": ([2357.089, 2137.385, 2018.788, 2082.986, 2295.651, 2854.537, 3122.849, 3040.56, 3306.481,
             1473.847, 506.07, 2472.825, 1838.929],
            [1624.683, 1675.806, 1557.708, 1833.702, 1823.738, 1733.977, 1732.131, 1679.732, 1727.26,
             1024.687, 442.165, 1331.411, 1160.419]),
}
CHANNELS = {"s1": 2, "s2": 13, "s1s2": 15}


def radar(chip: str, kind: str) -> np.ndarray | None:
    """(2, 512, 512) VV, VH in dB, NaN where missing; None if this tile has no such image."""
    if kind == "grd":
        x = s1f.read(chip, "S1Hand").astype(np.float32)
        x[~np.isfinite(x)] = np.nan
        return x
    f = s1f.DIR / "S1RTC" / f"{chip}.npz"
    return np.load(f)["x"] if f.exists() else None


def optical_l1c(chip: str) -> np.ndarray:
    """(13, 512, 512) reflectance in units of 1e-4 (as TerraMind's statistics), NaN where no image."""
    raw = s1f.read(chip, "S2Hand").astype(np.float32)
    raw[:, (raw == 0).all(axis=0)] = np.nan
    return raw


def inputs(chip: str, inp: str, radar_kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(x, present): x scaled, NaN -> 0; present = every input image has a value at the pixel."""
    parts, present = [], np.ones((512, 512), bool)
    if inp in ("s1", "s1s2"):
        r = radar(chip, radar_kind)
        if r is None:
            return None
        parts.append((r, *STATS[radar_kind]))
    if inp in ("s2", "s1s2"):
        parts.append((optical_l1c(chip), *STATS["l1c"]))
    xs = []
    for a, mean, std in parts:
        present &= np.isfinite(a).all(axis=0)
        xs.append((a - np.array(mean, np.float32)[:, None, None]) / np.array(std, np.float32)[:, None, None])
    x = np.nan_to_num(np.concatenate(xs), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, present


def scored(chip: str, inp: str, radar_kind: str):
    """(x, y, m) or None: scaled input, hand label, scored pixels."""
    got = inputs(chip, inp, radar_kind)
    if got is None:
        return None
    x, present = got
    y = label(chip)
    return x, y, present & (y >= 0)


def official_parts() -> dict[str, list[str]]:
    sp = s1f.splits()
    out: dict[str, list[str]] = {}
    for c in s1f.chips():
        out.setdefault(sp[c], []).append(c)
    return out


def miou(c) -> float:
    """Mean of the water IoU and the dry-land IoU, from pooled [TP, FP, FN, TN]."""
    tp, fp, fn, tn = np.sum(np.atleast_2d(c), axis=0).astype(float)
    return float((tp / (tp + fp + fn) + tn / (tn + fn + fp)) / 2)
