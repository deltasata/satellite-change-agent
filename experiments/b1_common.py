"""B1 shared parts: the two kinds of image, the label, the counts, and the event bootstrap.

Images are given to every detector in one form: x of shape (7, H, W), float32 reflectance (0-1),
bands blue, green, red, nir (B08), nir08 (B8A), swir16 (B11), swir22 (B12); NaN where there is no image.
The pipeline's water rule uses green and B08. The models get the other six (MODEL_BANDS): the six
bands of the published Prithvi flood model. All seven exist in Sen1Floods11's L1C images and in
Earth Search's L2A images.

A pixel is scored when: the hand label is 0 or 1 (not -1), the image has data there, and, for L2A, the
scene classification does not call it cloud, cloud shadow or no data. The L2A cloud mask removes
pixels from the scoring for every detector equally, so the comparison stays paired.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.data import sen1floods11 as s1f
from core.data.cloud_mask import INVALID_SCL

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "b1"
BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
L1C_BANDS = [s1f.S2_INDEX[b] for b in BANDS7]
MODEL_BANDS = [0, 1, 2, 4, 5, 6]  # x[MODEL_BANDS]: blue, green, red, nir08, swir16, swir22
EVENTS = ("Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia",
          "Spain", "Sri-Lanka", "USA")


def label(chip: str) -> np.ndarray:
    return s1f.read(chip, "LabelHand")[0].astype(np.int8)  # 1 water, 0 dry, -1 not labelled


def image(chip: str, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(x, usable) for kind 'l1c' or 'l2a'; None if no such image exists (L2A of India 2016)."""
    if kind == "l1c":
        raw = s1f.read(chip, "S2Hand")[L1C_BANDS].astype(np.float32)
        nodata = (raw == 0).all(axis=0)
        x = raw * 1e-4
        x[:, nodata] = np.nan
        return x, ~nodata
    f = s1f.DIR / "S2L2A" / f"{chip}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    x = z["x"]
    usable = np.isfinite(x).all(axis=0) & ~np.isin(z["scl"], list(INVALID_SCL))
    return x, usable


def scored(chip: str, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(x, y, m): image, label, and the mask of pixels that are scored."""
    im = image(chip, kind)
    if im is None:
        return None
    x, usable = im
    y = label(chip)
    return x, y, usable & (y >= 0)


def counts(pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> list[int]:
    """[TP, FP, FN, TN] over the scored pixels, water = positive."""
    p, t = pred[m].astype(bool), y[m] == 1
    return [int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum()), int((~p & ~t).sum())]


def iou(c) -> float:
    if len(c) == 0:
        return float("nan")
    tp, fp, fn, _ = np.asarray(c, dtype=float).T if np.ndim(c) == 2 else c
    tp, fp, fn = (np.sum(v) for v in (tp, fp, fn))
    return float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan")


def save_counts(name: str, kind: str, per_chip: dict[str, list[int]]) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"counts_{kind}_{name}.json"
    p.write_text(json.dumps(per_chip))
    return p


def load_counts(name: str, kind: str) -> dict[str, list[int]]:
    return json.loads((RUNS / f"counts_{kind}_{name}.json").read_text())


def pooled_iou(per_chip: dict[str, list[int]], chips) -> float:
    return iou(np.array([per_chip[c] for c in chips if c in per_chip]))


def bootstrap_difference(a: dict, b: dict, chips, n: int = 10000, seed: int = 0):
    """Pooled IoU of b minus pooled IoU of a, with a 95 % interval from resampling whole flood events
    (cluster bootstrap): pixels and tiles of one event are not independent."""
    chips = [c for c in chips if c in a and c in b]
    events = sorted({s1f.event(c) for c in chips})
    A = {e: np.sum([a[c] for c in chips if s1f.event(c) == e], axis=0) for e in events}
    B = {e: np.sum([b[c] for c in chips if s1f.event(c) == e], axis=0) for e in events}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(events, size=len(events), replace=True)
        diffs.append(iou(np.sum([B[e] for e in pick], axis=0)) - iou(np.sum([A[e] for e in pick], axis=0)))
    d = iou(np.sum(list(B.values()), axis=0)) - iou(np.sum(list(A.values()), axis=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(d), float(lo), float(hi), len(events)
