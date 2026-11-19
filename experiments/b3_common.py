"""B3 shared parts: images, labels, scored pixels, counts, IoU and the place bootstrap.
Protocol: docs/METHODS.md section 6.

Kinds of image (test part):
  es    the Earth Search copy (core/data/cloudsen12_es.py): the kind the pipeline reads. The decision.
  gee   CloudSEN12's own Level-2A images (Google Earth Engine). Reported. Validation and training use it.

Classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow. "Unusable" = 1, 2 or 3.

Per image and detector we keep a 4 x 4 table of pixel counts, cm[label class, predicted class]. A detector
that only says usable / unusable predicts 0 or 1 (1 standing for "unusable", not for "thick cloud").
From the tables:
  - the main number: IoU of "unusable", pooled over images (counts summed first, as in B1);
  - clouds let through: unusable pixels called usable, divided by all unusable pixels;
  - good pixels thrown away: clear pixels called unusable, divided by all clear pixels;
  - IoU of thick, thin and shadow, for detectors that tell them apart.

A pixel is scored when the label is 0-3, all 7 bands of the data set's image are present and, on the
test part, all 7 bands of the Earth Search copy are present. The same pixels for every detector and for
both kinds of image. Test images without an Earth Search copy are left out everywhere.
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.data.cloudsen12 import DIR

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "b3"
SCL_UNUSABLE = (3, 8, 9, 10)           # the pipeline's rule (core/data/cloud_mask.py), cloud and shadow classes
SCL_CLASS = {3: 3, 8: 1, 9: 1, 10: 2}  # SCL class -> label class, for the per-class IoU
# TerraMind's Level-2A pretraining mean and spread for B02 B03 B04 B08 B8A B11 B12 (terramind_register.py,
# "untok_sen2l2a@224"; units of 1e-4 reflectance). The U-Net gets the same numbers.
MEAN = np.array([1503.317, 1718.197, 1853.91, 3083.234, 3132.22, 2424.884, 1857.648], np.float32)[:, None, None] * 1e-4
STD = np.array([2141.107, 2038.973, 2134.138, 1871.918, 1753.829, 1434.261, 1334.311], np.float32)[:, None, None] * 1e-4
TM_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]


@lru_cache(maxsize=1)
def meta() -> dict[str, dict]:
    with open(DIR / "meta.csv") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def ids(split: str) -> list[str]:
    """Images of one part that are on disk; for the test part, only those with an Earth Search copy."""
    out = [i for i, r in meta().items() if r["split"] == split and (DIR / split / f"{i}.npz").exists()]
    if split == "test":
        out = [i for i in out if (DIR / "test_es" / f"{i}.npz").exists()]
    return sorted(out)


def place(i: str) -> str:
    return meta()[i]["roi_id"]


def load(i: str, split: str) -> dict:
    return dict(np.load(DIR / split / f"{i}.npz"))


def reflectance(raw: np.ndarray) -> np.ndarray:
    """uint16 (x 10 000) of the data set -> reflectance, NaN where 0 (no data)."""
    return np.where(raw == 0, np.nan, raw.astype(np.float32) * 1e-4)


def image(i: str, split: str, kind: str) -> np.ndarray:
    if kind == "gee":
        return reflectance(load(i, split)["x"])
    return np.load(DIR / "test_es" / f"{i}.npz")["x"]


def scl(i: str, split: str, kind: str) -> np.ndarray:
    if kind == "gee":
        return load(i, split)["scl"]
    return np.load(DIR / "test_es" / f"{i}.npz")["scl"]


def scored(i: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """(y, m): the label and the scored pixels."""
    d = load(i, split)
    y = d["y"].astype(np.int64)
    m = (y <= 3) & (d["x"] > 0).all(axis=0)
    if split == "test":
        m &= np.isfinite(np.load(DIR / "test_es" / f"{i}.npz")["x"]).all(axis=0)
    return y, m


def confusion(pred: np.ndarray, y: np.ndarray, m: np.ndarray) -> list[list[int]]:
    """4 x 4 counts cm[label, predicted] over the scored pixels."""
    cm = np.bincount(y[m] * 4 + pred[m].astype(np.int64), minlength=16).reshape(4, 4)
    return cm.tolist()


def scl_classes(s: np.ndarray, unusable=SCL_UNUSABLE, widen: int = 0) -> np.ndarray:
    """SCL -> predicted classes. widen = 0: per class (3 shadow, 8/9 thick, 10 thin). Otherwise (and for any
    other set of classes) only usable / unusable: 1 = unusable, the area widened by `widen` pixels."""
    if tuple(unusable) == SCL_UNUSABLE and widen == 0:
        out = np.zeros(s.shape, np.int64)
        for k, c in SCL_CLASS.items():
            out[s == k] = c
        return out
    bad = np.isin(s, list(unusable))
    if widen:
        from scipy.ndimage import binary_dilation
        bad = binary_dilation(bad, structure=np.ones((2 * widen + 1, 2 * widen + 1), bool))
    return bad.astype(np.int64)


def binary(cm) -> tuple[float, float, float, float]:
    """[TP, FP, FN, TN] for "unusable" from a (summed) 4 x 4 table."""
    cm = np.asarray(cm, dtype=float)
    tp, fn = cm[1:, 1:].sum(), cm[1:, 0].sum()
    fp, tn = cm[0, 1:].sum(), cm[0, 0]
    return tp, fp, fn, tn


def iou_unusable(cm) -> float:
    tp, fp, fn, _ = binary(cm)
    return float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan")


def summary(cms: dict, chosen: list[str]) -> dict:
    cm = np.sum([cms[i] for i in chosen], axis=0)
    tp, fp, fn, tn = binary(cm)
    out = {"iou": iou_unusable(cm), "clouds_let_through": fn / (tp + fn), "good_pixels_thrown_away": fp / (fp + tn),
           "n_images": len(chosen), "n_places": len({place(i) for i in chosen})}
    for k, name in ((1, "thick"), (2, "thin"), (3, "shadow")):
        t = cm[k, k]
        out[f"iou_{name}"] = float(t / (cm[k, :].sum() + cm[:, k].sum() - t)) if cm[:, k].sum() else None
    return out


def bootstrap_difference(a: dict, b: dict, chosen: list[str], n: int = 10000, seed: int = 0):
    """Pooled "unusable" IoU of b minus that of a, with a 95 % interval from resampling whole places: the
    5 dates of one place share the ground and are not independent."""
    chosen = [i for i in chosen if i in a and i in b]
    places = sorted({place(i) for i in chosen})
    A = {p: np.sum([a[i] for i in chosen if place(i) == p], axis=0) for p in places}
    B = {p: np.sum([b[i] for i in chosen if place(i) == p], axis=0) for p in places}
    Ab, Bb = np.stack([A[p] for p in places]), np.stack([B[p] for p in places])
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(places), len(places))
        diffs[k] = iou_unusable(Bb[pick].sum(0)) - iou_unusable(Ab[pick].sum(0))
    d = iou_unusable(Bb.sum(0)) - iou_unusable(Ab.sum(0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(d), float(lo), float(hi), len(places)


def save(name: str, kind: str, cms: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"cm_{kind}_{name}.json"
    p.write_text(json.dumps(cms))
    return p


def read(name: str, kind: str) -> dict:
    return json.loads((RUNS / f"cm_{kind}_{name}.json").read_text())


def scale(x: np.ndarray) -> np.ndarray:
    """Network input: (x - mean) / spread with TerraMind's statistics; missing values -> 0 (the mean)."""
    return np.nan_to_num((x - MEAN) / STD, nan=0.0).astype(np.float32)


def same_day_ids() -> set[str]:
    """The 16 test images taken on the same day, in the same Sentinel-2 tile, as an image of OmniCloudMask's
    training material (found from the catalogue before the protocol was written). Written by b3_rules."""
    p = RUNS / "same_day_as_ocm_training.json"
    return set(json.loads(p.read_text())) if p.exists() else set()
