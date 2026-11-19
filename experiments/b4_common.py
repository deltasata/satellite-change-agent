"""B4 shared pieces (protocol: docs/METHODS.md section 7): windows, median images, scored pixels, IoU of
"changed", the bootstrap over groups of cities.

Inputs per city are made once, in the pipeline environment (`python -m experiments.b4_common`), and saved to
data_cache/oscd/pc/inputs/<city>.npz, so that training (.venv-ml) needs only NumPy files:
    before, after  (7, H, W) float32: median of the clear dates of each period (NaN where none)
    usable         (H, W) bool: at least 2 clear dates before and 1 after (the rule's condition per pixel)
    label          (H, W) bool: urban change
    score          (H, W) float32: the pipeline rule's ratio map (after its 3 x 3 median filter), NaN where
                   not usable
    half           the half width of the windows used, 60 or 90 days
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import oscd_pc as D  # noqa: E402

RUNS = ROOT / "runs" / "b4"
INPUTS = D.OUT / "inputs"
CITIES = D.cities()
GROUPS = sorted({D.group(c) for c in CITIES})
BANDS7 = D.BANDS7
RULE_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)   # the fitted rule chooses from these (protocol)
# TerraMind's Level-2A pretraining mean and spread for these 7 bands (the same numbers as B3,
# experiments/b3_common.py), in reflectance
MEAN = np.array([1503.317, 1718.197, 1853.91, 3083.234, 3132.22, 2424.884, 1857.648], np.float32)[:, None, None] * 1e-4
STD = np.array([2141.107, 2038.973, 2134.138, 1871.918, 1753.829, 1434.261, 1334.311], np.float32)[:, None, None] * 1e-4
TM_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def windows(z: dict, half: int):
    w = timedelta(days=half)
    return (z["d1"] - w, z["d1"] + w), (z["d2"] - w, z["d2"] + w)


def _in(dates, win) -> np.ndarray:
    d = np.array(dates, dtype="datetime64[D]")
    return (d >= np.datetime64(win[0])) & (d <= np.datetime64(win[1]))


def make_inputs(city: str) -> dict:
    """Choose the windows (60 days, else 90, else drop), then the median images and the rule's map."""
    import xarray as xr
    from core.models.baseline import TS_BANDS, detect_change_timeseries
    z = D.load(city)
    x = z["x"].astype(np.float32)
    for half in (60, 90):
        wb, wa = windows(z, half)
        ib, ia = _in(z["dates"], wb), _in(z["dates"], wa)
        nb = np.isfinite(x[ib, 2]).sum(0)
        na = np.isfinite(x[ia, 2]).sum(0)
        usable = (nb >= 2) & (na >= 1)
        if usable.mean() >= 0.5:
            break
    else:
        return {"city": city, "dropped": True, "usable_fraction": float(usable.mean())}
    sel = ib | ia
    ds = xr.Dataset({b: (("time", "y", "x"), x[sel, k]) for k, b in enumerate(BANDS7)},
                    coords={"time": np.array(z["dates"], dtype="datetime64[ns]")[sel],
                            "y": np.arange(x.shape[2]) * -10.0, "x": np.arange(x.shape[3]) * 10.0})
    r = detect_change_timeseries(ds, wb, wa, bands=TS_BANDS)
    assert r["status"] == "ok", r
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = np.nanmedian(x[ib], axis=0)
        after = np.nanmedian(x[ia], axis=0)
    INPUTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INPUTS / f"{city}.npz", before=before.astype(np.float32), after=after.astype(np.float32),
                        usable=usable, label=z["label"], score=r["_score"].astype(np.float32), half=half)
    return {"city": city, "dropped": False, "half": half, "n_dates_before": int(ib.sum()), "n_dates_after": int(ia.sum()),
            "usable_fraction": float(usable.mean()), "changed_label_fraction": float(z["label"][usable].mean()),
            "shape": list(x.shape[2:])}


def inputs(city: str) -> dict | None:
    f = INPUTS / f"{city}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def kept() -> list[str]:
    """Cities that met the pipeline's condition (the others are dropped for every detector)."""
    return [c for c in CITIES if (INPUTS / f"{c}.npz").exists()]


def scale(x: np.ndarray) -> np.ndarray:
    """Network input: (x - mean) / spread with TerraMind's statistics; missing values -> 0 (the mean)."""
    return np.nan_to_num((x - MEAN) / STD, nan=0.0).astype(np.float32)


def confusion(pred: np.ndarray, z: dict) -> list[int]:
    """[TP, FP, FN, TN] of "changed" over the scored pixels (usable)."""
    m, y, p = z["usable"], z["label"].astype(bool), pred.astype(bool)
    return [int((p & y & m).sum()), int((p & ~y & m).sum()), int((~p & y & m).sum()), int((~p & ~y & m).sum())]


def iou(cm) -> float:
    tp, fp, fn, _ = np.asarray(cm, float)
    return float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan")


def summary(cms: dict) -> dict:
    tot = np.sum(list(cms.values()), axis=0)
    tp, fp, fn, tn = tot.astype(float)
    i = iou(tot)
    return {"iou": i, "f1": 2 * i / (1 + i), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn), "per_city": {c: iou(v) for c, v in cms.items()}}


def bootstrap_difference(a: dict, b: dict, n: int = 10000, seed: int = 0):
    """Pooled IoU of b minus that of a; 95 % interval from resampling whole groups of cities."""
    cities = [c for c in a if c in b]
    groups = sorted({D.group(c) for c in cities})
    A = np.stack([np.sum([a[c] for c in cities if D.group(c) == g], axis=0) for g in groups]).astype(float)
    B = np.stack([np.sum([b[c] for c in cities if D.group(c) == g], axis=0) for g in groups]).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        diffs[k] = iou(B[pick].sum(0)) - iou(A[pick].sum(0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(iou(B.sum(0)) - iou(A.sum(0))), float(lo), float(hi), len(groups)


def save(name: str, cms: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"cm_{name}.json"
    p.write_text(json.dumps(cms))
    return p


def read(name: str) -> dict:
    return json.loads((RUNS / f"cm_{name}.json").read_text())


def main() -> None:
    rows = []
    for c in CITIES:
        try:
            r = make_inputs(c)
        except FileNotFoundError as e:
            r = {"city": c, "error": str(e)[:200]}
        rows.append(r)
        print(r, flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "inputs.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
