"""B5 shared pieces (protocol: docs/METHODS.md section 8): which patches, scored pixels, IoU of "burnt",
the bootstrap over fires, network scaling, and the packed training arrays.

Inputs come from core/data/impactmesh_es.py (data_cache/burn/impactmesh/es/<patch>.npz).

    python -m experiments.b5_common          # counts, dropped patches -> runs/b5/inputs.json
    python -m experiments.b5_common pack     # the training patches as one array (for b5_train)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from core.data import impactmesh_es as D
from experiments.b4_common import MEAN, STD, TM_BANDS  # noqa: F401  (TerraMind's Level-2A statistics)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "b5"
PACKED = D.DIR / "packed"
RATIO_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
DNBR_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.27, 0.35, 0.44)
NIR, SWIR22 = 3, 6     # B08 and B12 in the 7 bands


def fire(name: str) -> str:
    return name.split("_")[0]


def built(part: str) -> list[str]:
    """Patches of a part ("test" = the held-out fires, "train" = training and validation without the excluded
    fires) that were built and not dropped."""
    parts = ("test_holdout",) if part == "test" else ("train", "val")
    out = []
    for p in D.patches():
        if p["part"] in parts and D.used(p):
            f = D.OUT / f"{p['name']}.npz"
            if f.exists() and "dropped" not in np.load(f).files:
                out.append(p["name"])
    return out


def load(name: str) -> dict:
    z = np.load(D.OUT / f"{name}.npz")
    out = {k: z[k] for k in z.files}
    for k in ("before", "after", "score"):
        out[k] = out[k].astype(np.float32)
    return out


def scored(z: dict) -> np.ndarray:
    """Labelled pixels (0 or 1) with at least 2 clear observations before and 1 after."""
    return z["usable"] & (z["label"] >= 0)


def confusion(pred: np.ndarray, z: dict) -> list[int]:
    """[TP, FP, FN, TN] of "burnt" over the scored pixels."""
    m, y, p = scored(z), z["label"] == 1, pred.astype(bool)
    return [int((p & y & m).sum()), int((p & ~y & m).sum()), int((~p & y & m).sum()), int((~p & ~y & m).sum())]


def iou(cm) -> float:
    tp, fp, fn, _ = np.asarray(cm, float)
    return float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan")


def summary(cms: dict) -> dict:
    tot = np.sum(list(cms.values()), axis=0).astype(float)
    tp, fp, fn, _ = tot
    i = iou(tot)
    per_fire = {}
    for n, v in cms.items():
        per_fire.setdefault(fire(n), []).append(v)
    fire_iou = {f: iou(np.sum(v, axis=0)) for f, v in per_fire.items()}
    fi = [v for v in fire_iou.values() if np.isfinite(v)]
    return {"iou": i, "f1": 2 * i / (1 + i), "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "mean_fire_iou": float(np.mean(fi)),
            "per_fire": fire_iou}


def bootstrap_difference(a: dict, b: dict, n: int = 10000, seed: int = 0):
    """Pooled IoU of b minus that of a; 95 % interval from resampling whole fires."""
    names = [k for k in a if k in b]
    fires = sorted({fire(k) for k in names})
    A = np.stack([np.sum([a[k] for k in names if fire(k) == f], axis=0) for f in fires]).astype(float)
    B = np.stack([np.sum([b[k] for k in names if fire(k) == f], axis=0) for f in fires]).astype(float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for k in range(n):
        pick = rng.integers(0, len(fires), len(fires))
        diffs[k] = iou(B[pick].sum(0)) - iou(A[pick].sum(0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(iou(B.sum(0)) - iou(A.sum(0))), float(lo), float(hi), len(fires)


def scale(x: np.ndarray) -> np.ndarray:
    """Network input: (x - mean) / spread with TerraMind's statistics; missing values -> 0 (the mean)."""
    return np.nan_to_num((x - MEAN) / STD, nan=0.0).astype(np.float32)


def nbr(x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return (x[NIR] - x[SWIR22]) / (x[NIR] + x[SWIR22])


def dnbr(z: dict) -> np.ndarray:
    return np.nan_to_num(nbr(z["before"]) - nbr(z["after"]), nan=0.0)


def save(name: str, cms: dict) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    p = RUNS / f"cm_{name}.json"
    p.write_text(json.dumps(cms))
    return p


def read(name: str) -> dict:
    return json.loads((RUNS / f"cm_{name}.json").read_text())


def pack(part: str = "train") -> None:
    """One file per array, read as memory maps during training: before/after float16 (N, 7, 256, 256),
    y int8 (N, 256, 256): 1 burnt, 0 not, -1 not used in the loss (no label, or a band missing)."""
    names = built(part)
    PACKED.mkdir(parents=True, exist_ok=True)
    n = len(names)
    B = np.lib.format.open_memmap(PACKED / f"{part}_before.npy", "w+", np.float16, (n, 7, 256, 256))
    A = np.lib.format.open_memmap(PACKED / f"{part}_after.npy", "w+", np.float16, (n, 7, 256, 256))
    Y = np.lib.format.open_memmap(PACKED / f"{part}_y.npy", "w+", np.int8, (n, 256, 256))
    for k, name in enumerate(names):
        z = np.load(D.OUT / f"{name}.npz")
        B[k], A[k] = z["before"], z["after"]
        ok = np.isfinite(z["before"].astype(np.float32)).all(0) & np.isfinite(z["after"].astype(np.float32)).all(0)
        Y[k] = np.where(ok, z["label"], -1)
    B.flush(), A.flush(), Y.flush()
    (PACKED / f"{part}_names.json").write_text(json.dumps(names))
    print(f"packed {n} {part} patches")


def main() -> None:
    rows, c = [], Counter()
    for p in D.patches():
        if not D.used(p):
            continue
        part = "test" if p["part"] == "test_holdout" else "train"
        f = D.OUT / f"{p['name']}.npz"
        if not f.exists():
            c[(part, "missing")] += 1
            continue
        z = np.load(f)
        if "dropped" in z.files:
            c[(part, "dropped")] += 1
            rows.append({"name": p["name"], "dropped": True, "usable_fraction": float(z["usable_fraction"])})
            continue
        c[(part, str(z["stage"]))] += 1
    out = {"counts": {f"{a} {b}": v for (a, b), v in sorted(c.items())},
           "dropped_fires": dict(Counter(fire(r["name"]) for r in rows))}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "inputs.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    pack() if len(sys.argv) > 1 and sys.argv[1] == "pack" else main()
