"""Which pixels changed: a trained network on the two median images (construction: milestone B4; burnt
vegetation: milestone B5, names starting with "burn-").

The pipeline's rule (core/models/baseline.py) marks a pixel changed when its change is large compared
with the before images' date-to-date scatter. B4 tests trained Siamese networks against it on OSCD's
hand labels of urban change (protocol: docs/METHODS.md section 7). A network sees the before and the
after median image (7 bands) and gives each pixel a probability of urban change.

The networks run from ONNX files on the CPU (data_cache/models/change_<model>_<group>_s<seed>.onnx, made
by experiments/b4_export.py); the 3 seeds' probabilities are averaged, and a pixel is "changed" where the
mean is above 0.5. Inputs are scaled as in training; missing values become 0 (the mean). The image is
cut into 224 x 224 windows at stride 144 and the overlaps are averaged; an area smaller than 224 pixels
is first extended by mirroring its edge.

Burn networks (B5, protocol docs/METHODS.md section 8) work the same way, with their own files
(data_cache/models/burn_<model>_s<seed>.onnx, made by experiments/b5_export.py) and the same inputs.

"burn-utae" (B7's U-TAE, adopted in B8, protocol docs/METHODS.md section 11) sees every clear date instead of
the two medians (files made by experiments/b8_export.py). The dates are chosen as in its training copy
(core/data/impactmesh_ts.py): per period, every date with at least 1 % clear pixels in the area, at most the
12 clearest, in time order. Each date is 8 channels (7 bands scaled as above, missing 0, and a clear flag),
with its day counted from the first day of the after period and its period (0 before, 1 after).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("EO_DATA_CACHE", Path(__file__).resolve().parents[2] / "data_cache")) / "models"
NAMES = {"rule": "change rule: ratio to the before scatter above 4",
         "fsd": "FC-Siam-diff (milestone B4)", "terramind": "TerraMind-1.0-base, Siamese (milestone B4)",
         "burn-unet": "U-Net, burnt pixels (milestone B5)",
         "burn-terramind": "TerraMind-1.0-base, Siamese, burnt pixels (milestone B5)",
         "burn-utae": "U-TAE on every clear date, burnt pixels (milestones B7, B8)",
         "terramind+water": "TerraMind-1.0-base, Siamese (milestone B4), minus the water of the after image (B9)"}
WATER_SUFFIX = "+water"            # B9: remove the water network's water in the after image from the change map
SERIES = {"burn-utae"}            # networks that see every clear date, not two medians
MIN_CLEAR, MAX_PER_PERIOD = 0.01, 12
CROP, STRIDE = 224, 144
MEAN = np.array([1503.317, 1718.197, 1853.91, 3083.234, 3132.22, 2424.884, 1857.648], np.float32)[:, None, None] * 1e-4
STD = np.array([2141.107, 2038.973, 2134.138, 1871.918, 1753.829, 1434.261, 1334.311], np.float32)[:, None, None] * 1e-4


def _parse(name: str) -> tuple[str, str]:
    """'fsd' -> ('fsd', 'ALL'); 'fsd-without-paris_bercy' -> ('fsd', 'paris_bercy')."""
    model, _, group = name.partition("-without-")
    return model, group or "ALL"


def available(name: str) -> bool:
    """True if the network's files are present (the rule needs none). The construction network is not
    published (its OSCD training labels are CC BY-NC-SA), so a public copy may not have it."""
    if name == "rule":
        return True
    name = name.replace(WATER_SUFFIX, "")
    if name.startswith("burn-"):
        return all((MODELS / f"burn_{name[5:]}_s{s}.onnx").exists() for s in range(3))
    model, group = _parse(name)
    return all((MODELS / f"change_{model}_{group}_s{s}.onnx").exists() for s in range(3))


@lru_cache(maxsize=4)
def _sessions(name: str):
    import onnxruntime as ort
    if name.startswith("burn-"):
        files = [MODELS / f"burn_{name[5:]}_s{s}.onnx" for s in range(3)]
        how = "experiments.b8_export" if name in SERIES else "experiments.b5_export"
    else:
        model, group = _parse(name)
        files = [MODELS / f"change_{model}_{group}_s{s}.onnx" for s in range(3)]
        how = "experiments.b4_export"
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} missing in {MODELS}; run .venv-ml/bin/python -m {how}")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    return [ort.InferenceSession(str(f), opt, providers=["CPUExecutionProvider"]) for f in files]


def scale(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - MEAN) / STD, nan=0.0).astype(np.float32)


def probability(before: np.ndarray, after: np.ndarray, name: str) -> np.ndarray:
    """before, after: (7, H, W) reflectance (blue, green, red, nir, nir08, swir16, swir22), NaN where
    missing. Returns the mean probability of the network's class over the 3 seeds (urban change, or burnt
    for the "burn-" networks), (H, W)."""
    b, a = scale(before), scale(after)
    H, W = b.shape[1:]
    ph, pw = max(0, CROP - H), max(0, CROP - W)
    if ph or pw:
        pad = ((0, 0), (0, ph), (0, pw))
        b, a = np.pad(b, pad, mode="reflect"), np.pad(a, pad, mode="reflect")
    Hp, Wp = b.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(Hp) for j in starts(Wp)]
    xb = np.stack([b[:, i:i + CROP, j:j + CROP] for i, j in wins])
    xa = np.stack([a[:, i:i + CROP, j:j + CROP] for i, j in wins])
    prob, hits = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    for s in _sessions(name):
        for k in range(0, len(wins), 8):
            logits = s.run(None, {"before": xb[k:k + 8], "after": xa[k:k + 8]})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            p = e[:, 1] / e.sum(1)
            for (i, j), pi in zip(wins[k:k + 8], p):
                prob[i:i + CROP, j:j + CROP] += pi
                hits[i:i + CROP, j:j + CROP] += 1
    return (prob / hits)[:H, :W]


def select(clear: np.ndarray) -> list[int]:
    """Indices of the dates kept in one period from each date's clear fraction: at least MIN_CLEAR, at most
    MAX_PER_PERIOD (the clearest), in time order (the rule of core/data/impactmesh_ts.py)."""
    idx = [k for k in range(len(clear)) if clear[k] >= MIN_CLEAR]
    if len(idx) > MAX_PER_PERIOD:
        idx = sorted(sorted(idx, key=lambda k: -clear[k])[:MAX_PER_PERIOD])
    return idx


def probability_series(x: np.ndarray, days: np.ndarray, period: np.ndarray, name: str) -> np.ndarray:
    """x: (T, 7, H, W) reflectance of the chosen dates, NaN where not clear; days: (T,) days since the after
    period began; period: (T,) 0 before, 1 after. Returns the mean probability of "burnt" over the 3 seeds,
    (H, W). Windows of 224 at stride 144 as above; one window at a time (the number of dates makes them large)."""
    clear = np.isfinite(x).all(1, keepdims=True).astype(np.float32)
    z = np.concatenate([np.stack([scale(d) for d in x]), clear], axis=1)          # (T, 8, H, W)
    H, W = z.shape[2:]
    ph, pw = max(0, CROP - H), max(0, CROP - W)
    if ph or pw:
        z = np.pad(z, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
    Hp, Wp = z.shape[2:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(Hp) for j in starts(Wp)]
    d = np.asarray(days, np.int64)[None]
    p = np.asarray(period, np.int64)[None]
    pad = np.zeros(d.shape, bool)
    prob, hits = np.zeros((Hp, Wp), np.float32), np.zeros((Hp, Wp), np.float32)
    for s in _sessions(name):
        for i, j in wins:
            logits = s.run(None, {"x": z[None, :, :, i:i + CROP, j:j + CROP], "days": d, "period": p, "pad": pad})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            prob[i:i + CROP, j:j + CROP] += (e[:, 1] / e.sum(1))[0]
            hits[i:i + CROP, j:j + CROP] += 1
    return (prob / hits)[:H, :W]


def changed(before: np.ndarray, after: np.ndarray, name: str) -> np.ndarray:
    return probability(before, after, name) > 0.5
