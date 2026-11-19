"""Which pixels changed through construction: a trained network on the two median images (milestone B4).

The pipeline's rule (core/models/baseline.py) marks a pixel changed when its change is large compared
with the before images' date-to-date scatter. B4 tests trained Siamese networks against it on OSCD's
hand labels of urban change (protocol: docs/METHODS.md section 7). A network sees the before and the
after median image (7 bands) and gives each pixel a probability of urban change.

The networks run from ONNX files on the CPU (data_cache/models/change_<model>_<group>_s<seed>.onnx, made
by experiments/b4_export.py); the 3 seeds' probabilities are averaged, and a pixel is "changed" where the
mean is above 0.5. Inputs are scaled as in training; missing values become 0 (the mean). The image is
cut into 224 x 224 windows at stride 144 and the overlaps are averaged; an area smaller than 224 pixels
is first extended by mirroring its edge.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS = Path(os.environ.get("EO_DATA_CACHE", Path(__file__).resolve().parents[2] / "data_cache")) / "models"
NAMES = {"rule": "change rule: ratio to the before scatter above 4",
         "fsd": "FC-Siam-diff (milestone B4)", "terramind": "TerraMind-1.0-base, Siamese (milestone B4)"}
CROP, STRIDE = 224, 144
MEAN = np.array([1503.317, 1718.197, 1853.91, 3083.234, 3132.22, 2424.884, 1857.648], np.float32)[:, None, None] * 1e-4
STD = np.array([2141.107, 2038.973, 2134.138, 1871.918, 1753.829, 1434.261, 1334.311], np.float32)[:, None, None] * 1e-4


def _parse(name: str) -> tuple[str, str]:
    """'fsd' -> ('fsd', 'ALL'); 'fsd-without-paris_bercy' -> ('fsd', 'paris_bercy')."""
    model, _, group = name.partition("-without-")
    return model, group or "ALL"


@lru_cache(maxsize=4)
def _sessions(name: str):
    import onnxruntime as ort
    model, group = _parse(name)
    files = [MODELS / f"change_{model}_{group}_s{s}.onnx" for s in range(3)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{missing} missing in {MODELS}; run .venv-ml/bin/python -m experiments.b4_export")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "8"))
    return [ort.InferenceSession(str(f), opt, providers=["CPUExecutionProvider"]) for f in files]


def scale(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num((x - MEAN) / STD, nan=0.0).astype(np.float32)


def probability(before: np.ndarray, after: np.ndarray, name: str) -> np.ndarray:
    """before, after: (7, H, W) reflectance (blue, green, red, nir, nir08, swir16, swir22), NaN where
    missing. Returns the mean probability of urban change over the 3 seeds, (H, W)."""
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


def changed(before: np.ndarray, after: np.ndarray, name: str) -> np.ndarray:
    return probability(before, after, name) > 0.5
