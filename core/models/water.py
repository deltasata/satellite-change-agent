"""Which pixels of a composite image are water: the rule NDWI > 0, or the U-Net chosen in milestone B1.

B1 (docs/METHODS.md section 3) compared, on 376 hand-labelled Sen1Floods11
tiles in Level-2A, each detector trained with the tested flood event left out:
    NDWI > 0 (the old rule)                 water IoU 0.690
    U-Net trained from nothing              water IoU 0.833   (difference +0.143, 95 % interval +0.083..+0.219)
So the U-Net replaces the rule for flood questions in measure_change. The step functions of mcp_server/server.py keep
NDWI > 0 as their default, so their recorded results still describe the code.

Input of both detectors: c, shape (7, H, W), reflectance 0-1, NaN where no clear image, bands
BANDS7 = blue, green, red, nir (B08), nir08 (B8A), swir16 (B11), swir22 (B12).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "data_cache" / "models"
BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
MODEL_BANDS = [0, 1, 2, 4, 5, 6]                     # the six bands the U-Net reads
MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], np.float32)[:, None, None] * 1e-4
STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], np.float32)[:, None, None] * 1e-4
CROP, STRIDE = 224, 144
NAMES = {"ndwi": "NDWI > 0 (green and B08)",
         "unet": "U-Net trained on Sen1Floods11 in Level-2A (milestone B1)"}


def ndwi_water(c: np.ndarray) -> np.ndarray:
    green, nir = c[1], c[3]
    with np.errstate(all="ignore"):
        return (green - nir) / (green + nir) > 0


@lru_cache(maxsize=12)
def _session(weights: str):
    import onnxruntime as ort
    path = MODELS / f"water_unet_{weights}.onnx"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; make it with: .venv-ml/bin/python -m experiments.b1_export")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def unet_probability(c: np.ndarray, weights: str = "ALL") -> np.ndarray:
    """Water probability per pixel, from 224 x 224 windows at stride 144, overlaps averaged.
    An image smaller than 224 pixels is padded with the mean value (0 after scaling) and cut back."""
    x = np.nan_to_num((c[MODEL_BANDS] - MEAN) / STD, nan=0.0).astype(np.float32)
    H, W = x.shape[1:]
    if H < CROP or W < CROP:
        x = np.pad(x, ((0, 0), (0, max(0, CROP - H)), (0, max(0, CROP - W))))
    h, w = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(h) for j in starts(w)]
    logits = _session(weights).run(None, {"x": np.stack([x[:, i:i + CROP, j:j + CROP] for i, j in wins])})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (e[:, 1] / e.sum(axis=1)).astype(np.float32)
    prob, hits = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for (i, j), pi in zip(wins, p):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return (prob / hits)[:H, :W]


# ---------------------------------------------------------------- radar (milestone B2)
# B2 (docs/METHODS.md section 4) compared radar water detectors on 379 hand-labelled Sen1Floods11 tiles
# in Planetary Computer's RTC radar, each tested on a flood event it was not trained on:
#     VH < -20.5 dB (threshold fitted on the training events)   water IoU 0.477
#     U-Net trained from nothing                                0.558  (interval of the gain touches 0)
#     TerraMind-1.0-base fine-tuned, mean of 3 seeds            0.611  (+0.134, interval +0.090..+0.192)
# Input: x of shape (2, H, W), VV and VH in dB, NaN where no radar value.
RADAR_MEAN = np.array([-10.93, -17.329], np.float32)[:, None, None]     # TerraMind's S1RTC statistics
RADAR_STD = np.array([4.391, 4.459], np.float32)[:, None, None]
RADAR_SEEDS = 3
RADAR_NAME = "TerraMind-1.0-base fine-tuned on Sen1Floods11 radar (milestone B2)"


@lru_cache(maxsize=6)
def _radar_session(name: str):
    import onnxruntime as ort
    path = MODELS / f"radar_terramind_{name}.onnx"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; make it with: .venv-ml/bin/python -m experiments.b2_export")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _windows(x: np.ndarray, run) -> np.ndarray:
    """Probability per pixel from CROP windows at STRIDE (overlaps averaged); pads small images."""
    H, W = x.shape[1:]
    if H < CROP or W < CROP:
        x = np.pad(x, ((0, 0), (0, max(0, CROP - H)), (0, max(0, CROP - W))))
    h, w = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(h) for j in starts(w)]
    logits = run(np.stack([x[:, i:i + CROP, j:j + CROP] for i, j in wins]))
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = (e[:, 1] / e.sum(axis=1)).astype(np.float32)
    prob, hits = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    for (i, j), pi in zip(wins, p):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return (prob / hits)[:H, :W]


def radar_probability(x: np.ndarray, fold: str = "ALL") -> np.ndarray:
    """Water probability of one radar scene: the mean of the 3 seeds' networks. NaN where no radar value.
    fold 'ALL' = trained on every event; '<event>' = that event left out (for tests on that event)."""
    present = np.isfinite(x).all(axis=0)
    xs = np.nan_to_num((x - RADAR_MEAN) / RADAR_STD, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    p = np.mean([_windows(xs, lambda b, s=s: _radar_session(f"{fold}_s{s}").run(None, {"x": b})[0])
                 for s in range(RADAR_SEEDS)], axis=0)
    return np.where(present, p, np.nan)


def radar_water_fraction(before: np.ndarray, after: np.ndarray, fold: str = "ALL"):
    """Water fractions of the two periods from radar stacks (n, 2, H, W): per scene a probability, per
    period the per-pixel median over the scenes, water where > 0.5. Returns (fb, fa, usable fraction)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pb = np.nanmedian(np.stack([radar_probability(s, fold) for s in before]), axis=0)
        pa = np.nanmedian(np.stack([radar_probability(s, fold) for s in after]), axis=0)
    usable = np.isfinite(pb) & np.isfinite(pa)
    if not usable.any():
        return None, None, 0.0
    return float((pb[usable] > 0.5).mean()), float((pa[usable] > 0.5).mean()), float(usable.mean())


def water_mask(c: np.ndarray, detector: str = "ndwi") -> np.ndarray:
    """detector: 'ndwi', 'unet' (trained on every event), or 'unet-without-<event>' (for tests on that event)."""
    if detector == "ndwi":
        return ndwi_water(c)
    if detector == "unet":
        return unet_probability(c, "ALL") > 0.5
    if detector.startswith("unet-without-"):
        return unet_probability(c, detector.removeprefix("unet-without-")) > 0.5
    raise ValueError(f"unknown water detector {detector!r}")
