"""Baseline change detection: difference imaging with no learned parameters
(except one threshold).

Two entry points:

1. Image pair (used for scoring on OSCD):
       score = change_score_pair(before, after)   # (bands, H, W) x2 -> (H, W)
       changed = score > threshold
   Method: per-image band standardisation (removes illumination / atmosphere
   differences to first order), then change vector analysis (CVA): the
   Euclidean length of the per-pixel difference vector across bands.

2. Gapped time series (used by the agent on our own cubes):
       result = detect_change_timeseries(ds, before=(d0, d1), after=(d2, d3))
   Method: per-pixel median composites of the clear observations in each
   window; noise = per-pixel robust scatter within the before window;
   score = |after - before| / noise per band, combined across bands.
   Abstains when a window has too few clear observations.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import xarray as xr
from scipy import ndimage

# ---------------------------------------------------------------- indices


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b), NaN where undefined. NDVI = nd(nir, red), etc."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out), out, np.nan)


def spectral_indices(red, green, nir, swir16) -> dict[str, np.ndarray]:
    """The three indices that distinguish our three change types."""
    return {
        "ndvi": normalized_difference(nir, red),  # vegetation: high
        "ndwi": normalized_difference(green, nir),  # open water: > 0
        "ndbi": normalized_difference(swir16, nir),  # built-up / bare: higher
    }


# ---------------------------------------------------------------- pair method


def standardize(img: np.ndarray) -> np.ndarray:
    """Per band: subtract the median, divide by a robust sigma (1.4826 * MAD).

    Robust statistics so that the changed pixels themselves (the signal) do
    not set the normalisation.
    """
    img = img.astype(np.float32)
    med = np.nanmedian(img, axis=(1, 2), keepdims=True)
    mad = np.nanmedian(np.abs(img - med), axis=(1, 2), keepdims=True)
    return (img - med) / (1.4826 * mad + 1e-6)


def change_score_pair(
    before: np.ndarray, after: np.ndarray, smooth: int = 3, normalize: bool = True
) -> np.ndarray:
    """CVA magnitude per pixel, after per-image standardisation (if `normalize`).

    `smooth` applies a median filter of that size to the score: isolated
    single-pixel responses are mostly noise and residual misregistration.
    """
    if before.shape != after.shape:
        raise ValueError(f"shape mismatch {before.shape} vs {after.shape}")
    d = standardize(after) - standardize(before) if normalize else after - before
    score = np.sqrt(np.nansum(d**2, axis=0) / d.shape[0])  # RMS over bands
    if smooth and smooth > 1:
        score = ndimage.median_filter(score, size=smooth)
    return score


def otsu_threshold(score: np.ndarray, nbins: int = 256) -> float:
    """Unsupervised threshold that best splits the score histogram into two classes."""
    from skimage.filters import threshold_otsu

    v = score[np.isfinite(score)]
    return float(threshold_otsu(v, nbins=nbins))


# ---------------------------------------------------------------- metrics


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Pixel metrics for the 'change' class. Also returns raw counts so
    metrics can be pooled over many images (sum counts, then divide)."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **metrics_from_counts(tp, fp, fn, tn)}


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "accuracy": accuracy}


# ---------------------------------------------------------------- time-series method

TS_BANDS = ("blue", "green", "red", "nir", "swir16")


def _window(ds: xr.Dataset, d0: date, d1: date) -> xr.Dataset:
    t = ds.time.values.astype("datetime64[D]")
    sel = (t >= np.datetime64(d0)) & (t <= np.datetime64(d1))
    return ds.isel(time=np.flatnonzero(sel))


def detect_change_timeseries(
    ds: xr.Dataset,
    before: tuple[date, date],
    after: tuple[date, date],
    bands: tuple[str, ...] = TS_BANDS,
    min_obs_before: int = 2,
    min_obs_after: int = 1,
    z_threshold: float = 4.0,
    noise_floor: float = 0.02,
    min_valid_pixels: float = 0.5,
    water: str = "ndwi",
) -> dict:
    """Compare two time windows of a masked cube.

    Per pixel and band:
        ref  = median of clear observations in the before window
        new  = median of clear observations in the after window
        sig  = max(robust date-to-date scatter in the before window, noise_floor)
        z    = (new - ref) / sig
    score = RMS of z over bands; changed = score > z_threshold.

    NOTE: z is an *effect size*, not a significance. sig is the scatter of single
    dates (natural variability + noise), while ref and new are medians of many
    dates, whose own uncertainty is ~sig/sqrt(N). So "z > 4" means "the change is
    four times larger than ordinary date-to-date variation", a practical-relevance
    criterion. Measured on a no-change area (E03 null): median z = 0.32,
    99.9th percentile = 1.5.

    Returns a dict with status "ok" or "insufficient_data" (the abstention
    path), maps, and summary numbers an LLM can reason about.
    """
    wb, wa = _window(ds, *before), _window(ds, *after)
    nb = np.isfinite(wb["red"]).sum("time").values if wb.sizes["time"] else np.zeros(ds["red"].shape[1:])
    na = np.isfinite(wa["red"]).sum("time").values if wa.sizes["time"] else np.zeros(ds["red"].shape[1:])
    usable = (nb >= min_obs_before) & (na >= min_obs_after)
    frac_usable = float(usable.mean())

    summary = {
        "n_dates_before": int(wb.sizes["time"]),
        "n_dates_after": int(wa.sizes["time"]),
        "median_clear_obs_before": float(np.median(nb)),
        "median_clear_obs_after": float(np.median(na)),
        "fraction_pixels_usable": frac_usable,
        "min_obs_required": {"before": min_obs_before, "after": min_obs_after},
    }
    if frac_usable < min_valid_pixels:
        return {"status": "insufficient_data", **summary,
                "reason": f"only {frac_usable:.0%} of pixels have >= {min_obs_before} clear "
                          f"observations before and >= {min_obs_after} after "
                          f"(need {min_valid_pixels:.0%})"}

    z2 = np.zeros(usable.shape, np.float32)
    comp = {}
    for b in bands:
        xb, xa = wb[b].values, wa[b].values
        with np.errstate(all="ignore"):
            ref = np.nanmedian(xb, axis=0)
            new = np.nanmedian(xa, axis=0)
            scatter = 1.4826 * np.nanmedian(np.abs(xb - ref), axis=0)
        sig = np.fmax(np.nan_to_num(scatter, nan=noise_floor), noise_floor)
        z2 += ((new - ref) / sig) ** 2
        comp[b] = (ref, new)
    score = np.sqrt(z2 / len(bands))
    score = np.where(usable, score, np.nan)
    score = ndimage.median_filter(np.nan_to_num(score, nan=0.0), size=3)
    score = np.where(usable, score, np.nan)
    changed = np.nan_to_num(score, nan=0.0) > z_threshold

    ib = spectral_indices(*(comp[k][0] for k in ("red", "green", "nir", "swir16")))
    ia = spectral_indices(*(comp[k][1] for k in ("red", "green", "nir", "swir16")))
    deltas = {}
    for k in ib:
        d = ia[k] - ib[k]
        deltas[f"delta_{k}_changed_median"] = float(np.nanmedian(d[changed])) if changed.any() else 0.0
        deltas[f"delta_{k}_all_median"] = float(np.nanmedian(d[usable]))
    if water == "ndwi":
        water_before = float(np.nanmean(ib["ndwi"][usable] > 0))
        water_after = float(np.nanmean(ia["ndwi"][usable] > 0))
    else:  # a trained water detector (milestone B1) on 7-band median composites
        from core.models.water import BANDS7, water_mask
        missing = [k for k in BANDS7 if k not in ds]
        if missing:
            raise ValueError(f"water detector {water!r} needs bands {missing}; stack the cube with bands=BANDS7")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # all-NaN pixels give NaN medians; they are outside `usable`
            cb = np.stack([np.nanmedian(wb[k].values, axis=0) for k in BANDS7])
            ca = np.stack([np.nanmedian(wa[k].values, axis=0) for k in BANDS7])
        water_before = float(water_mask(cb, water)[usable].mean())
        water_after = float(water_mask(ca, water)[usable].mean())

    return {
        "status": "ok",
        **summary,
        "z_threshold": z_threshold,
        "changed_fraction": float(changed[usable].mean()),
        "score_percentiles": {q: float(np.nanpercentile(score[usable], q)) for q in (50, 90, 99, 99.9)},
        "changed_area_m2": float(changed.sum() * abs(float(ds.x[1] - ds.x[0])) * abs(float(ds.y[1] - ds.y[0]))),
        "water_fraction_before": water_before,
        "water_fraction_after": water_after,
        **deltas,
        "_score": score,  # maps: leading underscore = not for the LLM / JSON
        "_changed": changed,
        "_composites": comp,
    }
