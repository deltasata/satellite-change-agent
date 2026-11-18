"""Sentinel-1 radar images of one area and period, from Microsoft Planetary Computer (free, no account).

Collection "sentinel-1-rtc": backscatter gamma naught with radiometric terrain correction (the
brightness is corrected for the slope of the ground), in power units. We convert it to dB.
Radar sees through cloud, so it can answer flood questions when clouds hide the ground (milestone B2).

    load_radar(bbox, start, end) -> RadarStack: x (n_days, 2, H, W) VV and VH in dB on a 10 m grid
    in the area's UTM zone, NaN where a scene has no data; one entry per day with a scene.
Results are kept in data_cache/radar/ so the same request is read only once.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
CACHE = Path(__file__).resolve().parents[2] / "data_cache" / "radar"


@dataclass
class RadarStack:
    x: np.ndarray | None          # (n, 2, H, W) dB, or None if no scene
    days: list[str]
    scenes: list[str]


def to_db(power: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(power > 0, 10 * np.log10(power), np.nan).astype(np.float32)


def load_radar(bbox: list[float], start: date, end: date) -> RadarStack:
    key = hashlib.sha1(json.dumps([[round(v, 6) for v in bbox], str(start), str(end)]).encode()).hexdigest()[:16]
    f = CACHE / f"{key}.npz"
    if f.exists():
        z = np.load(f)
        return RadarStack(z["x"] if "x" in z else None, list(z["days"]), list(z["scenes"]))
    import odc.stac
    import planetary_computer
    from pystac_client import Client

    cat = Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)
    end1 = (end + timedelta(days=1)).isoformat()
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start.isoformat()}/{end1}").items())
    items = [it for it in items if "vv" in it.assets and "vh" in it.assets]   # some scenes are HH/HV or VV only
    out = RadarStack(None, [], [it.id for it in items])
    if items:
        lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=10,
                               groupby="solar_day", resampling="bilinear", nodata=-32768, dtype="float32",
                               fail_on_error=True)
        x = np.stack([to_db(np.where(ds[b].values == -32768, np.nan, ds[b].values)) for b in ("vv", "vh")], axis=1)
        keep = [k for k in range(x.shape[0]) if np.isfinite(x[k]).all(axis=0).mean() > 0.01]
        days = [str(t)[:10] for t in ds.time.values]
        if keep:
            out = RadarStack(x[keep], [days[k] for k in keep], out.scenes)
    CACHE.mkdir(parents=True, exist_ok=True)
    extra = {"x": out.x} if out.x is not None else {}
    np.savez_compressed(f, days=np.array(out.days), scenes=np.array(out.scenes), **extra)
    return out
