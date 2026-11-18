"""Build a cloud-masked, irregular image time series (a data cube) for one area.

Output: an xarray Dataset with dimensions (time, y, x):
  - one float32 variable per band, in physical surface reflectance,
    NaN wherever the pixel is cloud / shadow / no-data;
  - "scl": the raw ESA class map (kept for evidence and debugging);
  - "valid_fraction": per time step, the fraction of the area that is usable.

Times are wherever the satellite happened to pass, not a regular grid, and
many time steps are mostly NaN. Nothing is interpolated or dropped here:
deciding what to do with gaps belongs to the change-detection step.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import xarray as xr

from core.data.cloud_mask import valid_mask
from core.data.stac_search import REQUIRED_ASSETS, fetch_items_by_id, is_free_https
from core.schemas import BBox, Scene

# blue, green, red, near-infrared (10 m) and short-wave infrared (20 m).
# Enough for the standard indices: NDVI (vegetation), NDWI (water), NDBI (built-up).
DEFAULT_BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir16")


def to_reflectance(dn: np.ndarray, scale: float, offset: float, nodata: int = 0) -> np.ndarray:
    """Integer digital numbers -> physical reflectance (float32); nodata -> NaN."""
    dn = np.asarray(dn)
    refl = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    return np.where(dn == nodata, np.float32(np.nan), refl)


def utm_epsg(bbox: BBox) -> int:
    """EPSG code of the UTM zone containing the bbox centre (326xx north, 327xx south).

    Fixing the CRS from the area itself (instead of from whichever tiles were
    found) guarantees that two cubes of the same area share one pixel grid.
    """
    lon = (bbox.west + bbox.east) / 2
    lat = (bbox.south + bbox.north) / 2
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_timeseries(
    scenes: list[Scene],
    bbox: BBox,
    bands: tuple[str, ...] = DEFAULT_BANDS,
    resolution: float = 10.0,
) -> xr.Dataset:
    """Download the pixels of `scenes` inside `bbox` and return the masked cube."""
    import odc.stac  # heavy import, only needed here

    if not scenes:
        raise ValueError("no scenes given")
    items = fetch_items_by_id([s.id for s in scenes], collection="sentinel-2-l2a")
    for it in items:  # zero-money guard: never touch a requester-pays (s3://) file
        bad = [k for k in [*bands, "scl"] if not is_free_https(it.assets[k].href)]
        if bad:
            raise ValueError(f"{it.id}: assets {bad} are not free HTTPS files; refusing to read them")

    # Calibrate per TILE, before tiles of one overpass are merged. Two tiles of the same day can
    # differ (one already offset-corrected by the provider, one not), found by benchmark v2.
    # So: load each calibration group separately, convert to reflectance, then fuse.
    by_id = {s.id: s for s in scenes}
    groups: dict[tuple[float, float], list] = defaultdict(list)
    for it in items:
        s = by_id[it.id]
        groups[(s.reflectance_scale, s.reflectance_offset)].append(it)

    parts = []
    for (scale, offset), its in groups.items():
        raw = odc.stac.load(
            its,
            bands=[*bands, "scl"],
            bbox=bbox.as_list(),
            crs=f"EPSG:{utm_epsg(bbox)}",
            resolution=resolution,
            groupby="solar_day",  # tiles from the same overpass become one time step
            resampling={"scl": "nearest", "*": "bilinear"},  # never average class labels
            chunks={},  # lazy: read only the pixels inside bbox, in parallel
            fail_on_error=True,  # a file that cannot be read must stop us, not become silent NaN
        ).compute()
        refl = {b: to_reflectance(raw[b].values, scale, offset) for b in bands}
        parts.append((raw, refl))

    raw = parts[0][0]
    if len(parts) == 1:
        refl = parts[0][1]
        scl = raw["scl"].values
        coords = raw.coords
    else:  # fuse: union of dates; per pixel the first group with data wins
        times = np.unique(np.concatenate([p[0].time.values for p in parts]))
        shape = (len(times), raw.sizes["y"], raw.sizes["x"])
        refl = {b: np.full(shape, np.nan, np.float32) for b in bands}
        scl = np.zeros(shape, raw["scl"].dtype)
        for r, rf in parts:
            k = np.searchsorted(times, r.time.values)
            s_new = r["scl"].values
            take = (scl[k] == 0) & (s_new != 0)  # SCL 0 = no data
            for b in bands:
                cur = refl[b][k]
                refl[b][k] = np.where(take, rf[b], cur)
            scl[k] = np.where(take, s_new, scl[k])
        coords = {"time": times, "y": raw.y, "x": raw.x}
        if "spatial_ref" in raw.coords:
            coords["spatial_ref"] = raw.coords["spatial_ref"]

    mask = valid_mask(scl)  # (time, y, x) bool
    out = xr.Dataset(coords=coords)
    for b in bands:
        cube = refl[b].astype(np.float32)
        cube[~mask] = np.nan
        out[b] = (("time", "y", "x"), cube)
        out[b].attrs["units"] = "surface reflectance (0-1)"
    out["scl"] = (("time", "y", "x"), scl)
    out["valid_fraction"] = ("time", mask.reshape(mask.shape[0], -1).mean(axis=1))
    out.attrs["scene_ids"] = [s.id for s in scenes]
    out.attrs["bbox"] = bbox.as_list()
    return out
