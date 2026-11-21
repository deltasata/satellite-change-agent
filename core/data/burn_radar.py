"""Radar dates and elevation for the burn patches (milestone B8, protocol docs/METHODS.md section 11).

For every patch of the B5 copy (core/data/impactmesh_es.py: ImpactMesh's patches, or with BURN_SOURCE=cems2026
the new fires of 2026), on the same grid and in the same windows as its optical dates (normal or wide, as B5
chose for that patch):
  - Sentinel-1 radar, Microsoft Planetary Computer's "sentinel-1-rtc" (brightness corrected for the slope of
    the ground, as in B2), VV and VH in dB, one image per day; at most MAX_PER_WINDOW per window, those
    covering most of the patch, then those closest to the patch's date, kept in time order;
  - elevation, the Copernicus 30 m model ("cop-dem-glo-30"), bilinear onto the 10 m grid.

Per patch, in <DIR>/s1/:
  <name>.npy        (T, 2, 256, 256) float16 VV, VH in dB, NaN where no data
  <name>.json       {"dates", "days" (since the after window began), "period" (0 before, 1 after)}
  <name>_dem.npy    (256, 256) float16 metres

    python -m core.data.burn_radar                      # ImpactMesh's patches
    BURN_SOURCE=cems2026 python -m core.data.burn_radar # the new fires
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import impactmesh_es as E  # noqa: E402

S1 = E.DIR / "s1"
MPC = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAX_PER_WINDOW = 8
MIN_COVER = 0.01


def select(dates: list[str], cover: np.ndarray, window: tuple[date, date], d: date) -> list[int]:
    """Indices kept in one window: at least MIN_COVER of the patch covered, at most MAX_PER_WINDOW (most covered,
    to 0.001, then closest to the patch's date d), in time order."""
    t = [date.fromisoformat(x) for x in dates]
    idx = [k for k in range(len(dates)) if window[0] <= t[k] <= window[1] and cover[k] >= MIN_COVER]
    if len(idx) > MAX_PER_WINDOW:
        idx = sorted(sorted(idx, key=lambda k: (-round(float(cover[k]), 3), abs((t[k] - d).days)))[:MAX_PER_WINDOW])
    return idx


def _catalog():
    import planetary_computer
    from pystac_client import Client
    return Client.open(MPC, modifier=planetary_computer.sign_inplace)


def _load(collection: str, bands: list[str], gbox, d0: date | None = None, d1: date | None = None, **kw):
    import odc.stac
    bbox = list(gbox.geographic_extent.boundingbox)
    q = dict(collections=[collection], bbox=bbox)
    if d0:
        q["datetime"] = f"{d0.isoformat()}/{(d1 + timedelta(days=1)).isoformat()}"
    items = list(_catalog().search(**q).items())
    if collection == "sentinel-1-rtc":
        items = [it for it in items if all(b in it.assets for b in bands)]
    if not items:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return odc.stac.load(items, bands=bands, geobox=gbox, resampling="bilinear", fail_on_error=True, **kw)


def build_block(key: str) -> str:
    todo = [p for p in E._blocks_cached()[key]
            if not (S1 / f"{p['name']}.json").exists() and (E.OUT / f"{p['name']}.npz").exists()]
    stages = {}
    for p in todo:
        z = np.load(E.OUT / f"{p['name']}.npz")
        if "dropped" not in z.files:
            stages.setdefault(str(z["stage"]), []).append(p)
    if not stages:
        return "cached"
    report = []
    for stage, ps in stages.items():
        gbox, (gx0, gy1) = E._grid(ps)
        d = date.fromisoformat(ps[0]["date"])
        wb, wa = E._windows(d, stage)
        ds = _load("sentinel-1-rtc", ["vv", "vh"], gbox, wb[0], wa[1], groupby="solar_day", nodata=-32768,
                   dtype="float32")
        dem = _load("cop-dem-glo-30", ["data"], gbox, dtype="float32")
        if ds is not None:
            raw = np.stack([ds[b].values for b in ("vv", "vh")], axis=1)
            with np.errstate(all="ignore"):
                x = np.where((raw > 0) & (raw != -32768), 10 * np.log10(raw), np.nan).astype(np.float32)
            dates = [str(t)[:10] for t in ds.time.values]
        else:
            x, dates = np.zeros((0, 2, *gbox.shape), np.float32), []
        h = dem["data"].max("time").values.astype(np.float32) if dem is not None else np.full(gbox.shape, np.nan, np.float32)
        for p in ps:
            i, j = round((gy1 - p["y0"]) / E.PIX), round((p["x0"] - gx0) / E.PIX)
            xp = x[:, :, i:i + E.PATCH, j:j + E.PATCH]
            cover = np.isfinite(xp).all(1).mean(axis=(1, 2)) if len(dates) else np.zeros(0)
            kb, ka = select(dates, cover, wb, d), select(dates, cover, wa, d)
            keep = kb + ka
            np.save(S1 / f"{p['name']}.npy", xp[keep].astype(np.float16))
            np.save(S1 / f"{p['name']}_dem.npy", h[i:i + E.PATCH, j:j + E.PATCH].astype(np.float16))
            (S1 / f"{p['name']}.json").write_text(json.dumps(dict(
                dates=[dates[k] for k in keep], days=[(date.fromisoformat(dates[k]) - wa[0]).days for k in keep],
                period=[0] * len(kb) + [1] * len(ka))))
        report.append(f"{stage}: {len(dates)} radar days, {len(ps)} patches")
    return "; ".join(report)


def ours(name: str, days: list[str]) -> np.ndarray:
    """Our VV, VH in dB for one patch on the given days, (len(days), 2, 256, 256), NaN where no scene."""
    p = next(q for q in E.patches() if q["name"] == name)
    gbox, _ = E._grid([p])
    out = np.full((len(days), 2, E.PATCH, E.PATCH), np.nan, np.float32)
    for k, s in enumerate(days):
        d = date.fromisoformat(s)
        ds = _load("sentinel-1-rtc", ["vv", "vh"], gbox, d, d, groupby="solar_day", nodata=-32768, dtype="float32")
        if ds is not None:
            raw = np.stack([ds[b].values[0] for b in ("vv", "vh")])
            with np.errstate(all="ignore"):
                out[k] = np.where((raw > 0) & (raw != -32768), 10 * np.log10(raw), np.nan)
    return out


def main(workers: int = 24) -> None:
    S1.mkdir(parents=True, exist_ok=True)
    bl = E.blocks()
    keys = sorted(bl, key=lambda k: -len(bl[k]))
    log = E.DIR / "s1_log.jsonl"
    with ProcessPoolExecutor(workers) as pool:
        futs = {k: pool.submit(build_block, k) for k in keys}
        for n, (k, f) in enumerate(futs.items(), 1):
            try:
                msg = f.result()
            except Exception as e:      # recorded; the block is retried on the next run
                msg = f"ERROR {type(e).__name__}: {str(e)[:300]}"
            with log.open("a") as fh:
                fh.write(json.dumps({"block": k, "msg": msg}) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(keys)} blocks", flush=True)


if __name__ == "__main__":
    main()
