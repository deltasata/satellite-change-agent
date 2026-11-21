"""TerraMind-base-Fire's inputs for any burn patch, built by us in ImpactMesh's file format (milestone B8,
protocol docs/METHODS.md section 11, detector 4).

ImpactMesh gives each patch four time steps (pre-month, pre-event, event, post-event) of Sentinel-2 Level-2A
(12 bands) and Sentinel-1 radar, plus elevation, all from Microsoft Planetary Computer. Its rule for the dates
is not published; measured on its test files, the steps lie about -51, -24, +1 and
+22 days from the patch's date, the optical images nearly always clear. Our rule, fixed before any score:

  window (days from the patch date d)    optical image                               radar scene
  pre-month    d-75 .. d-36              most clear pixels (ESA's SCL), ties: middle  closest to d-51
  pre-event    d-35 .. d-6               most clear pixels, ties: closest to middle   closest to d-24
  event        d-2  .. d+5               closest to d (if none: closest in d-10..d+10) closest to d
  post-event   d+8  .. d+40              most clear pixels, ties: closest to middle   closest to d+22

Optical values: 12 bands at 10 m (nearest neighbour for the 20 m and 60 m bands, as ImpactMesh: checked on
EMSR382, correlation 0.997-0.999 against 0.94-0.98 with bilinear), DN with ESA's offset of 1000 removed
for processing baseline 04.00 and later (ImpactMesh's files have no offset), clouds not removed (ImpactMesh
keeps them too). Radar: VV, VH in dB. Elevation: Copernicus 30 m.

    python -m core.data.tmfire_inputs build           # ImpactMesh's held-out patches -> DIR/tm4/<name>.npz
    BURN_SOURCE=cems2026 python -m core.data.tmfire_inputs build
    .venv-fire/bin/python -m core.data.tmfire_inputs write   # npz -> DIR/tm4_fire/{S2L2A,S1RTC,DEM} (zarr, tif)
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

OUT = E.DIR / "tm4"
ROOTDIR = E.DIR / "tm4_fire"      # the name must contain "fire": the impactmesh package reads the disaster type from it
BANDS12 = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
STEPS = {"pre-month": (-75, -36, -51), "pre-event": (-35, -6, -24), "event": (-2, 5, 0), "post-event": (8, 40, 22)}
CLEAR_SCL = (2, 4, 5, 6, 7, 11)      # dark area, vegetation, bare, water, unclassified, snow: not cloud, shadow, no data


def wanted(p: dict) -> bool:
    """The patches that get TerraMind inputs: ImpactMesh's held-out fires, or every new patch."""
    return p["part"] in ("test_holdout", "cems2026")


def pick_optical(days: list[date], clear: np.ndarray, d: date) -> list[int | None]:
    """Index of the chosen optical day for each step (None if the window is empty)."""
    out = []
    for step, (a, b, mid) in STEPS.items():
        idx = [k for k, t in enumerate(days) if d + timedelta(days=a) <= t <= d + timedelta(days=b)]
        if step == "event":
            if not idx:
                idx = [k for k, t in enumerate(days) if abs((t - d).days) <= 10]
            out.append(min(idx, key=lambda k: (abs((days[k] - d).days), -days[k].toordinal())) if idx else None)
        else:
            out.append(min(idx, key=lambda k: (-round(float(clear[k]), 3), abs((days[k] - d).days - mid))) if idx else None)
    return out


def pick_radar(days: list[date], d: date) -> list[int | None]:
    out = []
    for a, b, mid in STEPS.values():
        idx = [k for k, t in enumerate(days) if d + timedelta(days=a) <= t <= d + timedelta(days=b)]
        out.append(min(idx, key=lambda k: abs((days[k] - d).days - mid)) if idx else None)
    return out


def _s2_items(gbox, d0: date, d1: date):
    from core.data.burn_radar import _catalog
    bbox = list(gbox.geographic_extent.boundingbox)
    return list(_catalog().search(collections=["sentinel-2-l2a"], bbox=bbox,
                                  datetime=f"{d0.isoformat()}/{(d1 + timedelta(days=1)).isoformat()}").items())


def _odc(items, bands, gbox, **kw):
    import odc.stac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return odc.stac.load(items, bands=list(bands), geobox=gbox, groupby="solar_day", fail_on_error=True,
                             chunks={}, **kw).compute()


def build_block(key: str) -> str:
    from core.data.burn_radar import _load
    ps = [p for p in E._blocks_cached()[key] if wanted(p) and not (OUT / f"{p['name']}.npz").exists()]
    if not ps:
        return "cached"
    gbox, (gx0, gy1) = E._grid(ps)
    d = date.fromisoformat(ps[0]["date"])
    items = _s2_items(gbox, d - timedelta(days=75), d + timedelta(days=40))
    if not items:
        return "no optical items"
    try:
        scl = _odc(items, ["SCL"], gbox, resampling="nearest")
    except AssertionError:          # a file without a coordinate system (EMSR873: 5 April 2026, tile 32TPP): skipped
        ok = []
        for it in items:
            try:
                _odc([it], ["SCL"], gbox, resampling="nearest")
                ok.append(it)
            except AssertionError:
                with (E.DIR / "tm4_skipped.jsonl").open("a") as fh:
                    fh.write(json.dumps({"block": key, "item": it.id, "reason": "file has no coordinate system"}) + "\n")
        items = ok
        scl = _odc(items, ["SCL"], gbox, resampling="nearest")
    days = [date.fromisoformat(str(t)[:10]) for t in scl.time.values]
    s = scl["SCL"].values
    by_day = {}
    for it in items:
        by_day.setdefault(it.datetime.date(), []).append(it)
    radar = _load("sentinel-1-rtc", ["vv", "vh"], gbox, d - timedelta(days=75), d + timedelta(days=40),
                  groupby="solar_day", nodata=-32768, dtype="float32")
    rdays = [date.fromisoformat(str(t)[:10]) for t in radar.time.values] if radar is not None else []
    dem = _load("cop-dem-glo-30", ["data"], gbox, dtype="float32")
    h = dem["data"].max("time").values.astype(np.float32) if dem is not None else np.full(gbox.shape, np.nan, np.float32)
    picks, cache = {}, {}
    for p in ps:
        i, j = round((gy1 - p["y0"]) / E.PIX), round((p["x0"] - gx0) / E.PIX)
        sp = s[:, i:i + E.PATCH, j:j + E.PATCH]
        clear = np.isin(sp, CLEAR_SCL).mean(axis=(1, 2))
        picks[p["name"]] = (pick_optical(days, clear, d), pick_radar(rdays, d), i, j)
    need = sorted({k for o, _, _, _ in picks.values() for k in o if k is not None})
    for k in need:      # the 12 bands, only for the chosen days
        its = by_day.get(days[k]) or [it for it in items if abs((it.datetime.date() - days[k]).days) <= 0]
        x = _odc(its, BANDS12, gbox, resampling="nearest", dtype="float32", nodata=0)
        dn = np.stack([x[b].values[0] for b in BANDS12]).astype(np.float32)
        base = max(float(it.properties.get("s2:processing_baseline", "0") or 0) for it in its)
        if base >= 4.0:
            dn = np.where(dn > 0, np.maximum(dn - 1000, 1), 0)
        cache[k] = dn
    for p in ps:
        o, r, i, j = picks[p["name"]]
        s2 = np.zeros((4, 12, E.PATCH, E.PATCH), np.int16)
        for t, k in enumerate(o):
            if k is not None:
                s2[t] = cache[k][:, i:i + E.PATCH, j:j + E.PATCH].round().astype(np.int16)
        s1 = np.full((4, 2, E.PATCH, E.PATCH), np.nan, np.float16)
        for t, k in enumerate(r):
            if k is not None:
                raw = np.stack([radar[b].values[k][i:i + E.PATCH, j:j + E.PATCH] for b in ("vv", "vh")])
                with np.errstate(all="ignore"):
                    s1[t] = np.where((raw > 0) & (raw != -32768), 10 * np.log10(raw), np.nan)
        np.savez_compressed(OUT / f"{p['name']}.npz", s2=s2, s1=s1, dem=h[i:i + E.PATCH, j:j + E.PATCH],
                            s2_days=np.array([str(days[k]) if k is not None else "" for k in o]),
                            s1_days=np.array([str(rdays[k]) if k is not None else "" for k in r]),
                            epsg=p["epsg"], x0=p["x0"], y0=p["y0"])
    return f"{len(ps)} patches, {len(days)} optical days, {len(need)} read in 12 bands, {len(rdays)} radar days"


def build(workers: int = 16) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bl = {k: v for k, v in E.blocks().items() if any(wanted(p) for p in v)}
    log = E.DIR / "tm4_log.jsonl"
    with ProcessPoolExecutor(workers) as pool:
        futs = {k: pool.submit(build_block, k) for k in sorted(bl, key=lambda k: -len(bl[k]))}
        for n, (k, f) in enumerate(futs.items(), 1):
            try:
                msg = f.result()
            except Exception as e:      # recorded; the block is retried on the next run
                msg = f"ERROR {type(e).__name__}: {str(e)[:300]}"
            with log.open("a") as fh:
                fh.write(json.dumps({"block": k, "msg": msg}) + "\n")
            if n % 20 == 0:
                print(f"{n}/{len(futs)} blocks", flush=True)


def write() -> None:
    """In .venv-fire: the npz files as ImpactMesh's zarr and tif files, and a split file of their names."""
    import rasterio
    import zarr
    from rasterio.transform import from_origin
    for m in ("S2L2A", "S1RTC", "DEM"):
        (ROOTDIR / m).mkdir(parents=True, exist_ok=True)
    names = []
    for f in sorted(OUT.glob("*.npz")):
        n = f.stem
        names.append(n)
        if (ROOTDIR / "DEM" / f"{n}_DEM.tif").exists():
            continue
        z = np.load(f)
        for m, arr, bands in (("S2L2A", z["s2"], list(BANDS12)), ("S1RTC", z["s1"].astype(np.float32), ["vv", "vh"])):
            store = zarr.storage.ZipStore(str(ROOTDIR / m / f"{n}_{m}.zarr.zip"), mode="w")
            g = zarr.group(store=store)
            g.create_dataset("bands", data=arr, chunks=arr.shape)
            g.create_dataset("band", data=np.array(bands))
            zarr.consolidate_metadata(store)
            store.close()
        with rasterio.open(ROOTDIR / "DEM" / f"{n}_DEM.tif", "w", driver="GTiff", height=E.PATCH, width=E.PATCH,
                           count=1, dtype="float32", crs=f"EPSG:{int(z['epsg'])}",
                           transform=from_origin(float(z["x0"]), float(z["y0"]), E.PIX, E.PIX)) as r:
            r.write(z["dem"].astype(np.float32), 1)
    (ROOTDIR / "split.txt").write_text("\n".join(names) + "\n")
    print(len(names), "patches written")


if __name__ == "__main__":
    {"build": build, "write": write}[sys.argv[1]]()
