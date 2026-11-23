"""A Planetary Computer copy of SpaceNet 7 for urban change (milestone B9, avenue 2).

Why: OSCD (24 cities) is the only urban-change label set the construction network has seen. SpaceNet 7
(Van Etten et al. 2021; CC BY-SA 4.0; free on the public AWS bucket spacenet-dataset, no account) has
hand-drawn building outlines for 60 areas of about 4 km x 4 km, every month for about two years
(2017-2020), with matched building numbers across months. "New building" between the first and the last
labelled month is a label of urban change.

Per area:
- the outlines of the first and last month ("labels_match", whose numbers follow each building through
  time) and the masks of unusable image parts of both months (clouds in the Planet mosaics, "UDM");
- a 10 m grid in the local UTM zone over the area;
- label: 1 where buildings that are new in the last month cover at least NEW_FRACTION of the 10 m pixel
  (outlines drawn on a 2 m grid, then averaged); 0 elsewhere; 255 (no label) where either month's image is
  unusable;
- every Sentinel-2 Level-2A date within +- 90 days of the middle of each of the two months, read from
  Planetary Computer onto the grid and cloud-masked exactly as the OSCD copy (core/data/oscd_pc.py).

    python -m core.data.spacenet7 labels      # outlines and masks (about 240 small files)
    python -m core.data.spacenet7             # the Sentinel-2 dates of every area
Writes data_cache/spacenet7/labels/<area>/*.geojson and data_cache/spacenet7/pc/<area>.npz with
x (T, 7, H, W) float16 reflectance, NaN where not clear; dates; label (H, W) uint8; d1, d2.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data_cache" / "spacenet7"
LAB = DIR / "labels"
OUT = DIR / "pc"
BUCKET = "https://spacenet-dataset.s3.amazonaws.com"
PREFIX = "spacenet/SN7_buildings/train/"
NEW_FRACTION = 0.25
SUPER = 5                     # 2 m sub-pixels per 10 m pixel side
HALF_WINDOW = 90


def _list(prefix: str, delimiter: str | None = None) -> list[str]:
    out, token = [], None
    while True:
        q = f"{BUCKET}/?list-type=2&prefix={prefix}" + (f"&delimiter={delimiter}" if delimiter else "") + \
            (f"&continuation-token={urllib.request.quote(token)}" if token else "")
        txt = urllib.request.urlopen(q, timeout=60).read().decode()
        out += re.findall(r"<Prefix>([^<]*)</Prefix>" if delimiter else r"<Key>([^<]*)</Key>", txt)
        m = re.search(r"<NextContinuationToken>([^<]*)</NextContinuationToken>", txt)
        if not m:
            return [p for p in out if p != prefix]
        token = m.group(1)


def areas() -> list[str]:
    f = DIR / "areas.json"
    if not f.exists():
        DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps([p.removeprefix(PREFIX).strip("/") for p in _list(PREFIX, "/")]))
    return json.loads(f.read_text())


def _month(key: str) -> str:
    return re.search(r"global_monthly_(\d{4}_\d{2})_mosaic", key).group(1)


def fetch_labels(area: str) -> dict:
    d = LAB / area
    meta = d / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())
    keys = [k for k in _list(f"{PREFIX}{area}/labels_match/") if k.endswith("_Buildings.geojson")]
    months = sorted({_month(k) for k in keys})
    m1, m2 = months[0], months[-1]
    d.mkdir(parents=True, exist_ok=True)
    for m in (m1, m2):
        stem = f"global_monthly_{m}_mosaic_{area}"
        for sub, name in (("labels_match", f"{stem}_Buildings.geojson"), ("labels", f"{stem}_UDM.geojson")):
            urllib.request.urlretrieve(f"{BUCKET}/{PREFIX}{area}/{sub}/{name}", d / f"{m}_{name.split('_')[-1]}")
    import rasterio
    with rasterio.open(f"/vsicurl/{BUCKET}/{PREFIX}{area}/images/global_monthly_{m1}_mosaic_{area}.tif") as r:
        bounds, crs = list(r.bounds), str(r.crs)
    out = {"area": area, "months": months, "m1": m1, "m2": m2, "bounds": bounds, "crs": crs}
    meta.write_text(json.dumps(out))
    return out


def _mid(m: str) -> date:
    y, mo = map(int, m.split("_"))
    return date(y, mo, 15)


def grid(meta: dict):
    """A 10 m grid in the area's UTM zone covering the image's extent."""
    from odc.geo.geobox import GeoBox
    from odc.geo.geom import BoundingBox
    from pyproj import Transformer
    w, s, e, n = meta["bounds"]
    lon, lat = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform((w + e) / 2, (s + n) / 2)
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    t = Transformer.from_crs(meta["crs"], f"EPSG:{epsg}", always_xy=True)
    xs, ys = t.transform([w, e, w, e], [s, s, n, n])
    bb = BoundingBox(min(xs), min(ys), max(xs), max(ys), crs=f"EPSG:{epsg}")
    return GeoBox.from_bbox(bb, resolution=10, tight=True)


def _features(path: Path) -> tuple[list[dict], str]:
    g = json.loads(path.read_text())
    name = (g.get("crs") or {}).get("properties", {}).get("name", "EPSG:4326")
    crs = "EPSG:4326" if "CRS84" in name else "EPSG:" + name.split("EPSG::")[-1] if "EPSG::" in name else name
    return g["features"], crs


def _geoms(feats: list[dict], src: str, dst: str) -> list[dict]:
    from rasterio.warp import transform_geom
    return [transform_geom(src, dst, f["geometry"]) for f in feats if f.get("geometry")]


def label(meta: dict, gbox) -> np.ndarray:
    from affine import Affine
    from rasterio.features import rasterize
    d = LAB / meta["area"]
    crs = str(gbox.crs)
    f1, _ = _features(d / f"{meta['m1']}_Buildings.geojson")
    f2, c2 = _features(d / f"{meta['m2']}_Buildings.geojson")
    old = {f["properties"]["Id"] for f in f1}
    new = [f for f in f2 if f["properties"]["Id"] not in old]
    H, W = gbox.shape
    fine = gbox.affine * Affine.scale(1 / SUPER)
    frac = np.zeros((H, W), np.float32)
    if new:
        m = rasterize([(g, 1) for g in _geoms(new, c2, crs)], out_shape=(H * SUPER, W * SUPER), transform=fine,
                      dtype=np.uint8)
        frac = m.reshape(H, SUPER, W, SUPER).mean(axis=(1, 3))
    lab = (frac >= NEW_FRACTION).astype(np.uint8)
    bad = []
    for m in (meta["m1"], meta["m2"]):
        f, c = _features(d / f"{m}_UDM.geojson")
        bad += _geoms(f, c, crs)
    if bad:
        lab[rasterize([(g, 1) for g in bad], out_shape=(H, W), transform=gbox.affine, dtype=np.uint8,
                      all_touched=True) == 1] = 255
    return lab


def build(area: str) -> str:
    from core.data.cloud_mask import valid_mask
    from core.data.oscd_pc import _items, _load
    from core.data.timeseries import FILE_FAULTS
    from core.models.cloud import classes
    from odc.geo.geom import BoundingBox  # noqa: F401
    dest = OUT / f"{area}.npz"
    if dest.exists():
        return "cached"
    meta = fetch_labels(area)
    gbox = grid(meta)
    lab = label(meta, gbox)
    d1, d2 = _mid(meta["m1"]), _mid(meta["m2"])
    bounds = list(gbox.geographic_extent.boundingbox)
    W = timedelta(days=HALF_WINDOW)
    items = _items(bounds, d1 - W, d1 + W) + _items(bounds, d2 - W, d2 + W)
    if not items:
        return "no Sentinel-2 items"
    dates, x, scl = _load(items, gbox)
    keep, xs = [], []
    for t in range(len(dates)):
        cc = classes(x[t][[2, 1, 4]])   # red, green, B8A
        ok = valid_mask(scl[t], FILE_FAULTS) & (cc == 0) & np.isfinite(x[t]).all(0)
        if ok.any():
            xt = x[t].copy()
            xt[:, ~ok] = np.nan
            xs.append(xt.astype(np.float16))
            keep.append(dates[t])
    if not xs:
        return "no clear date"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=np.stack(xs), dates=np.array(keep), label=lab, d1=str(d1), d2=str(d2),
                        items=np.array([it.id for it in items]), crs=str(gbox.crs),
                        transform=np.array(list(gbox.affine)[:6]))
    return f"ok: {len(items)} items, {len(dates)} days, {len(keep)} clear; new-building pixels {int((lab == 1).sum())}"


def _safe(fn, a):
    try:
        return a, fn(a)
    except Exception as e:      # recorded; the area is then left out
        return a, f"error {type(e).__name__}: {str(e)[:200]}"


def main(which: str = "build", workers: int = 4) -> None:
    fn = fetch_labels if which == "labels" else build
    res = {}
    with ProcessPoolExecutor(workers) as pool:
        for a, r in pool.map(_safe, [fn] * len(areas()), areas()):
            res[a] = r if isinstance(r, str) else {k: r[k] for k in ("m1", "m2")}
            print(a, res[a], flush=True)
    (DIR / f"{which}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build")
