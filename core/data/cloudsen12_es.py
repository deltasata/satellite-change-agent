"""An Earth Search copy of the CloudSEN12+ test images (milestone B3, docs/METHODS.md section 6).

Why: the B3 decision is made on the kind of image the pipeline reads (as with the Level-2A copy of B1).
CloudSEN12's own Level-2A images were made by Google Earth Engine; our pipeline reads Earth Search's
`sentinel-2-l2a`. For each test image: the Earth Search product of the same Sentinel-2 tile and day (one
per tile and day: the highest processing baseline, as in stac_search.deduplicate), the pipeline's 7 bands
and the scene classification SCL, read onto the label's 509 x 509 pixel grid (bilinear for reflectance,
nearest for SCL, as in core/data/timeseries.py).

    python -m core.data.cloudsen12_es          # needs data_cache/cloudsen12/meta.csv (core.data.cloudsen12)
Writes data_cache/cloudsen12/test_es/<id>.npz:
    x       (7, 509, 509) float32 reflectance, NaN where no data
    scl     (509, 509) uint8
    scenes  the Earth Search product used
and data_cache/cloudsen12/test_es_report.json (what happened to each image).
"""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from core.data.cloudsen12 import BAND_NAMES, DIR, N
from core.data.stac_search import EARTH_SEARCH_URL, is_free_https

OUT = DIR / "test_es"


def test_rows() -> list[dict]:
    with open(DIR / "meta.csv") as f:
        return [r for r in csv.DictReader(f) if r["split"] == "test"]


def fetch(row: dict) -> str:
    import odc.stac
    from affine import Affine
    from odc.geo.geobox import GeoBox
    from pystac_client import Client

    dest = OUT / f"{row['id']}.npz"
    if dest.exists():
        return "cached"
    g = [float(v) for v in row["geotransform"].split()]
    gbox = GeoBox((N, N), Affine(g[1], g[2], g[0], g[4], g[5], g[3]), row["crs"])
    tile = re.search(r"_T(\d\d[A-Z]{3})_", row["s2_id"]).group(1)
    d = date(*map(int, re.search(r"_(\d{4})(\d\d)(\d\d)T", row["s2_id"]).groups()))
    lon0, lat0, lon1, lat1 = gbox.geographic_extent.boundingbox
    items = list(Client.open(EARTH_SEARCH_URL).search(
        collections=["sentinel-2-l2a"], bbox=[lon0, lat0, lon1, lat1],
        datetime=f"{d.isoformat()}T00:00:00Z/{(d + timedelta(days=1)).isoformat()}T00:00:00Z").items())
    items = [it for it in items if it.properties.get("grid:code") == f"MGRS-{tile}"
             and all(is_free_https(it.assets[b].href) for b in (*BAND_NAMES, "scl"))]
    if not items:
        return "no Earth Search product of that tile and day"
    best = max(items, key=lambda it: float(it.properties["s2:processing_baseline"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = odc.stac.load([best], bands=[*BAND_NAMES, "scl"], geobox=gbox,
                            resampling={"scl": "nearest", "*": "bilinear"}, fail_on_error=True).isel(time=0)
    scale = best.assets["red"].extra_fields["raster:bands"][0].get("scale", 1e-4)
    # Earth Search v1 pixels never contain the -0.1 offset (measured; see stac_search.item_to_scene)
    x = np.stack([np.where(raw[b].values == 0, np.nan, raw[b].values * scale) for b in BAND_NAMES]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=x, scl=raw["scl"].values.astype(np.uint8), scenes=np.array([best.id]))
    return "ok"


def main() -> None:
    rows = test_rows()
    report = {}

    def job(r):
        try:
            report[r["id"]] = fetch(r)
        except Exception as e:  # recorded, not hidden
            report[r["id"]] = f"error: {type(e).__name__}: {str(e)[:150]}"

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(job, rows), 1):
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}", flush=True)
    (DIR / "test_es_report.json").write_text(json.dumps(report, indent=1))
    print(Counter(v if not v.startswith("error") else "error" for v in report.values()))


if __name__ == "__main__":
    main()
