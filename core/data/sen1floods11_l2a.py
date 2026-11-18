"""A Level-2A copy of Sen1Floods11: the same tiles, the same day, the same pixel grid as the hand labels.

Why: Sen1Floods11's own Sentinel-2 images are Level-1C (light at the top of the atmosphere). Our
pipeline reads Level-2A (surface reflectance, the atmosphere's effect removed) from Earth Search.
A water detector that is good on L1C may be worse on L2A. With this copy, every detector (the rule and
each trained model) can be tested, and trained, on the kind of image the pipeline actually uses.

For each tile: find the L2A scenes of the tile's date that cover it, read 7 bands and the scene
classification (SCL) directly onto the label's grid (EPSG:4326, 512 x 512), and save
data_cache/sen1floods11/S2L2A/<tile>.npz with
    x    (7, 512, 512) float32 reflectance: blue, green, red, nir (B08), nir08 (B8A), swir16 (B11),
         swir22 (B12). B08 is for the pipeline's water rule; the other six are the models' input.
    scl  (512, 512) uint8, 0 where no image
A tile with no L2A scene that day (India 2016, before the L2A archive) gets no file; this is recorded.

    python -m core.data.sen1floods11_l2a        # about 15-30 minutes, 8 tiles at a time
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from core.data import sen1floods11 as s1f
from core.data.stac_search import EARTH_SEARCH_URL, is_free_https

BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
OUT = s1f.DIR / "S2L2A"


def _dates() -> dict[str, str]:
    return {r["chip"]: r["date"] for r in json.loads((s1f.DIR / "chip_stats.json").read_text())}


def fetch(chip: str, day: str) -> str:
    import odc.stac
    import rasterio
    from odc.geo.geobox import GeoBox
    from pystac_client import Client

    dest = OUT / f"{chip}.npz"
    if dest.exists():
        return "cached"
    with rasterio.open(s1f.path(chip, "LabelHand")) as src:
        gbox = GeoBox((src.height, src.width), src.transform, src.crs)
        bounds = list(src.bounds)
    d = date.fromisoformat(day)
    items = list(Client.open(EARTH_SEARCH_URL).search(
        collections=["sentinel-2-l2a"], bbox=bounds,
        datetime=f"{d.isoformat()}T00:00:00Z/{(d + timedelta(days=1)).isoformat()}T00:00:00Z").items())
    items = [it for it in items if all(is_free_https(it.assets[b].href) for b in (*BANDS7, "scl"))]
    if not items:
        return "no L2A scene on that day"
    # one processing per tile and day: the highest processing baseline, as in stac_search.deduplicate
    best = {}
    for it in items:
        key = it.properties["grid:code"]
        if key not in best or float(it.properties["s2:processing_baseline"]) > float(best[key].properties["s2:processing_baseline"]):
            best[key] = it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = odc.stac.load(list(best.values()), bands=[*BANDS7, "scl"], geobox=gbox, groupby="solar_day",
                            resampling={"scl": "nearest", "*": "bilinear"}, fail_on_error=True).isel(time=0)
    scale = best[next(iter(best))].assets["red"].extra_fields["raster:bands"][0].get("scale", 1e-4)
    # Earth Search v1 pixels never contain the -0.1 offset (measured; see stac_search.item_to_scene)
    x = np.stack([np.where(raw[b].values == 0, np.nan, raw[b].values * scale) for b in BANDS7]).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=x, scl=raw["scl"].values.astype(np.uint8),
                        scenes=np.array([it.id for it in best.values()]))
    return "ok"


def main() -> None:
    dates = _dates()
    chips = s1f.chips()
    report = {}

    def job(c):
        try:
            report[c] = fetch(c, dates[c])
        except Exception as e:  # recorded, not hidden
            report[c] = f"error: {type(e).__name__}: {str(e)[:150]}"
        return c

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(job, chips), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(chips)}", flush=True)
    (s1f.DIR / "S2L2A_report.json").write_text(json.dumps(report, indent=1))
    from collections import Counter
    print(Counter(v if not v.startswith("error") else "error" for v in report.values()))


if __name__ == "__main__":
    main()
