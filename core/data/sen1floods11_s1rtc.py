"""A radar copy of Sen1Floods11 in the form the pipeline will read: Sentinel-1 RTC from Planetary Computer.

Why: Sen1Floods11's own radar images (S1Hand) were made in Google Earth Engine: backscatter "sigma
naught" in dB, corrected for the terrain's position but not for its slope. The free archive our
pipeline can read without an account is Microsoft Planetary Computer's "sentinel-1-rtc": backscatter
"gamma naught" with radiometric terrain correction (RTC), meaning the brightness is also corrected
for the slope of the ground, stored as power (not dB). On flat ground the two differ by a roughly
constant factor; on slopes they differ more. As with the Level-2A copy in B1, every radar detector is
tested and trained on the kind of image the pipeline will actually use.

For each tile: the RTC scene of the event's radar date and relative orbit (from Sen1Floods11's
metadata.geojson), VV and VH read onto the label's grid (EPSG:4326, 512 x 512, bilinear in power),
converted to dB, saved to data_cache/sen1floods11/S1RTC/<tile>.npz:
    x       (2, 512, 512) float32, VV and VH in dB, NaN where no image
    scenes  the scene names used
A tile with no scene is recorded in S1RTC_report.json and gets no file.

    python -m core.data.sen1floods11_s1rtc        # 8 tiles at a time
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np

from core.data import sen1floods11 as s1f

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
OUT = s1f.DIR / "S1RTC"
# metadata.geojson names the events differently from the tile names in two cases
META_NAME = {"Mekong": "Cambodia", "Sri-Lanka": "Sri-Lanka"}


def radar_dates() -> dict[str, tuple[str, int]]:
    """event -> (radar date, relative orbit), from Sen1Floods11's metadata.geojson."""
    g = json.loads((s1f.DIR / "metadata.geojson").read_text())
    by_loc = {f["properties"]["location"]: f["properties"] for f in g["features"]}
    out = {}
    for ev in {s1f.event(c) for c in s1f.chips()}:
        p = by_loc[META_NAME.get(ev, ev)]
        out[ev] = (p["s1_date"].replace("/", "-"), int(p["rel_orbit_num"]))
    return out


def to_db(power: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(power > 0, 10 * np.log10(power), np.nan).astype(np.float32)


def fetch(chip: str, day: str, orbit: int) -> str:
    import odc.stac
    import planetary_computer
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
    cat = Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bounds,
                            datetime=f"{d.isoformat()}T00:00:00Z/{(d + timedelta(days=1)).isoformat()}T00:00:00Z").items())
    items = [it for it in items if it.properties.get("sat:relative_orbit") == orbit]
    if not items:
        return "no RTC scene of that date and orbit"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day",
                            resampling="bilinear", nodata=-32768, dtype="float32",
                            fail_on_error=True).isel(time=0)
    x = np.stack([to_db(np.where(raw[b].values == -32768, np.nan, raw[b].values)) for b in ("vv", "vh")])
    if not np.isfinite(x).any():
        return "scene found but no data on this tile"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=x, scenes=np.array([it.id for it in items]))
    return "ok"


def main() -> None:
    dates = radar_dates()
    chips = s1f.chips()
    report = {}

    def job(c):
        day, orbit = dates[s1f.event(c)]
        for attempt in range(3):
            try:
                report[c] = fetch(c, day, orbit)
                break
            except Exception as e:  # recorded, not hidden
                report[c] = f"error: {type(e).__name__}: {str(e)[:150]}"
        return c

    with ThreadPoolExecutor(8) as pool:
        for i, _ in enumerate(pool.map(job, chips), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(chips)}", flush=True)
    (s1f.DIR / "S1RTC_report.json").write_text(json.dumps(report, indent=1))
    from collections import Counter
    print(Counter(v if not v.startswith("error") else "error" for v in report.values()))


if __name__ == "__main__":
    main()
