"""MMFlood re-read as the pipeline reads radar (milestone B9, avenue 3).

MMFlood (Montello et al. 2022; Zenodo 6534637, CC BY 4.0): 1 748 tiles of 95 flood events, 2014-2021, with a
mask of the flooded area drawn from the Copernicus emergency service's delineation maps, on a grid of about
10 m in latitude and longitude. Its own radar is Sentinel-1 GRD from Sentinel Hub; the pipeline reads
Planetary Computer's Sentinel-1 RTC (gamma naught, terrain-corrected). As for Sen1Floods11 in B2, every tile
is re-read from Planetary Computer onto the mask's grid:

- the radar scene: the RTC scene acquired at MMFlood's own acquisition time of the tile (the tag
  `acquisition_date` of its s1_raw file, read straight from the archive), within 1 hour, covering the tile;
  VV and VH, bilinear in power, converted to dB;
- the label, "water" as in Sen1Floods11's hand labels (flood and permanent water): MMFlood's flood mask, or
  permanent water from the JRC Global Surface Water occurrence layer (water in at least PERMANENT % of the
  months 1984-2021; Planetary Computer collection jrc-gsw), read with nearest neighbour;
- no label (-1) where the radar has no value.

    python -m core.data.mmflood_s1rtc
Writes data_cache/mmflood/s1rtc/<tile>.npz (x: (2, H, W) VV, VH dB; label: (H, W) int8; scene; date) and
data_cache/mmflood/s1rtc_report.json.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data_cache" / "mmflood"
ZIP = DIR / "download" / "mmflood.zip"
MASKS = DIR / "extracted" / "mmflood"
OUT = DIR / "s1rtc"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PERMANENT = 50


def tiles() -> list[str]:
    return sorted(f.stem for f in MASKS.glob("*/mask/*.tif"))


def acquisition(tile: str) -> str:
    """The acquisition time of MMFlood's own radar image of the tile, from the archive (no full extraction)."""
    import rasterio
    area = tile.rsplit("-", 1)[0]
    raw = subprocess.run(["unzip", "-p", str(ZIP), f"mmflood/{area}/s1_raw/{tile}.tif"], capture_output=True,
                         check=True).stdout
    with rasterio.io.MemoryFile(raw) as m, m.open() as r:
        return r.tags()["acquisition_date"]


def to_db(power: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.where(power > 0, 10 * np.log10(power), np.nan).astype(np.float32)


def fetch(tile: str) -> str:
    import odc.stac
    import planetary_computer
    import rasterio
    from odc.geo.geobox import GeoBox
    from pystac_client import Client
    dest = OUT / f"{tile}.npz"
    if dest.exists():
        return "cached"
    area = tile.rsplit("-", 1)[0]
    with rasterio.open(MASKS / area / "mask" / f"{tile}.tif") as src:
        gbox = GeoBox((src.height, src.width), src.transform, src.crs)
        bounds = list(src.bounds)
        mask = src.read(1)
    t = datetime.fromisoformat(acquisition(tile))
    cat = Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bounds,
                            datetime=f"{(t - timedelta(hours=1)).isoformat()}Z/{(t + timedelta(hours=1)).isoformat()}Z").items())
    if not items:
        return f"no RTC scene within 1 hour of {t.isoformat()}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = odc.stac.load(items, bands=["vv", "vh"], geobox=gbox, groupby="solar_day", resampling="bilinear",
                            nodata=-32768, dtype="float32", fail_on_error=True).isel(time=0)
        gsw = list(cat.search(collections=["jrc-gsw"], bbox=bounds).items())
        occ = odc.stac.load(gsw, bands=["occurrence"], geobox=gbox, resampling="nearest",
                            fail_on_error=True).isel(time=0)["occurrence"].values if gsw else np.zeros(mask.shape)
    x = np.stack([to_db(np.where(raw[b].values == -32768, np.nan, raw[b].values)) for b in ("vv", "vh")])
    if not np.isfinite(x).any():
        return "scene found but no data on this tile"
    permanent = (occ >= PERMANENT) & (occ <= 100)                   # 255 = no data in the JRC layer
    label = np.where(np.isfinite(x).all(0), ((mask == 1) | permanent).astype(np.int8), -1).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=x, label=label, scene=np.array([it.id for it in items]), date=t.isoformat(),
                        flood_fraction=float((mask == 1).mean()), permanent_fraction=float(permanent.mean()))
    return "ok"


def _safe(tile: str) -> tuple[str, str]:
    try:
        return tile, fetch(tile)
    except Exception as e:      # recorded; the tile is left out
        return tile, f"error {type(e).__name__}: {str(e)[:160]}"


def main(workers: int = 8) -> None:
    report = {}
    with ThreadPoolExecutor(workers) as pool:
        for k, (t, r) in enumerate(pool.map(_safe, tiles()), 1):
            report[t] = r
            if k % 50 == 0:
                print(f"  {k} tiles, ok {sum(v in ('ok', 'cached') for v in report.values())}", flush=True)
    (DIR / "s1rtc_report.json").write_text(json.dumps(report, indent=1))
    print(f"{len(report)} tiles, ok {sum(v in ('ok', 'cached') for v in report.values())}")


if __name__ == "__main__":
    if sys.argv[1:] == ["one"]:
        print(fetch(tiles()[0]))
    else:
        main()
