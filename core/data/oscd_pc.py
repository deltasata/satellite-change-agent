"""A Planetary Computer copy of OSCD (milestone B4): every image date around the two label dates, on the label's grid.

Why: OSCD (Daudt et al. 2018) has hand labels of urban change for 24 cities, made on Level-1C images
from 2015 to 2018. The pipeline compares median images of many cloud-masked Level-2A dates. To test the
pipeline's change rule and the networks on that kind of input, we read every Level-2A date within
± 90 days of each label date from Microsoft Planetary Computer (Earth Search lacks most 2015-2016
images) onto the label's pixel grid, and cloud-mask each date as the pipeline does (OmniCloudMask, and
SCL for no-data and saturated pixels). Protocol: docs/METHODS.md section 7.

The original files (from the authors' servers, see `download()`) carry their map position: the label is
drawn on the grid of the first image's B04 (EPSG:4326, about 10 m). Common copies (for example the
Hugging Face one) do not.

For each city: data_cache/oscd/pc/<city>.npz with
    x       (T, 7, H, W) float16 reflectance, NaN where no data or not clear: blue, green, red, nir (B08),
            nir08 (B8A), swir16 (B11), swir22 (B12)
    dates   (T,) "YYYY-MM-DD"
    label   (H, W) uint8, 1 = urban change
    d1, d2  the label dates
Dates with no clear pixel at all are not stored.

    python -m core.data.oscd_pc download     # the original files (0.5 GB)
    python -m core.data.oscd_pc              # all 24 cities, about 1 hour
    python -m core.data.oscd_pc check        # alignment check on three cities
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from core.data.oscd import data_dir

ORIG = data_dir() / "original"
IMAGES = ORIG / "Onera Satellite Change Detection dataset - Images"
LABELS = {"train": ORIG / "Onera Satellite Change Detection dataset - Train Labels",
          "test": ORIG / "Onera Satellite Change Detection dataset - Test Labels"}
OUT = data_dir() / "pc"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
URLS = {"images.zip": "https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download",
        "train_labels.zip": "https://partage.imt.fr/index.php/s/2D6n03k58ygBSpu/download",
        "test_labels.zip": "https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"}
MD5 = {"images.zip": "c50d4a2941da64e03a47ac4dec63d915", "train_labels.zip": "4d2965af8170c705ebad3d6ee71b6990",
       "test_labels.zip": "8177d437793c522653c442aa4e66c617"}

BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
PC_ASSET = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "nir08": "B8A", "swir16": "B11",
            "swir22": "B12", "scl": "SCL"}
HALF_WINDOW = 90   # days fetched around each label date; the protocol uses 60, widened once to 90
# Neighbouring cities count as one group (they share ground or dates). Protocol section 10.
GROUPS = {"saclay_e": "saclay", "saclay_w": "saclay", "paris": "paris_bercy", "bercy": "paris_bercy"}


# OSCD's official split, as listed in the data set's train.txt and test.txt (written here so that the
# list is known before the download; `check()` compares it with the files)
SPLIT = {"train": ["aguasclaras", "bercy", "bordeaux", "nantes", "paris", "rennes", "saclay_e", "abudhabi",
                   "cupertino", "pisa", "beihai", "hongkong", "beirut", "mumbai"],
         "test": ["brasilia", "montpellier", "norcia", "rio", "saclay_w", "valencia", "dubai", "lasvegas",
                  "milano", "chongqing"]}


def cities(split: str = "all") -> list[str]:
    return SPLIT["train"] + SPLIT["test"] if split == "all" else list(SPLIT[split])


def group(city: str) -> str:
    return GROUPS.get(city, city)


def label_dates(city: str) -> tuple[date, date]:
    d = [datetime.strptime(line.split()[-1], "%Y%m%d").date() for line in (IMAGES / city / "dates.txt").read_text().splitlines() if line.strip()]
    return d[0], d[1]


def grid(city: str):
    """The label's grid: the first image's B04 file (EPSG:4326)."""
    import rasterio
    from odc.geo.geobox import GeoBox
    f = sorted((IMAGES / city / "imgs_1").glob("*B04.tif"))[0]
    with rasterio.open(f) as src:
        return GeoBox((src.height, src.width), src.transform, src.crs), list(src.bounds), src.read(1)


def label(city: str) -> np.ndarray:
    from PIL import Image
    split = "train" if city in cities("train") else "test"
    m = np.array(Image.open(LABELS[split] / city / "cm" / "cm.png"))
    return (m[..., 0] if m.ndim == 3 else m) > 0


def _items(bounds, d0: date, d1: date):
    import planetary_computer
    from pystac_client import Client
    cat = Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=bounds, datetime=f"{d0}/{d1}").items())
    best = {}   # one processing per tile and day: the highest processing baseline
    for it in items:
        key = (it.properties.get("s2:mgrs_tile"), it.datetime.date())
        pb = float(it.properties.get("s2:processing_baseline", 0))
        if key not in best or pb > float(best[key].properties.get("s2:processing_baseline", 0)):
            best[key] = it
    return list(best.values())


def _offset(item) -> float:
    """Reflectance = DN / 10000 + offset. ESA added an offset of -1000 DN from processing baseline 04.00 on."""
    return -0.1 if float(item.properties.get("s2:processing_baseline", 0)) >= 4.0 else 0.0


def _load(items, gbox) -> tuple[list[str], np.ndarray, np.ndarray]:
    """All items onto the grid, one time step per day; calibrated per processing before days are fused."""
    import odc.stac
    groups: dict[float, list] = {}
    for it in items:
        groups.setdefault(_offset(it), []).append(it)
    days, refl, scl = {}, {}, {}
    for off, its in groups.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = odc.stac.load(its, bands=[PC_ASSET[b] for b in (*BANDS7, "scl")], geobox=gbox, groupby="solar_day",
                                resampling={"SCL": "nearest", "*": "bilinear"}, fail_on_error=True, chunks={}).compute()
        for k, t in enumerate(raw.time.values):
            d = str(t)[:10]
            dn = np.stack([raw[PC_ASSET[b]].values[k] for b in BANDS7])
            r = np.where(dn == 0, np.nan, dn * 1e-4 + off).astype(np.float32)
            s = raw["SCL"].values[k]
            if d in days:  # another processing of the same day: fill only where it has no data yet
                take = (scl[d] == 0) & (s != 0)
                refl[d] = np.where(take, r, refl[d])
                scl[d] = np.where(take, s, scl[d])
            else:
                days[d], refl[d], scl[d] = True, r, s
    order = sorted(days)
    return order, np.stack([refl[d] for d in order]), np.stack([scl[d] for d in order])


def build(city: str) -> str:
    from core.data.cloud_mask import valid_mask
    from core.data.timeseries import FILE_FAULTS
    from core.models.cloud import classes
    dest = OUT / f"{city}.npz"
    if dest.exists():
        return "cached"
    gbox, bounds, _ = grid(city)
    d1, d2 = label_dates(city)
    lab = label(city)
    if lab.shape != tuple(gbox.shape):
        return f"label shape {lab.shape} differs from grid {tuple(gbox.shape)}"
    W = timedelta(days=HALF_WINDOW)
    items = _items(bounds, d1 - W, d1 + W) + _items(bounds, d2 - W, d2 + W)
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
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, x=np.stack(xs), dates=np.array(keep), label=lab.astype(np.uint8),
                        d1=str(d1), d2=str(d2), items=np.array([it.id for it in items]))
    return f"ok: {len(items)} items, {len(dates)} days, {len(keep)} with clear pixels"


def load(city: str) -> dict:
    z = np.load(OUT / f"{city}.npz")
    return dict(x=z["x"], dates=[str(d) for d in z["dates"]], label=z["label"].astype(bool),
                d1=date.fromisoformat(str(z["d1"])), d2=date.fromisoformat(str(z["d2"])))


def check(names=("beirut", "montpellier", "chongqing")) -> dict:
    """Alignment: correlation of Planetary Computer's B04 on the first label date with OSCD's own B04
    (Level-1C) of that date, and after a one-pixel shift in each direction. Also: the city list above
    equals the data set's own train.txt and test.txt."""
    for s in SPLIT:
        if (IMAGES / f"{s}.txt").read_text().strip().split(",") != SPLIT[s]:
            raise ValueError(f"OSCD's {s}.txt differs from the list in this file")
    out = {}
    for c in names:
        gbox, bounds, b04 = grid(c)
        d1, _ = label_dates(c)
        its = _items(bounds, d1, d1)
        dates, x, _ = _load(its, gbox)
        red = x[0][2]
        a = b04.astype(np.float64)

        def corr(p, q):
            m = np.isfinite(p) & np.isfinite(q) & (q > 0)
            return float(np.corrcoef(p[m], q[m])[0, 1])
        out[c] = dict(day=dates[0], same=corr(red, a), shift_x=corr(red[:, 1:], a[:, :-1]), shift_y=corr(red[1:], a[:-1]))
        print(c, out[c], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check_alignment.json").write_text(json.dumps(out, indent=1))
    return out


def download() -> None:
    import hashlib
    import subprocess
    import zipfile
    ORIG.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        f = ORIG / name
        if not f.exists():
            subprocess.run(["curl", "-sL", "--max-time", "1200", "-o", str(f), url], check=True)
        if hashlib.md5(f.read_bytes()).hexdigest() != MD5[name]:
            raise RuntimeError(f"{name}: checksum differs")
        zipfile.ZipFile(f).extractall(ORIG)


def main() -> None:
    names = cities()
    report = {}
    with ProcessPoolExecutor(6) as pool:
        for c, r in zip(names, pool.map(_safe, names)):
            report[c] = r
            print(f"  {c}: {r}", flush=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=1))


def _safe(c: str) -> str:
    try:
        return build(c)
    except Exception as e:  # recorded, not hidden
        return f"error: {type(e).__name__}: {str(e)[:200]}"


if __name__ == "__main__":
    {"download": download, "check": check}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
