"""Sen1Floods11 (Bonafilia et al. 2020): 446 hand-labelled 512 x 512 tiles from 11 flood events.

Files per tile (all from the free public bucket gs://sen1floods11, read over plain HTTPS):
  S2Hand    Sentinel-2 Level-1C, 13 bands B01..B12 (with B8A), top-of-atmosphere reflectance x 10000
  S1Hand    Sentinel-1 radar, 2 bands VV, VH (dB)            (for milestone B2)
  LabelHand hand label: 1 water, 0 not water, -1 not labelled (cloud or unsure)
Official division into parts (splits): train 252, valid 89, test 90, and Bolivia (15) held out.

    python -m core.data.sen1floods11          # downloads what is missing (about 4 GB)
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data_cache" / "sen1floods11"
URL = "https://storage.googleapis.com/sen1floods11/v1.1"
SPLIT_URL = f"{URL}/splits/flood_handlabeled/flood_{{part}}_data.csv"
KINDS = ("S2Hand", "S1Hand", "LabelHand")

# Band positions in S2Hand (B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12)
S2_INDEX = {"coastal": 0, "blue": 1, "green": 2, "red": 3, "re1": 4, "re2": 5, "re3": 6, "nir": 7,
            "nir08": 8, "wv": 9, "cirrus": 10, "swir16": 11, "swir22": 12}


def chips() -> list[str]:
    return sorted(p.name.removesuffix("_LabelHand.tif") for p in (DIR / "LabelHand").glob("*_LabelHand.tif"))


def event(chip: str) -> str:
    return chip.rsplit("_", 1)[0]


def path(chip: str, kind: str) -> Path:
    return DIR / kind / f"{chip}_{kind}.tif"


def splits() -> dict[str, str]:
    """chip -> 'train' | 'valid' | 'test' | 'bolivia' (the official parts)."""
    out = {}
    for part in ("train", "valid", "test", "bolivia"):
        f = DIR / "splits" / f"flood_{part}_data.csv"
        for row in csv.reader(io.StringIO(f.read_text())):
            if row:
                out[row[0].strip().removesuffix("_S1Hand.tif")] = part
    return out


def read(chip: str, kind: str) -> np.ndarray:
    import rasterio
    with rasterio.open(path(chip, kind)) as src:
        return src.read()


def _get(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.rename(dest)


def download() -> None:
    for part in ("train", "valid", "test", "bolivia"):
        _get(SPLIT_URL.format(part=part), DIR / "splits" / f"flood_{part}_data.csv")
    jobs = [(f"{URL}/data/flood_events/HandLabeled/{k}/{c}_{k}.tif", path(c, k)) for c in chips() for k in KINDS]
    with ThreadPoolExecutor(16) as pool:
        for i, _ in enumerate(pool.map(lambda j: _get(*j), jobs), 1):
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} files", flush=True)
    print(f"  done: {len(chips())} tiles, parts: { {p: list(splits().values()).count(p) for p in set(splits().values())} }")


if __name__ == "__main__":
    download()
    sys.exit(0)
