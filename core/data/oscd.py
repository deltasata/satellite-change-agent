"""Loader for the OSCD change-detection benchmark (Hugging Face copy blanchon/OSCD_MSI).

24 Sentinel-2 image pairs (14 train, 10 test), 2015-2018, Level-1C (top-of-
atmosphere) reflectance as uint16 DN (reflectance = DN / 10000), 13 bands all
resampled to 10 m, with a hand-labelled binary mask of urban change.

The parquet copy has no city names or coordinates; pairs are identified as
"train_00" ... "test_09" in file order.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OSCD_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12")
# OSCD band -> our common names (for indices and RGB rendering)
COMMON = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir16": "B11"}
HF_URL = "https://huggingface.co/datasets/blanchon/OSCD_MSI/resolve/main/data/{split}-00000-of-00001.parquet"


def data_dir() -> Path:
    root = Path(os.environ.get("EO_DATA_CACHE", Path(__file__).resolve().parents[2] / "data_cache"))
    return root / "oscd"


@dataclass
class OSCDPair:
    name: str
    before: np.ndarray  # (13, H, W) float32 reflectance
    after: np.ndarray  # (13, H, W)
    mask: np.ndarray  # (H, W) bool, True = changed

    def band(self, which: str, common: str) -> np.ndarray:
        img = self.before if which == "before" else self.after
        return img[OSCD_BANDS.index(COMMON[common])]


def _decode_mask(png_bytes: bytes) -> np.ndarray:
    from PIL import Image

    m = np.array(Image.open(io.BytesIO(png_bytes)))
    if m.ndim == 3:
        m = m[..., 0]
    return m > 0


def _convert(split: str) -> Path:
    """Parquet -> one compressed .npz per pair (done once)."""
    import pyarrow.parquet as pq

    src = data_dir() / f"{split}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing; download from {HF_URL.format(split=split)}")
    out = data_dir() / split
    out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    k = 0
    for batch in pf.iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        b = np.asarray(row["image1"], dtype=np.uint16)
        a = np.asarray(row["image2"], dtype=np.uint16)
        m = _decode_mask(row["mask"]["bytes"])
        np.savez_compressed(out / f"{split}_{k:02d}.npz", before=b, after=a, mask=m)
        k += 1
    return out


def load_split(split: str) -> list[OSCDPair]:
    """All pairs of 'train' or 'test', as float32 reflectance."""
    d = data_dir() / split
    if not d.exists() or not any(d.glob("*.npz")):
        _convert(split)
    pairs = []
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        pairs.append(OSCDPair(f.stem, z["before"].astype(np.float32) / 1e4,
                              z["after"].astype(np.float32) / 1e4, z["mask"].astype(bool)))
    return pairs
