"""CloudSEN12+ (milestone B3): Sentinel-2 images with every pixel labelled by experts as clear, thick
cloud, thin cloud or cloud shadow. Public domain (CC0). Protocol: docs/METHODS.md section 6.

Source: Hugging Face `tacofoundation/cloudsen12`, read image by image over the web with `tacoreader<1.0`
(the reader for this older file format), so only the images we use are downloaded. Runs in `.venv-ml`.

We keep the high-quality 509 x 509 images (training 8 490, validation 535, test 975). For each:
    x      (7, 509, 509) uint16, the pipeline's 7 bands B02 B03 B04 B08 B8A B11 B12 of the Level-2A image
           made by Google Earth Engine (reflectance x 10 000)
    y      (509, 509) uint8, the label: 0 clear, 1 thick cloud, 2 thin cloud, 3 cloud shadow
and, for validation and test only, the masks stored in the data set's "extra" part:
    scl                 ESA's scene classification of the same Level-2A product (classes 0-11)
    unetmobv2_v2, sensei_v2, s2cloudless, cloudscore_cs_v1, cloudscore_cs_cdf_v1   published methods
The files are padded from 509 to 512 with zeros in the last 3 rows and columns, and the label there is 0
("clear"); we cut the padding away, so it can never be scored.

Pitfall: in the reader's table a row's name is not its position. Images are read by position.

    .venv-ml/bin/python -m core.data.cloudsen12            # all three parts, 16 reads at a time
    .venv-ml/bin/python -m core.data.cloudsen12 test       # one part
Writes data_cache/cloudsen12/<split>/<id>.npz and data_cache/cloudsen12/meta.csv.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

DIR = Path(__file__).resolve().parents[2] / "data_cache" / "cloudsen12"
SPLITS = ("train", "validation", "test")
# indices (1-based, as rasterio counts) of B02 B03 B04 B08 B8A B11 B12 in the Level-2A file (14 bands)
L2A_BANDS = [2, 3, 4, 8, 9, 11, 12]
BAND_NAMES = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")   # the pipeline's names
EXTRA = ("cloudmask_sen2cor", "cloudmask_unetmobv2_v2", "cloudmask_sensei_v2", "cloudmask_s2cloudless",
         "cloudmask_cloudscore_cs_v1", "cloudmask_cloudscore_cs_cdf_v1")
N = 509


TACO = DIR / "taco"   # the data set's files, downloaded whole (see below)


def catalogues():
    """The catalogues of the Level-2A and extra parts: from the local files when they are all present.

    Reading image by image over the web (16 at a time, no account) made Hugging Face refuse requests
    ("429 Too Many Requests") on 18 November. The 9 files are therefore downloaded whole (about 153 GB,
    9 requests) into data_cache/cloudsen12/taco/ and read from there."""
    import tacoreader
    out = []
    for mode, n in (("l2a", 6), ("extra", 3)):
        files = [TACO / f"cloudsen12-{mode}.{i:04d}.part.taco" for i in range(n)]
        if all(f.exists() for f in files):
            out.append(tacoreader.load([str(f) for f in files]))
        else:
            out.append(tacoreader.load(f"tacofoundation:cloudsen12-{mode}"))
    return tuple(out)


def high509(l2, split: str) -> np.ndarray:
    """Positions (not row names) of the high-quality 509-pixel images of one part."""
    return np.flatnonzero(((l2.label_type == "high") & (l2.real_proj_shape == 509)
                           & (l2["tortilla:data_split"] == split)).values)


def fetch(l2, ex, ex_pos: dict, p: int, split: str) -> str:
    import rasterio as rio
    row = l2.iloc[p]
    tid = row["tortilla:id"]
    dest = DIR / split / f"{tid}.npz"
    if dest.exists():
        return "exists"
    s = l2.read(int(p))
    with rio.open(s.read(0)) as src:
        x = src.read(L2A_BANDS)[:, :N, :N]
    with rio.open(s.read(1)) as src:
        y = src.read(1)[:N, :N]
    out = dict(x=x.astype(np.uint16), y=y.astype(np.uint8))
    if split != "train":
        e = ex.read(int(ex_pos[tid]))
        names = list(e["tortilla:id"])
        for name in EXTRA:
            with rio.open(e.read(names.index(name))) as src:
                out[name.removeprefix("cloudmask_").replace("sen2cor", "scl")] = src.read(1)[:N, :N]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.rename(dest)
    return "ok"


def write_meta(l2) -> None:
    import pandas as pd
    # new name -> catalogue column. Built column by column: selecting several columns at once from the
    # reader's table returned them in the table's own order, not in the order asked for.
    cols = {"id": "tortilla:id", "split": "tortilla:data_split", "roi_id": "roi_id", "s2_id": "s2_id",
            "crs": "stac:crs", "geotransform": "stac:geotransform", "time": "stac:time_start",
            "thick": "thick_percentage", "thin": "thin_percentage", "shadow": "cloud_shadow_percentage",
            "clear": "clear_percentage"}
    sub = l2.iloc[np.concatenate([high509(l2, s) for s in SPLITS])]
    rows = pd.DataFrame({new: sub[old].to_numpy() for new, old in cols.items()})
    rows["geotransform"] = rows["geotransform"].map(lambda g: " ".join(f"{v:.6f}" for v in g))
    DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(DIR / "meta.csv", index=False)


def same_day_as_ocm_training(cat) -> list[str]:
    """Test images (high, 509) taken on the same day in the same 100 km Sentinel-2 tile as an image of
    OmniCloudMask's training material: high-quality 509 training part, scribble (all parts), the 2000-pixel
    images (all parts). No test place is in that material (nearest image 15.4 km away), so these share the
    weather, not the ground. They are checked apart (protocol). `cat`: the full Level-2A catalogue."""
    import pandas as pd
    d = pd.DataFrame({"id": cat["tortilla:id"].to_numpy(), "split": cat["tortilla:data_split"].to_numpy(),
                      "label_type": cat["label_type"].to_numpy(), "shape": cat["real_proj_shape"].to_numpy(),
                      "s2_id": cat["s2_id"].to_numpy()})
    d["tile_day"] = d.s2_id.str.extract(r"_T(\d\d[A-Z]{3})_")[0] + d.s2_id.str.extract(r"_(\d{8})T")[0]
    high509 = (d.label_type == "high") & (d["shape"] == 509)
    ocm = d[(high509 & (d.split == "train")) | (d.label_type == "scribble") | (d["shape"] == 2000)]
    test = d[high509 & (d.split == "test")]
    return sorted(test[test.tile_day.isin(set(ocm.tile_day))].id)


def main(splits=SPLITS, workers: int = 16) -> None:
    l2, ex = catalogues()
    write_meta(l2)
    out = Path(__file__).resolve().parents[2] / "runs" / "b3" / "same_day_as_ocm_training.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(same_day_as_ocm_training(l2), indent=1))
    ex_pos = {t: i for i, t in enumerate(ex["tortilla:id"].values)}
    for split in splits:
        pos = high509(l2, split)
        t0, n_ok, errors = time.time(), 0, []
        with ThreadPoolExecutor(workers) as pool:
            futs = {pool.submit(fetch, l2, ex, ex_pos, int(p), split): p for p in pos}
            for k, f in enumerate(as_completed(futs), 1):
                try:
                    f.result()
                    n_ok += 1
                except Exception as err:   # noqa: BLE001  (recorded and retried on the next run)
                    errors.append(f"{l2.iloc[futs[f]]['tortilla:id']}: {type(err).__name__}: {str(err)[:120]}")
                if k % 500 == 0:
                    print(f"  {split}: {k}/{len(pos)} ({time.time() - t0:.0f} s)", flush=True)
        print(f"{split}: {n_ok} of {len(pos)} images, {len(errors)} errors, {time.time() - t0:.0f} s", flush=True)
        for e in errors[:10]:
            print("   ", e)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or SPLITS)
