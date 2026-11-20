"""An Earth Search copy of ImpactMesh-Fire's labelled patches (milestone B5, protocol docs/METHODS.md section 8).

Why: ImpactMesh-Fire (IBM, DLR and ESA Phi-lab; Hugging Face `ibm-esa-geospatial/ImpactMesh-Fire`, CC BY
4.0) has burnt-area labels drawn from the Copernicus Emergency Management Service's maps, 256 x 256 pixels
at 10 m, for 229 fires. Each label is a GeoTIFF on a UTM grid whose tags give the fire's date. The pipeline
compares two median images of cloud-masked Earth Search Level-2A dates, so for every patch we read every
Earth Search date in the before window (fire date - 60 to - 3 days) and the after window (+ 7 to + 60) onto
the label's grid, mask clouds as the pipeline does (OmniCloudMask, and SCL for no-data and saturated
pixels), and keep the two median images and the pipeline's change-rule map. If fewer than half of a patch's
pixels have 2 clear observations before and 1 after, the windows are widened once (- 120 and + 90 days);
if still not, the patch is dropped (for every detector).

Patches are read in blocks (one fire, one tile, at most about 13 km across), so one read covers many
patches. The cloud masks are computed with the OmniCloudMask library on the GPUs (run in `.venv-ml`; the
pipeline's CPU version gives the same classes on 99.996 % of pixels, B3); the CPU version is about ten
times slower here. Only the patches B5 uses are built: the training and validation parts without the ten fires at
exam places (EXCLUDED), and the 42 held-out fires (`test_holdout`).

For each patch: data_cache/burn/impactmesh/es/<name>.npz with
    before, after   (7, 256, 256) float16 median reflectance, NaN where no clear observation:
                    blue, green, red, nir (B08), nir08 (B8A), swir16 (B11), swir22 (B12)
    nb, na          (256, 256) uint8 clear observations before / after
    usable          (256, 256) bool: nb >= 2 and na >= 1
    score           (256, 256) float16 the change rule's ratio map (NaN where not usable)
    label           (256, 256) int8: 1 burnt, 0 not burnt, -1 no label
    stage           "normal" or "wide" (which windows)
or, for a dropped patch, only `dropped` and `usable_fraction`.

    python -m core.data.impactmesh_es download   # the labels and split lists (50 MB)
    .venv-ml/bin/python -m core.data.impactmesh_es   # all blocks, cloud masks on the GPUs (GPUS below)
    python -m core.data.impactmesh_es check      # alignment against ImpactMesh's own images (needs test/S2L2A.tar)
"""

from __future__ import annotations

import json
import math
import os
import sys
import tarfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(os.environ.get("EO_DATA_CACHE", ROOT / "data_cache")) / "burn" / "impactmesh"
LABELS = DIR / "MASK"
SPLITS = DIR / "split"
OUT = DIR / "es"
HF = "https://huggingface.co/datasets/ibm-esa-geospatial/ImpactMesh-Fire/resolve/main"
BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
PATCH, PIX = 256, 10.0
BLOCK = 10240.0     # metres: patches whose corner falls in one 10.24 km cell form one block
WINDOWS = {"normal": ((-60, -3), (7, 60)), "wide": ((-120, -3), (7, 90))}
# Fires within 25 km of an exam fire, or the exam fire itself (protocol section 11): not used for training.
EXCLUDED = frozenset({"EMSR207", "EMSR300", "EMSR372", "EMSR448", "EMSR527", "EMSR589", "EMSR598", "EMSR715",
                      "EMSR733", "EMSR746"})


def download() -> None:
    import urllib.request
    SPLITS.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test", "test_holdout", "test_seen_events"):
        urllib.request.urlretrieve(f"{HF}/split/impactmesh_fire_{s}.txt", SPLITS / f"impactmesh_fire_{s}.txt")
    for s in ("train", "val", "test"):
        tmp = DIR / f"MASK_{s}.tar"
        urllib.request.urlretrieve(f"{HF}/{s}/MASK.tar", tmp)
        with tarfile.open(tmp) as t:
            t.extractall(DIR, filter="data")      # members are MASK/<name>_annotation_wildfire.tif
        tmp.unlink()


def split_of() -> dict[str, str]:
    out = {}
    for s in ("train", "val", "test_holdout", "test_seen_events"):
        for line in (SPLITS / f"impactmesh_fire_{s}.txt").read_text().split():
            out[line] = s
    return out


def patches() -> list[dict]:
    """Every labelled patch: name, fire (EMSR number), fire date, CRS, corner, part. Cached in patches.json."""
    f = DIR / "patches.json"
    if f.exists():
        return json.loads(f.read_text())
    import rasterio
    sp = split_of()
    out = []
    for p in sorted(LABELS.glob("*_annotation_wildfire.tif")):
        name = p.name.replace("_annotation_wildfire.tif", "")
        with rasterio.open(p) as r:
            t, tr = r.tags(), r.transform
            out.append(dict(name=name, fire=name.split("_")[0], aoi=t["event_id"], date=t["datetime"][:10],
                            epsg=r.crs.to_epsg(), x0=tr.c, y0=tr.f, part=sp.get(name, "?"),
                            lat=float(t["center_lat"]), lon=float(t["center_lon"])))
    f.write_text(json.dumps(out))
    return out


def used(p: dict) -> bool:
    """The patches B5 uses: training and validation without the excluded fires, and the held-out fires."""
    return p["part"] == "test_holdout" or (p["part"] in ("train", "val") and p["fire"] not in EXCLUDED)


def blocks() -> dict[str, list[dict]]:
    out = defaultdict(list)
    for p in patches():
        if used(p):
            out[f"{p['aoi']}_{math.floor(p['x0'] / BLOCK)}_{math.floor(p['y0'] / BLOCK)}"].append(p)
    return dict(out)


_BLOCKS: dict = {}


def _blocks_cached() -> dict[str, list[dict]]:
    if not _BLOCKS:
        _BLOCKS.update(blocks())
    return _BLOCKS


def label(name: str) -> np.ndarray:
    import rasterio
    with rasterio.open(LABELS / f"{name}_annotation_wildfire.tif") as r:
        return r.read(1).astype(np.int8)


def _items(lonlat, d0: date, d1: date) -> list:
    """Earth Search products over the area: one per tile and day (as the pipeline's deduplicate), free files only."""
    from pystac_client import Client

    from core.data.stac_search import EARTH_SEARCH_URL, deduplicate, is_free_https, item_to_scene
    items = list(Client.open(EARTH_SEARCH_URL).search(
        collections=["sentinel-2-l2a"], bbox=list(lonlat),
        datetime=f"{d0.isoformat()}T00:00:00Z/{d1.isoformat()}T23:59:59Z").items())
    items = [it for it in items if all(k in it.assets and is_free_https(it.assets[k].href) for k in (*BANDS7, "scl"))]
    keep = {s.id for s in deduplicate(item_to_scene(it.to_dict()) for it in items)}
    return [it for it in items if it.id in keep]


def _load(items, gbox):
    """All products onto the grid, one time step per day, reflectance = DN x scale (Earth Search's pixels never
    contain ESA's offset, see core/data/stac_search.py)."""
    import odc.stac
    groups = defaultdict(list)
    for it in items:
        groups[it.assets["red"].extra_fields["raster:bands"][0].get("scale", 1e-4)].append(it)
    days, refl, scl = {}, {}, {}
    for scale, its in groups.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = odc.stac.load(its, bands=[*BANDS7, "scl"], geobox=gbox, groupby="solar_day",
                                resampling={"scl": "nearest", "*": "bilinear"}, fail_on_error=True, chunks={}).compute()
        for k, t in enumerate(raw.time.values):
            d = str(t)[:10]
            dn = np.stack([raw[b].values[k] for b in BANDS7]).astype(np.float32)
            r = np.where(dn == 0, np.nan, dn * scale).astype(np.float32)
            s = raw["scl"].values[k]
            if d in days:
                take = (scl[d] == 0) & (s != 0)
                refl[d] = np.where(take, r, refl[d])
                scl[d] = np.where(take, s, scl[d])
            else:
                days[d], refl[d], scl[d] = True, r, s
    order = sorted(days)
    if not order:
        return [], np.zeros((0, 7, *gbox.shape), np.float32), np.zeros((0, *gbox.shape), np.uint8)
    return order, np.stack([refl[d] for d in order]), np.stack([scl[d] for d in order])


def _grid(ps: list[dict]):
    from affine import Affine
    from odc.geo.geobox import GeoBox
    x0, y1 = min(p["x0"] for p in ps), max(p["y0"] for p in ps)
    x1, y0 = max(p["x0"] for p in ps) + PATCH * PIX, min(p["y0"] for p in ps) - PATCH * PIX
    w, h = round((x1 - x0) / PIX), round((y1 - y0) / PIX)
    return GeoBox((h, w), Affine(PIX, 0, x0, 0, -PIX, y1), f"EPSG:{ps[0]['epsg']}"), (x0, y1)


def _windows(d: date, stage: str):
    (b0, b1), (a0, a1) = WINDOWS[stage]
    return (d + timedelta(days=b0), d + timedelta(days=b1)), (d + timedelta(days=a0), d + timedelta(days=a1))


def _patch(p: dict, dates: list[str], x: np.ndarray, stage: str, last: bool) -> bool:
    """Median images and the rule's map for one patch; saves it and returns True, or returns False (try wider)."""
    import xarray as xr

    from core.models.baseline import TS_BANDS, detect_change_timeseries
    d = date.fromisoformat(p["date"])
    wb, wa = _windows(d, stage)
    t = np.array(dates, dtype="datetime64[D]")
    ib = (t >= np.datetime64(wb[0])) & (t <= np.datetime64(wb[1]))
    ia = (t >= np.datetime64(wa[0])) & (t <= np.datetime64(wa[1]))
    nb = np.isfinite(x[ib, 2]).sum(0)
    na = np.isfinite(x[ia, 2]).sum(0)
    usable = (nb >= 2) & (na >= 1)
    dest = OUT / f"{p['name']}.npz"
    if usable.mean() < 0.5:
        if last:
            np.savez_compressed(dest, dropped=True, usable_fraction=float(usable.mean()))
            return True
        return False
    sel = ib | ia
    ds = xr.Dataset({b: (("time", "y", "x"), x[sel, k]) for k, b in enumerate(BANDS7)},
                    coords={"time": t[sel].astype("datetime64[ns]"),
                            "y": np.arange(PATCH) * -PIX, "x": np.arange(PATCH) * PIX})
    r = detect_change_timeseries(ds, wb, wa, bands=TS_BANDS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before, after = np.nanmedian(x[ib], axis=0), np.nanmedian(x[ia], axis=0)
    np.savez_compressed(dest, before=before.astype(np.float16), after=after.astype(np.float16),
                        nb=nb.astype(np.uint8), na=na.astype(np.uint8), usable=usable,
                        score=r["_score"].astype(np.float16), label=label(p["name"]), stage=stage,
                        n_dates_before=int(ib.sum()), n_dates_after=int(ia.sum()))
    return True


def _classes_gpu(x3: np.ndarray) -> np.ndarray:
    """OmniCloudMask (library 1.8.1, model version 4) on the GPU, as in experiments/b3_ocm.py."""
    from omnicloudmask import predict_from_array
    a = np.nan_to_num(x3, nan=0.0).astype(np.float32)
    out = predict_from_array(a, inference_device="cuda", mosaic_device="cuda", model_version=4.0,
                             model_download_source="google_drive",
                             destination_model_dir=ROOT / "data_cache" / "models" / "omnicloudmask")
    return out[0].astype(np.int8)


def build_block(key: str) -> str:
    from core.data.cloud_mask import valid_mask
    from core.data.timeseries import FILE_FAULTS
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        classes = _classes_gpu
    else:
        from core.models.cloud import classes
    todo = [p for p in _blocks_cached()[key] if not (OUT / f"{p['name']}.npz").exists()]
    if not todo:
        return "cached"
    report = []
    for stage in ("normal", "wide"):
        if not todo:
            break
        gbox, (gx0, gy1) = _grid(todo)
        d = date.fromisoformat(todo[0]["date"])
        (b0, _), (_, a1) = _windows(d, stage)
        lonlat = gbox.geographic_extent.boundingbox
        items = _items(lonlat, b0, a1)
        dates, x, scl = _load(items, gbox) if items else ([], None, None)
        for k in range(len(dates)):     # cloud mask per date, as the pipeline does
            cc = classes(x[k][[2, 1, 4]])
            bad = ~(valid_mask(scl[k], FILE_FAULTS) & (cc == 0) & np.isfinite(x[k]).all(0))
            x[k][:, bad] = np.nan
        left = []
        for p in todo:
            i, j = round((gy1 - p["y0"]) / PIX), round((p["x0"] - gx0) / PIX)
            xp = x[:, :, i:i + PATCH, j:j + PATCH] if dates else np.zeros((0, 7, PATCH, PATCH), np.float32)
            if not _patch(p, dates, xp, stage, last=stage == "wide"):
                left.append(p)
        report.append(f"{stage}: {len(items)} products, {len(dates)} days, {len(todo) - len(left)} of {len(todo)} done")
        todo = left
    return "; ".join(report)


GPUS = os.environ.get("B5_GPUS", "1,2,4,5").split(",")


def _init(gpu: bool = False) -> None:
    """Each worker: one GPU for the cloud mask (chosen by process number), or ONNX Runtime with 4 threads."""
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = GPUS[os.getpid() % len(GPUS)]
        return
    import onnxruntime as ort

    from core.models import cloud
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 4, 1

    def session(net, _cache={}):
        if net not in _cache:
            _cache[net] = ort.InferenceSession(str(cloud.MODELS / f"ocm_v4_{net}.onnx"), so, providers=["CPUExecutionProvider"])
        return _cache[net]
    cloud._session = session


def main(workers: int = 16) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bl = blocks()
    keys = sorted(bl, key=lambda k: -len(bl[k]))     # large blocks first
    log = DIR / "build_log.jsonl"
    try:
        import omnicloudmask  # noqa: F401  (present in .venv-ml: cloud masks on the GPUs)
        gpu = True
    except ImportError:
        gpu = False
    print(f"cloud masks on the {'GPUs ' + ','.join(GPUS) if gpu else 'CPU'}", flush=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(gpu,)) as pool:
        futs = {k: pool.submit(build_block, k) for k in keys}
        for n, (k, f) in enumerate(futs.items(), 1):
            try:
                msg = f.result()
            except Exception as e:      # recorded; the block is retried on the next run
                msg = f"ERROR {type(e).__name__}: {str(e)[:300]}"
            with log.open("a") as fh:
                fh.write(json.dumps({"block": k, "msg": msg}) + "\n")
            if n % 20 == 0:
                print(f"{n}/{len(keys)} blocks", flush=True)


def check(fires=("EMSR532", "EMSR534", "EMSR596")) -> dict:
    """Alignment: our B04 on a date against ImpactMesh's own B04 of the same date (its S2L2A.tar of the test
    part), and after a one-pixel shift. Needs the extracted zarr files in DIR/S2L2A (core files only, read
    with zipfile and numcodecs)."""
    import zipfile

    import numcodecs
    out = {}
    pmap = {p["name"]: p for p in patches()}
    for fire in fires:
        name = next(n for n in sorted(pmap) if n.startswith(fire + "_") and pmap[n]["part"] == "test_holdout"
                    and (DIR / "S2L2A" / f"{n}_S2L2A.zarr.zip").exists())
        p = pmap[name]
        z = zipfile.ZipFile(DIR / "S2L2A" / f"{name}_S2L2A.zarr.zip")
        meta = json.loads(z.read(".zmetadata"))["metadata"]

        def arr(k):
            a = meta[f"{k}/.zarray"]
            chunk = ".".join("0" * len(a["shape"]))
            raw = numcodecs.get_codec(a["compressor"]).decode(z.read(f"{k}/{chunk}"))
            return np.frombuffer(raw, dtype=a["dtype"]).reshape(a["shape"])
        bands = [str(b) for b in arr("band")]
        times = [str(np.datetime64(int(v), "ns"))[:10] for v in arr("time")]
        own = arr("bands")
        gbox, _ = _grid([p])
        best = None
        for k, day in enumerate(times):
            its = [it for it in _items(gbox.geographic_extent.boundingbox, date.fromisoformat(day), date.fromisoformat(day))]
            if not its:
                continue
            dates, x, _ = _load(its, gbox)
            a = own[k, bands.index("B04")].astype(np.float64)
            b = x[0, 2].astype(np.float64)
            m = np.isfinite(b) & (a > 0)
            if m.mean() < 0.5:
                continue
            def corr(u, v, mm):
                return float(np.corrcoef(u[mm], v[mm])[0, 1])
            shifted = [corr(a[1:], b[:-1], m[1:] & m[:-1]), corr(a[:, 1:], b[:, :-1], m[:, 1:] & m[:, :-1])]
            best = dict(patch=name, day=day, corr=corr(a, b, m), shifted=shifted, ratio=float(np.median(b[m] / (a[m] * 1e-4))))
            break
        out[fire] = best
        print(fire, best, flush=True)
    (DIR / "check_alignment.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    {"download": download, "check": check}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
