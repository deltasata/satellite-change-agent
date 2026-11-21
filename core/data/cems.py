"""A new test set of burnt land: the Copernicus Emergency Management Service's wildfire maps of 2026 (milestone
B8, protocol docs/METHODS.md section 11).

Why: the 42 held-out ImpactMesh fires have been used for two decisions (B5, B7). The rapid-mapping portal of
the Copernicus Emergency Management Service (CEMS) publishes, without an account, the burnt-area maps it
draws after each wildfire activation. Every wildfire activation that started after ImpactMesh's newest fire
(25 January 2026) is a fire no network here was trained or tested on.

Labels follow ImpactMesh's own rule, reproduced exactly on EMSR859: per mapped area,
the grading map ("GRA"), or the last delineation map ("DEL") if there is none; every pixel that a "burnt
area" polygon touches is burnt. Unlike ImpactMesh, land outside the mapped area is unlabelled (-1).
Patches: 256 x 256 pixels of 10 m on a 2 560 m grid in the area's UTM zone; kept if at least half lies
inside the mapped area. A patch's date is the latest image date of its map.

    python -m core.data.cems list        # the activations and the chosen map per area -> DIR/maps.json
    python -m core.data.cems download    # the chosen maps (burnt areas and mapped area only)
    python -m core.data.cems patches     # patches.json and labels/<name>.npy
Then the images, with the B5 and B7 code pointed here:
    BURN_SOURCE=cems2026 .venv-ml/bin/python -m core.data.impactmesh_es
    BURN_SOURCE=cems2026 .venv-ml/bin/python -m core.data.impactmesh_ts
Map data: (c) European Union, Copernicus Emergency Management Service.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = Path(os.environ.get("EO_DATA_CACHE", ROOT / "data_cache")) / "burn" / "cems"
API = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
AFTER = "2026-01-26"            # ImpactMesh's newest fire is from 25 January 2026
PATCH, PIX = 256, 10.0
CELL = PATCH * PIX              # 2 560 m
MIN_INSIDE = 0.5


def _get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


def choose(aoi: dict) -> dict | None:
    """The map ImpactMesh would use for one area: the grading map, or else the delineation map with the highest
    monitoring number; only finished maps that can be downloaded."""
    ps = [p for p in aoi["products"] if p["type"] in ("GRA", "DEL") and p["downloadPath"]
          and (p.get("version") or {}).get("statusCode") == "F"]
    gra = [p for p in ps if p["type"] == "GRA"]
    if gra:
        return max(gra, key=lambda p: p.get("monitoringNumber") or 0)
    dels = [p for p in ps if p["type"] == "DEL"]
    return max(dels, key=lambda p: p.get("monitoringNumber") or 0) if dels else None


def list_maps() -> list[dict]:
    """Wildfire activations after AFTER and, per area, the chosen map (cached in DIR/maps.json)."""
    f = DIR / "maps.json"
    if f.exists():
        return json.loads(f.read_text())
    DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads(_get(f"{API}/public-activations-info/?limit=1000"))["results"]
    fires = sorted(x["code"] for x in info if x["category"] == "Wildfire" and x["eventTime"][:10] >= AFTER)
    out = []
    for code in fires:
        a = json.loads(_get(f"{API}/public-activations/?code={code}"))["results"][0]
        for aoi in a["aois"]:
            p = choose(aoi)
            if p is None:
                continue
            out.append(dict(fire=code, aoi=f"{code}_{aoi['number']}", event_time=a["eventTime"],
                            country=", ".join(c["name"] for c in a["countries"]), name=a["name"],
                            type=p["type"], monitoring=p.get("monitoringNumber"), url=p["downloadPath"],
                            date=max(i["acquisitionTime"] for i in p["images"])[:10],
                            sensors=sorted({i["sensorName"] for i in p["images"]})))
        time.sleep(0.3)
    f.write_text(json.dumps(out, indent=1))
    return out


def _get_version(m: dict) -> bytes:
    """The map's zip file. Some links name a version that is not on the server (EMSR914 area 1 lists v2, only v1
    exists): then the highest earlier version of the same map is taken, and noted in maps_used.json."""
    import re
    import urllib.error
    url = m["url"]
    v = int(re.search(r"_v(\d+)\.zip$", url).group(1))
    for k in range(v, 0, -1):
        try:
            data = _get(re.sub(r"_v\d+\.zip$", f"_v{k}.zip", url), tries=2)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        if k != v:
            f = DIR / "maps_used.json"
            used = json.loads(f.read_text()) if f.exists() else {}
            used[m["aoi"]] = f"v{k} instead of v{v} (v{v} not on the server)"
            f.write_text(json.dumps(used, indent=1))
        return data
    raise FileNotFoundError(url)


def download() -> None:
    """Each chosen map's burnt-area and mapped-area layers (GeoJSON), from its zip file."""
    for m in list_maps():
        dest = DIR / "maps" / m["aoi"]
        if (dest / "observedEventA.json").exists():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        z = zipfile.ZipFile(io.BytesIO(_get_version(m)))
        for kind in ("observedEventA", "areaOfInterestA"):
            names = [n for n in z.namelist() if n.endswith(".json") and f"_{kind}_" in n]
            feats = []
            for n in names:
                feats += json.loads(z.read(n))["features"]
            (dest / f"{kind}.json").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        print(m["aoi"], m["type"], m["date"], flush=True)


def _shapes(path: Path, burnt_only: bool = False):
    from shapely.geometry import shape
    feats = json.loads(path.read_text())["features"]
    if burnt_only:
        feats = [f for f in feats if is_burnt(f["properties"])]
    return [shape(f["geometry"]) for f in feats if f["geometry"]]


def is_burnt(props: dict) -> bool:
    """A polygon of the burnt-area layer that marks burnt land. In the 63 maps of 2026 every polygon is "Burnt
    area" (forest fire, or brush and pasture); the test is kept in case other notations appear."""
    return "burnt" in str(props.get("notation", "")).lower()


def utm(lon: float, lat: float) -> int:
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def patches() -> list[dict]:
    """Cut every mapped area into patches and write their labels (cached in DIR/patches.json)."""
    f = DIR / "patches.json"
    if f.exists():
        return json.loads(f.read_text())
    from pyproj import Transformer
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely import STRtree
    from shapely.geometry import box
    from shapely.ops import transform, unary_union
    (DIR / "labels").mkdir(parents=True, exist_ok=True)
    out = []
    for m in list_maps():
        src = DIR / "maps" / m["aoi"]
        area = unary_union(_shapes(src / "areaOfInterestA.json"))
        burnt = _shapes(src / "observedEventA.json", burnt_only=True)
        c = area.centroid
        epsg = utm(c.x, c.y)
        to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
        to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            area_u = transform(to_utm, area)
            burnt_u = [transform(to_utm, g) for g in burnt]
        tree = STRtree(burnt_u) if burnt_u else None
        x0, y0, x1, y1 = area_u.bounds
        for i in range(math.floor(x0 / CELL), math.ceil(x1 / CELL)):
            for j in range(math.floor(y0 / CELL), math.ceil(y1 / CELL)):
                cell = box(i * CELL, j * CELL, (i + 1) * CELL, (j + 1) * CELL)
                if not cell.intersects(area_u):
                    continue
                tr = from_origin(i * CELL, (j + 1) * CELL, PIX, PIX)
                inside = rasterize([(area_u.intersection(cell), 1)], out_shape=(PATCH, PATCH), transform=tr,
                                   fill=0).astype(bool)
                if inside.mean() < MIN_INSIDE:
                    continue
                hit = [burnt_u[k] for k in tree.query(cell)] if tree is not None else []
                b = (rasterize([(g, 1) for g in hit], out_shape=(PATCH, PATCH), transform=tr, fill=0,
                               all_touched=True).astype(bool) if hit else np.zeros((PATCH, PATCH), bool))
                lab = np.where(inside, b.astype(np.int8), np.int8(-1)).astype(np.int8)
                name = f"{m['aoi']}_{epsg}_x{int(i * CELL)}_y{int((j + 1) * CELL)}"
                np.save(DIR / "labels" / f"{name}.npy", lab)
                lon, lat = to_ll((i + 0.5) * CELL, (j + 0.5) * CELL)
                out.append(dict(name=name, fire=m["fire"], aoi=m["aoi"], date=m["date"], epsg=epsg,
                                x0=i * CELL, y0=(j + 1) * CELL, part="cems2026", lat=lat, lon=lon,
                                burnt=float((lab == 1).sum() / max((lab >= 0).sum(), 1))))
        print(m["aoi"], sum(p["aoi"] == m["aoi"] for p in out), "patches", flush=True)
    f.write_text(json.dumps(out))
    return out


def label(name: str) -> np.ndarray:
    return np.load(DIR / "labels" / f"{name}.npy")


if __name__ == "__main__":
    {"list": lambda: print(len(list_maps()), "maps"), "download": download,
     "patches": lambda: print(len(patches()), "patches")}[sys.argv[1]]()
