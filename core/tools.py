"""The agent-facing tools: plain Python functions with typed inputs and outputs.

These four functions are the *only* way any agent touches data. The MCP
server (mcp_server/server.py) exposes exactly these, so all three agent
variants share identical tools.

Design rules:
- Inputs are phrased the way questions are phrased: a centre point and a size
  in km, dates as YYYY-MM-DD. The LLM never has to do coordinate arithmetic.
- Outputs are small summaries (numbers, short lists), never pixel arrays.
  Large objects stay on the server; tools hand back IDs (cube_id, result_id).
- IDs are deterministic hashes of the inputs, and cubes are cached on disk,
  so repeating a question never re-downloads data.
- Errors are raised with a message that tells the caller what to do next.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import xarray as xr
from pydantic import BaseModel, Field

from core.data.stac_search import search_imagery as _search
from core.data.timeseries import DEFAULT_BANDS, load_timeseries
from core.models.baseline import detect_change_timeseries
from core.schemas import BBox, SearchRequest

MAX_WINDOW_DAYS = 120
USABLE_CLEAR_FRACTION = 0.8
CACHE_VERSION = "v3"  # v2: requester-pays duplicates excluded; v3: offset never applied, per-tile fusion


def _root() -> Path:
    return Path(os.environ.get("EO_DATA_CACHE", Path(__file__).resolve().parents[1] / "data_cache"))


def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps([CACHE_VERSION, *map(str, parts)]).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ area


class Area(BaseModel):
    lat: float = Field(ge=-85, le=85, description="latitude of the area centre, degrees north")
    lon: float = Field(ge=-180, le=180, description="longitude of the area centre, degrees east")
    size_km: float = Field(default=5.0, ge=0.5, le=20, description="side length of the square area, km")

    def bbox(self) -> BBox:
        half_lat = self.size_km / 2 / 111.32
        half_lon = self.size_km / 2 / (111.32 * math.cos(math.radians(self.lat)))
        return BBox(west=round(self.lon - half_lon, 6), south=round(self.lat - half_lat, 6),
                    east=round(self.lon + half_lon, 6), north=round(self.lat + half_lat, 6))


def _check_window(start: date, end: date) -> None:
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if (end - start).days > MAX_WINDOW_DAYS:
        raise ValueError(f"window {start}..{end} is {(end - start).days} days; the maximum is "
                         f"{MAX_WINDOW_DAYS}. Use two shorter windows (before / after) instead.")
    if start > date.today():
        raise ValueError(f"start {start} is in the future; no imagery can exist yet")


# ------------------------------------------------------------------ 1. search


class SearchOutput(BaseModel):
    area_bbox: list[float]
    start: date
    end: date
    n_scenes: int
    n_duplicates_removed: int
    scenes_per_month: dict[str, int]
    tile_cloud_cover_median: float | None = Field(
        description="median cloud cover (%) of the whole 110 km tiles; a rough guide only")
    first_date: date | None
    last_date: date | None


def search_imagery(lat: float, lon: float, start: date, end: date, size_km: float = 5.0) -> SearchOutput:
    """Catalog-level check: which Sentinel-2 acquisitions exist for the area and dates (no pixels)."""
    area = Area(lat=lat, lon=lon, size_km=size_km)
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if start > date.today():
        raise ValueError(f"start {start} is in the future; no imagery can exist yet")
    res = _search(SearchRequest(bbox=area.bbox(), start=start, end=end))
    days = sorted({s.datetime.date() for s in res.scenes})
    return SearchOutput(
        area_bbox=area.bbox().as_list(), start=start, end=end,
        n_scenes=len(days), n_duplicates_removed=res.n_duplicates_removed,
        scenes_per_month=dict(sorted(Counter(d.strftime("%Y-%m") for d in days).items())),
        tile_cloud_cover_median=float(np.median([s.cloud_cover for s in res.scenes])) if res.scenes else None,
        first_date=days[0] if days else None, last_date=days[-1] if days else None)


# ------------------------------------------------------------------ 2. stack


class StackOutput(BaseModel):
    cube_id: str | None = Field(description="pass to detect_change; null if no imagery exists")
    area_bbox: list[float]
    start: date
    end: date
    n_dates: int
    n_usable_dates: int = Field(description=f"dates with >= {USABLE_CLEAR_FRACTION:.0%} of the area cloud-free")
    usable_dates: list[date]
    clear_fraction_per_month: dict[str, float] = Field(description="mean cloud-free fraction of the area per month")


def _cube_path(cube_id: str) -> Path:
    return _root() / "cubes" / f"{cube_id}.nc"


def _load_cube(cube_id: str) -> xr.Dataset:
    p = _cube_path(cube_id)
    if not p.exists():
        raise KeyError(f"unknown cube_id {cube_id!r}; call cloud_mask_and_stack first")
    return xr.load_dataset(p, engine="h5netcdf")


def cloud_mask_and_stack(lat: float, lon: float, start: date, end: date, size_km: float = 5.0,
                         bands: tuple[str, ...] = DEFAULT_BANDS, cloud: str = "scl") -> StackOutput:
    """Download, calibrate and cloud-mask all imagery of the area in [start, end] (max 120 days).
    `bands` and `cloud` are not offered to agents through MCP; measure_change asks for 7 bands for flood
    questions (the trained water detector of B1 reads B8A and B12 too) and chooses the cloud detector
    ("scl": ESA's classification; "ocm": OmniCloudMask, milestone B3)."""
    area = Area(lat=lat, lon=lon, size_km=size_km)
    _check_window(start, end)
    bbox = area.bbox()
    # the default bands keep their old cube ids, so earlier stored cubes are still found
    cube_id = _hash("cube", bbox.as_list(), start, end, *([] if tuple(bands) == DEFAULT_BANDS else [bands]),
                    *([] if cloud == "scl" else [cloud]))
    p = _cube_path(cube_id)
    if p.exists():
        ds = xr.load_dataset(p, engine="h5netcdf")
    else:
        scenes = _search(SearchRequest(bbox=bbox, start=start, end=end)).scenes
        if not scenes:
            return StackOutput(cube_id=None, area_bbox=bbox.as_list(), start=start, end=end, n_dates=0,
                               n_usable_dates=0, usable_dates=[], clear_fraction_per_month={})
        ds = load_timeseries(scenes, bbox, bands=tuple(bands), cloud=cloud)
        crs = f"EPSG:{ds.odc.crs.epsg}"
        ds = ds.drop_vars([v for v in ds.coords if v not in ("time", "x", "y")], errors="ignore")
        ds.attrs = {"bbox": bbox.as_list(), "crs": crs, "n_scenes": len(scenes)}
        p.parent.mkdir(parents=True, exist_ok=True)
        enc = {v: {"zlib": True, "complevel": 1} for v in ds.data_vars}
        tmp = p.with_suffix(f".{os.getpid()}.tmp")  # unique per process: parallel runs share the cache
        ds.to_netcdf(tmp, encoding=enc, engine="h5netcdf")
        tmp.rename(p)  # atomic on POSIX: readers see either nothing or the complete file
    t = ds.time.values.astype("datetime64[D]").astype(object)
    vf = ds["valid_fraction"].values
    months: dict[str, list[float]] = {}
    for d, v in zip(t, vf):
        months.setdefault(d.strftime("%Y-%m"), []).append(float(v))
    return StackOutput(
        cube_id=cube_id, area_bbox=bbox.as_list(), start=start, end=end, n_dates=len(t),
        n_usable_dates=int((vf >= USABLE_CLEAR_FRACTION).sum()),
        usable_dates=[d for d, v in zip(t, vf) if v >= USABLE_CLEAR_FRACTION],
        clear_fraction_per_month={m: round(float(np.mean(v)), 2) for m, v in months.items()})


def cube_dates(cube_id: str) -> list[tuple[date, float]]:
    """The dates of a stored cube with the cloud-free fraction of the area on each."""
    ds = _load_cube(cube_id)
    t = ds.time.values.astype("datetime64[D]").astype(object)
    return list(zip(t, (float(v) for v in ds["valid_fraction"].values)))


def subset_cube(cube_id: str, keep: list[date]) -> StackOutput:
    """A stored cube with only the given dates of another cube (milestone B10: the chosen images)."""
    ds = _load_cube(cube_id)
    t = ds.time.values.astype("datetime64[D]").astype(object)
    keep = sorted(set(keep))
    sub_id = _hash("subset", cube_id, keep)
    p = _cube_path(sub_id)
    if not p.exists():
        sub = ds.isel(time=[k for k, d in enumerate(t) if d in keep])
        enc = {v: {"zlib": True, "complevel": 1} for v in sub.data_vars}
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        sub.to_netcdf(tmp, encoding=enc, engine="h5netcdf")
        tmp.rename(p)
    vf = dict(cube_dates(cube_id))
    start, end = (keep[0], keep[-1]) if keep else (None, None)
    months: dict[str, list[float]] = {}
    for d in keep:
        months.setdefault(d.strftime("%Y-%m"), []).append(vf[d])
    return StackOutput(cube_id=sub_id if keep else None, area_bbox=list(ds.attrs["bbox"]), start=start or t[0],
                       end=end or t[-1], n_dates=len(keep),
                       n_usable_dates=sum(vf[d] >= USABLE_CLEAR_FRACTION for d in keep),
                       usable_dates=[d for d in keep if vf[d] >= USABLE_CLEAR_FRACTION],
                       clear_fraction_per_month={m: round(float(np.mean(v)), 2) for m, v in months.items()})


# ------------------------------------------------------------------ 3. detect


class ChangeOutput(BaseModel):
    result_id: str
    status: str = Field(description="'ok' or 'insufficient_data' (then answer 'cannot tell' or retry with wider windows)")
    reason: str | None = None
    before_window: list[date]
    after_window: list[date]
    fraction_pixels_usable: float
    changed_fraction: float | None = Field(None, description="fraction of usable area with significant spectral change")
    changed_area_km2: float | None = None
    water_fraction_before: float | None = Field(None, description="fraction of area that is open water (NDWI > 0)")
    water_fraction_after: float | None = None
    delta_ndvi_changed_median: float | None = Field(None, description="vegetation index change in changed pixels")
    delta_ndwi_changed_median: float | None = Field(None, description="water index change in changed pixels")
    delta_ndbi_changed_median: float | None = Field(None, description="built-up index change in changed pixels")
    n_dates_before: int
    n_dates_after: int
    fraction_one_look_after: float | None = Field(
        None, description="of the usable pixels, the share seen cloud-free only once in the after window")


def _result_path(result_id: str) -> Path:
    return _root() / "results" / f"{result_id}.json"


def _combined(before_id: str, after_id: str):
    b, a = _load_cube(before_id), _load_cube(after_id)
    if b.sizes["x"] != a.sizes["x"] or b.sizes["y"] != a.sizes["y"]:
        raise ValueError("the two cubes are of different areas; stack the same lat/lon/size_km twice")
    tb = b.time.values.astype("datetime64[D]")
    ta = a.time.values.astype("datetime64[D]")
    if tb.size and ta.size and tb.max() >= ta.min():
        raise ValueError("the before window must end before the after window starts")
    a = a.assign_coords(x=b.x, y=b.y)
    ds = xr.concat([b, a], dim="time")
    win_b = (tb.min().astype(object), tb.max().astype(object))
    win_a = (ta.min().astype(object), ta.max().astype(object))
    return ds, win_b, win_a


def detect_change(before_cube_id: str, after_cube_id: str, water: str = "ndwi", change: str = "rule") -> ChangeOutput:
    """Compare a 'before' cube with an 'after' cube of the same area.

    Interpretation guide (measured on test cases):
    - status 'insufficient_data': too few clear observations. Answer "cannot tell",
      or retry with a wider window (more dates) if that makes sense for the question.
    - flooding: water_fraction_after - water_fraction_before >= 0.05 means new open
      water on >= 5 % of the area. Use this, not changed_fraction (crops growing
      also count as change).
    - construction, fire, vegetation loss: changed_fraction >= 0.03 means significant
      change. The index deltas describe its kind: delta_ndvi_changed_median <= -0.1 =
      vegetation lost (fire, clearing); delta_ndbi_changed_median > 0 = more built-up or
      bare surface. In deserts there is no vegetation, so NDVI barely changes even for
      large construction.
    - changed_fraction below ~0.01 with status 'ok' means no significant change.
    - Compare windows in the same season where possible: crops and trees change with
      the seasons, and that is not the kind of change users ask about.
    """
    # water: the water detector (core/models/water.py). Not offered to agents through MCP; the default
    # NDWI > 0 keeps the old result ids; measure_change uses the U-Net of milestone B1 for floods.
    # change: which pixels changed (core/models/change.py, milestone B4); the default "rule" keeps the old ids.
    rid = _hash("result", before_cube_id, after_cube_id, *([] if water == "ndwi" else [water]),
                *([] if change == "rule" else [f"change={change}"]))
    if _result_path(rid).exists():
        return ChangeOutput.model_validate_json(_result_path(rid).read_text())
    ds, wb, wa = _combined(before_cube_id, after_cube_id)
    r = detect_change_timeseries(ds, wb, wa, water=water, change=change)
    px_km2 = abs(float(ds.x[1] - ds.x[0]) * float(ds.y[1] - ds.y[0])) / 1e6
    out = ChangeOutput(
        result_id=rid, status=r["status"], reason=r.get("reason"),
        before_window=list(wb), after_window=list(wa),
        fraction_pixels_usable=round(r["fraction_pixels_usable"], 3),
        n_dates_before=r["n_dates_before"], n_dates_after=r["n_dates_after"],
        fraction_one_look_after=round(r["fraction_one_look_after"], 3) if "fraction_one_look_after" in r else None,
        **({} if r["status"] != "ok" else dict(
            changed_fraction=round(r["changed_fraction"], 4),
            changed_area_km2=round(float(r["_changed"].sum()) * px_km2, 3),
            water_fraction_before=round(r["water_fraction_before"], 4),
            water_fraction_after=round(r["water_fraction_after"], 4),
            delta_ndvi_changed_median=round(r["delta_ndvi_changed_median"], 3),
            delta_ndwi_changed_median=round(r["delta_ndwi_changed_median"], 3),
            delta_ndbi_changed_median=round(r["delta_ndbi_changed_median"], 3))))
    _result_path(rid).parent.mkdir(parents=True, exist_ok=True)
    _result_path(rid).write_text(json.dumps({**out.model_dump(mode="json"), "_cubes": [before_cube_id, after_cube_id],
                                                 "_water": water, "_change": change}))
    return out


# ------------------------------------------------------------------ 4. evidence


class EvidenceOutput(BaseModel):
    path: str
    description: str


def render_evidence(result_id: str, title: str = "", cloud: str | None = None) -> EvidenceOutput:
    """Write the evidence figure (before / after / change map / timeline) for a detect_change result."""
    from core.evidence import render_evidence as _render

    p = _result_path(result_id)
    if not p.exists():
        raise KeyError(f"unknown result_id {result_id!r}; call detect_change first")
    saved = json.loads(p.read_text())
    ds, wb, wa = _combined(*saved["_cubes"])
    r = detect_change_timeseries(ds, wb, wa, water=saved.get("_water", "ndwi"), change=saved.get("_change", "rule"))
    path = _root() / "evidence" / f"{result_id}.png"
    _render(ds, r, wb, wa, path, title or f"result {result_id}", water=saved.get("_water", "ndwi"),
            change=saved.get("_change", "rule"), cloud=cloud or "scl")
    return EvidenceOutput(path=str(path), description=(
        "before and after true-colour composites, changed pixels in orange, and the "
        "cloud-free fraction of every acquisition with both windows shaded"))
