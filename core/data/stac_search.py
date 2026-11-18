"""Find Sentinel-2 scenes in a free STAC catalog.

Pipeline:  catalog query  ->  catalog items  ->  Scene objects
           ->  remove duplicate processings  ->  cloud filter  ->  sort by time
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from core.schemas import Scene, SearchRequest, SearchResult

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
# The assets our pipeline reads. Each must be a free public HTTPS file.
REQUIRED_ASSETS = ("blue", "green", "red", "nir", "swir16", "scl")


def is_free_https(href: str) -> bool:
    """Only the public sentinel-cogs HTTPS files are allowed. Some catalog entries point to
    s3://sentinel-s2-l2a/..., a REQUESTER-PAYS bucket: reading it bills the reader's AWS
    account. Zero-money rule: such links are never followed."""
    return href.startswith("https://")


def item_to_scene(item: dict[str, Any]) -> Scene:
    """Convert one STAC item (as a JSON dict) into our Scene schema.

    Keeping this a pure function of a dict means it can be tested offline
    on saved catalog responses.
    """
    p = item["properties"]
    # Scale and offset are identical for all reflectance bands; read them from "red".
    band_meta = item["assets"]["red"]["raster:bands"][0]
    # MEASURED 2026-11-13, twice:
    # (1) tiles flagged "earthsearch:boa_offset_applied"=true: pixels already have
    #     ESA's -0.1 offset removed although raster:bands still says offset=-0.1.
    # (2) benchmark v2 build: tiles of baseline 04.00 flagged FALSE also have it
    #     removed: same day, adjacent tiles, vegetation red 0.049 (offset ignored) vs 0.051 (flagged
    #     tile); applying -0.1 gave -0.051 (unphysical).
    # Conclusion for Earth Search v1: the pixels never contain the offset, whatever the flag says.
    offset_in_pixels = bool(p.get("earthsearch:boa_offset_applied", False))
    offset = 0.0  # effective offset for Earth Search COGs (the advertised one is kept only as a record)
    return Scene(
        id=item["id"],
        datetime=datetime.fromisoformat(p["datetime"].replace("Z", "+00:00")),
        platform=p["platform"],
        tile=p["grid:code"],
        cloud_cover=p["eo:cloud_cover"],
        nodata_percentage=p.get("s2:nodata_pixel_percentage", 0.0),
        processing_baseline=p["s2:processing_baseline"],
        sequence=int(p.get("s2:sequence", 0)),
        reflectance_scale=band_meta.get("scale", 1.0),
        reflectance_offset=offset,
        offset_already_in_pixels=offset_in_pixels,
        readable=all(k in item["assets"] and is_free_https(item["assets"][k]["href"]) for k in REQUIRED_ASSETS),
    )


def deduplicate(scenes: Iterable[Scene]) -> list[Scene]:
    """Keep one scene per (tile, acquisition day): a readable copy first, then the most
    recent processing (highest baseline, then highest sequence).

    The catalog can hold the same acquisition several times (the original
    processing and later reprocessings). Mixing versions would put an
    artificial calibration jump into the time series. Readability ranks first:
    some reprocessed copies exist only in the requester-pays bucket.
    """
    def rank(s: Scene):
        return (s.readable, float(s.processing_baseline), s.sequence)

    best: dict[tuple[str, str], Scene] = {}
    for s in scenes:
        key = (s.tile, s.datetime.date().isoformat())
        if key not in best or rank(s) > rank(best[key]):
            best[key] = s
    return list(best.values())


def scenes_from_items(items: Iterable[dict[str, Any]], request: SearchRequest) -> SearchResult:
    """Everything after the network call: convert, deduplicate, filter, sort."""
    all_scenes = [item_to_scene(it) for it in items]
    unique = deduplicate(all_scenes)
    readable = [s for s in unique if s.readable]
    kept = [s for s in readable if s.cloud_cover <= request.max_cloud_cover]
    kept.sort(key=lambda s: (s.datetime, s.tile))
    return SearchResult(
        request=request,
        scenes=kept,
        n_found=len(all_scenes),
        n_duplicates_removed=len(all_scenes) - len(unique),
        n_too_cloudy=len(readable) - len(kept),
        n_unreadable=len(unique) - len(readable),
    )


def fetch_items(
    request: SearchRequest, catalog_url: str = EARTH_SEARCH_URL, max_items: int = 5000
) -> list[dict[str, Any]]:
    """The only function here that touches the network."""
    from pystac_client import Client  # imported here so offline tests do not need it

    client = Client.open(catalog_url)
    search = client.search(
        collections=[request.collection],
        bbox=request.bbox.as_list(),
        datetime=f"{request.start.isoformat()}T00:00:00Z/{request.end.isoformat()}T23:59:59Z",
        max_items=max_items,
    )
    return [item.to_dict() for item in search.items()]


def search_imagery(request: SearchRequest, catalog_url: str = EARTH_SEARCH_URL) -> SearchResult:
    """Find all usable Sentinel-2 scenes for an area and date range."""
    return scenes_from_items(fetch_items(request, catalog_url), request)


def fetch_items_by_id(
    ids: list[str], collection: str = "sentinel-2-l2a", catalog_url: str = EARTH_SEARCH_URL
):
    """Re-fetch full catalog items (as pystac Items) for a list of scene ids.

    Tools pass small *references* (scene ids) between each other, not large
    objects; whoever needs the full item fetches it here.
    """
    from pystac_client import Client

    client = Client.open(catalog_url)
    items = list(client.search(collections=[collection], ids=ids, max_items=len(ids)).items())
    missing = set(ids) - {it.id for it in items}
    if missing:
        raise LookupError(f"catalog did not return scenes: {sorted(missing)}")
    order = {i: k for k, i in enumerate(ids)}
    return sorted(items, key=lambda it: order[it.id])
