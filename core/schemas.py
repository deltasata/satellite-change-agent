"""Typed data contracts shared by every part of the system.

Each class here is a *schema*: a declared shape (field names, types, allowed
ranges) that pydantic enforces at runtime. Anything that does not fit is
rejected with an error *before* it reaches the science code. Later the MCP
server will publish these same schemas to the agents, so an LLM gets the exact
contract for every tool.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BBox(BaseModel):
    """A lon/lat rectangle in degrees (WGS84), the area to look at."""

    model_config = ConfigDict(frozen=True)

    west: float = Field(ge=-180, le=180)
    south: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)

    @model_validator(mode="after")
    def _check_order(self) -> BBox:
        if not self.west < self.east:
            raise ValueError(f"west ({self.west}) must be < east ({self.east})")
        if not self.south < self.north:
            raise ValueError(f"south ({self.south}) must be < north ({self.north})")
        return self

    def as_list(self) -> list[float]:
        """[west, south, east, north]: the order the STAC standard uses."""
        return [self.west, self.south, self.east, self.north]


class SearchRequest(BaseModel):
    """What to search for: where, when, and how cloudy a scene may be."""

    bbox: BBox
    start: date
    end: date  # inclusive
    max_cloud_cover: float = Field(default=100.0, ge=0, le=100)
    collection: str = "sentinel-2-l2a"

    @model_validator(mode="after")
    def _check_dates(self) -> SearchRequest:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) is before start ({self.start})")
        return self


class Scene(BaseModel):
    """One Sentinel-2 acquisition of one tile, as described by the catalog.

    Pixel values are stored as integers (DN, "digital numbers"). The physical
    surface reflectance is   reflectance = DN * reflectance_scale + reflectance_offset.
    The offset depends on the processing version and on whether the data
    provider already applied it, so it is stored per scene (the *effective* offset).
    """

    id: str
    datetime: datetime
    platform: str  # sentinel-2a / -2b / -2c
    tile: str  # MGRS grid square, e.g. "MGRS-32UPU"
    cloud_cover: float  # % of the whole tile flagged cloudy by ESA
    nodata_percentage: float  # % of the tile outside the satellite's swath
    processing_baseline: str  # ESA processing version, e.g. "05.10"
    sequence: int  # 0 = first processing, 1 = reprocessed, ...
    reflectance_scale: float
    reflectance_offset: float  # effective offset to apply to the pixels we download
    offset_already_in_pixels: bool  # provider removed ESA's offset before publishing
    readable: bool = True  # all needed files are free public HTTPS COGs (see stac_search)


class SearchResult(BaseModel):
    """The output of search_imagery: the scenes, plus an account of what was dropped."""

    request: SearchRequest
    scenes: list[Scene]  # sorted by time, deduplicated, cloud-filtered
    n_found: int  # what the catalog returned
    n_duplicates_removed: int  # reprocessed copies of the same acquisition
    n_too_cloudy: int  # removed by max_cloud_cover
    n_unreadable: int = 0  # only available in the requester-pays (paid) bucket: never read
