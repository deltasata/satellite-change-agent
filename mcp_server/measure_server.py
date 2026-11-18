"""MCP server: offers one function, measure_change, the whole measurement in one call.

It is a separate program from mcp_server/server.py, so that the step-by-step agents still see exactly
their four step functions, and the chat agent sees only this one and cannot skip the measurement.

Run alone (it then waits for MCP messages, one per line, on its input):
    source activate.sh && python -m mcp_server.measure_server
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from core.measure import MeasureResult
from core.measure import measure_change as _measure_change
from mcp_server.server import Lat, Lon, Size, anticipated

logging.getLogger("rasterio").setLevel(logging.WARNING)

server = MCPServer(name="eo-measure", version="0.1.0", instructions=(
    "One function, measure_change, answers whether a flood, construction or vegetation loss happened "
    "at a place between two times, by measurement in Sentinel-2 satellite images (from July 2015)."))

Day = Annotated[date | None, Field(description="YYYY-MM-DD")]


@server.tool()
@anticipated
def measure_change(
    lat: Lat, lon: Lon,
    kind: Annotated[Literal["flood", "construction", "vegetation_loss"],
                    Field(description="flood; construction (new buildings, roads); vegetation_loss (fire, clearing)")],
    event_date: Annotated[date | None, Field(description="the day of the event, if the question names one; "
                                             "then the periods are chosen for you")] = None,
    before_start: Day = None, before_end: Day = None, after_start: Day = None, after_end: Day = None,
    size_km: Size = 5.0,
) -> MeasureResult:
    """Answer one question of the form "did this kind of change happen at this place between these
    times?" by measurement, with fixed code and no judgement: choose and check the periods, get and
    cloud-mask the Sentinel-2 images, measure, apply the decision rule (flood: new water >= 0.05 of the
    area, with water pixels marked by a U-Net network trained on hand-labelled flood images; other
    kinds: changed fraction >= 0.03), widen the periods once if clouds leave too few clear
    images, and draw the evidence figure. For a flood, if clouds still hide the ground, radar images
    (which see through cloud) of the first periods are used instead, with water pixels marked by a
    TerraMind network; the result then says sensor "radar" and has no figure. Returns answer yes / no /
    cannot_tell, the rule written out with the numbers, the periods actually used, the image counts,
    the sensor and the figure path.
    Give either event_date (then before = day-60 .. day-10 and after = day-3 .. day+3), or all four
    period dates. A period you give replaces the one computed from event_date. Each period is at most
    120 days long."""
    return _measure_change(lat=lat, lon=lon, kind=kind, event_date=event_date, before_start=before_start,
                           before_end=before_end, after_start=after_start, after_end=after_end, size_km=size_km)


if __name__ == "__main__":
    server.run("stdio")
