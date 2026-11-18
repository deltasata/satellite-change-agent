"""MCP server: offers the four eo-agent functions to other programs, as lines of text on its input and output.

Every agent variant connects to this same server, so differences between the
variants cannot come from the tools.

Run alone (it then waits for MCP messages, one per line, on its input):
    source activate.sh && python -m mcp_server.server
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

import functools
import logging

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from core import tools

INSTRUCTIONS = """Tools for answering questions about land-surface change (floods, construction,
vegetation loss) from Sentinel-2 satellite imagery (10 m pixels, available from about 2017).
Typical sequence: search_imagery (optional, cheap) -> cloud_mask_and_stack for a BEFORE window
-> cloud_mask_and_stack for an AFTER window (same lat/lon/size_km) -> detect_change ->
render_evidence. Each window may be at most 120 days. Clouds often make windows unusable; then
widen the window or answer 'cannot tell'."""

# rasterio (the image-reading library) prints a line per image read; show only its warnings.
logging.getLogger("rasterio").setLevel(logging.WARNING)

server = MCPServer(name="eo-agent", instructions=INSTRUCTIONS, version="0.1.0")


def anticipated(fn):
    """Our own ValueError/KeyError messages are written for the caller (e.g. 'window too long,
    use two shorter windows'), so pass them through as ToolError. Any other exception is a
    crash: the SDK hides its text from the model and logs the traceback on the server."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError) as e:
            raise ToolError(str(e)) from e

    return wrapper

Lat = Annotated[float, Field(ge=-85, le=85, description="latitude of the area centre, degrees north")]
Lon = Annotated[float, Field(ge=-180, le=180, description="longitude of the area centre, degrees east")]
Size = Annotated[float, Field(ge=0.5, le=20, description="side of the square area in km (default 5)")]
Start = Annotated[date, Field(description="first day, YYYY-MM-DD")]
End = Annotated[date, Field(description="last day (inclusive), YYYY-MM-DD")]


@server.tool()
@anticipated
def search_imagery(lat: Lat, lon: Lon, start: Start, end: End, size_km: Size = 5.0) -> tools.SearchOutput:
    """Cheap catalog check: how many Sentinel-2 acquisitions exist for the area and period.
    No pixels are read, and tile cloud cover is only a rough guide for a small area."""
    return tools.search_imagery(lat=lat, lon=lon, start=start, end=end, size_km=size_km)


@server.tool()
@anticipated
def cloud_mask_and_stack(lat: Lat, lon: Lon, start: Start, end: End, size_km: Size = 5.0) -> tools.StackOutput:
    """Download, calibrate and cloud-mask all imagery of the area in one window (max 120 days).
    Returns a cube_id plus how many dates are usable (>= 80 % cloud-free). Call it once for the
    BEFORE window and once for the AFTER window, with identical lat, lon and size_km."""
    return tools.cloud_mask_and_stack(lat=lat, lon=lon, start=start, end=end, size_km=size_km)


@server.tool(description=tools.detect_change.__doc__)
@anticipated
def detect_change(before_cube_id: str, after_cube_id: str) -> tools.ChangeOutput:
    return tools.detect_change(before_cube_id, after_cube_id)


@server.tool()
@anticipated
def render_evidence(result_id: str, title: str = "") -> tools.EvidenceOutput:
    """Write the evidence figure for a detect_change result; returns the PNG path to cite."""
    return tools.render_evidence(result_id, title)


if __name__ == "__main__":
    server.run("stdio")
