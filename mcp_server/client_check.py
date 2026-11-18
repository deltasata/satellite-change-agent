"""Smoke test: start the MCP server as a subprocess, list its tools, call one."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]


def server_params() -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(ROOT),
                                 env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": ""})


async def main():
    async with Client(server_params()) as c:
        tl = await c.list_tools()
        for t in tl.tools:
            print(f"- {t.name}: params {list(t.input_schema.get('properties', {}))}")
        r = await c.call_tool("search_imagery", {"lat": 48.157, "lon": 11.595, "start": "2024-07-01", "end": "2024-07-31"})
        print("isError:", r.is_error)
        print(json.dumps(r.structured_content)[:300])
        r = await c.call_tool("cloud_mask_and_stack", {"lat": 48.157, "lon": 11.595, "start": "2024-01-01", "end": "2024-12-31"})
        print("guardrail isError:", r.is_error, "|", r.content[0].text[:160])


if __name__ == "__main__":
    asyncio.run(main())
