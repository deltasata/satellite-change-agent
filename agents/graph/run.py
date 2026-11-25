"""Run the LangGraph workflow on one question with a chosen backend.

  python -m agents.graph.run --backend local  "Was the area ... flooded around ...?"
  python -m agents.graph.run --backend claude "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from agents.graph import graph as G
from agents.graph.graph import MCPTools, build_graph

ROOT = Path(__file__).resolve().parents[2]


def _cache_env() -> dict:
    """MCP starts the server with a minimal environment; pass our cache locations through."""
    import os
    return {k: os.environ[k] for k in ("EO_DATA_CACHE", "MPLCONFIGDIR", "XDG_CACHE_HOME") if k in os.environ}


def make_backend(backend: str, model: str | None = None):
    if backend == "local":
        from agents.llm_backends.local_llm import LocalLLM
        return LocalLLM(model=model)
    if backend == "claude":
        from agents.llm_backends.claude_sdk import ClaudeSDK
        return ClaudeSDK(model=model)
    if backend == "fake":
        from agents.llm_backends.fake import FakeLLM
        return FakeLLM()
    raise ValueError(backend)


async def _run(question: str, llm) -> dict:
    server = "mcp_server.measure_server" if G.GRAPH_VERSION >= 4 else "mcp_server.server"   # v4: measure_change
    params = StdioServerParameters(command=sys.executable, args=["-m", server], cwd=str(ROOT),
                                   env={"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT), **_cache_env()})
    async with Client(params) as client:
        graph = build_graph(llm, MCPTools(client))
        state = await graph.ainvoke({"question": question, "tool_calls": [], "path": []},
                                    {"recursion_limit": 25})
    final = state.get("final", {})
    return dict(answer=state.get("answer"), confidence=final.get("confidence"),
                explanation=final.get("explanation"), evidence=state.get("evidence"),
                n_llm_calls=llm.n_calls, tool_calls=state.get("tool_calls", []), model=llm.name,
                path=state.get("path"), plan=state.get("plan"), assessment=state.get("assessment"),
                abstain_reason=state.get("abstain_reason"))


def answer_question(question: str, backend: str = "local", model: str | None = None) -> dict:
    return asyncio.run(_run(question, make_backend(backend, model)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    ap.add_argument("--model")
    ap.add_argument("question", nargs="+")
    a = ap.parse_args()
    print(json.dumps(answer_question(" ".join(a.question), a.backend, a.model), indent=1, default=str))
