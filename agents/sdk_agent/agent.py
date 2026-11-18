"""Variant (1): the Claude Agent SDK. The model holds the orchestration: it decides
which tool to call, in which order, and when to stop.

Zero-cost safeguards:
- refuses to start if ANTHROPIC_API_KEY is set;
- aborts at session start unless the CLI reports apiKeySource == "none", i.e. a
  claude.ai subscription login (Pro), never an API key;
- max_turns caps a runaway loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agents.common import ANSWER_SCHEMA, DOMAIN_GUIDE

ROOT = Path(__file__).resolve().parents[2]
TOOL_NAMES = ["search_imagery", "cloud_mask_and_stack", "detect_change", "render_evidence"]
DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = DOMAIN_GUIDE + """

Finish with the final structured answer: answer, confidence, explanation (cite the numbers you
measured), evidence_path."""


class PaidCredentialError(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """The subscription's usage limit was hit. The runner stops (never pays, never retries)."""


def _options(model: str) -> ClaudeAgentOptions:
    if os.environ.get("ANTHROPIC_API_KEY"):
        raise PaidCredentialError("ANTHROPIC_API_KEY is set; refusing to run (zero-money rule)")
    cwd = ROOT / "runs" / "agent_cwd"  # neutral working dir: not the repo, so no project files
    cwd.mkdir(parents=True, exist_ok=True)
    return ClaudeAgentOptions(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[],  # no built-in tools (no Bash, Read, Web...)
        mcp_servers={"eo": {"type": "stdio", "command": sys.executable,
                            "args": ["-m", "mcp_server.server"],
                            "env": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
                                    **{k: os.environ[k] for k in ("EO_DATA_CACHE", "MPLCONFIGDIR") if k in os.environ}}}},
        strict_mcp_config=True,
        allowed_tools=[f"mcp__eo__{t}" for t in TOOL_NAMES],
        permission_mode="dontAsk",
        setting_sources=[],  # do not load project instructions, user settings or hooks
        output_format={"type": "json_schema", "schema": ANSWER_SCHEMA},
        max_turns=15,
        cwd=str(cwd),
    )


async def _run(question: str, model: str) -> dict:
    tool_calls, msg_ids, init = [], set(), {}
    result: ResultMessage | None = None
    text = []
    async for msg in query(prompt=question, options=_options(model)):
        if isinstance(msg, SystemMessage) and msg.subtype == "init":
            init = msg.data
            if init.get("apiKeySource") != "none":
                raise PaidCredentialError(f"apiKeySource={init.get('apiKeySource')!r}; only a "
                                          "subscription login is allowed (zero-money rule)")
        elif isinstance(msg, AssistantMessage):
            msg_ids.add(msg.message_id or id(msg))
            for b in msg.content:
                if isinstance(b, ToolUseBlock):
                    tool_calls.append({"tool": b.name.removeprefix("mcp__eo__"), "input": b.input})
                elif isinstance(b, TextBlock):
                    text.append(b.text)
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            for b in msg.content:
                if isinstance(b, ToolResultBlock) and tool_calls:
                    c = b.content if isinstance(b.content, str) else json.dumps(b.content)
                    tool_calls[-1]["result"] = (c or "")[:600]
                    tool_calls[-1]["is_error"] = bool(b.is_error)
        elif isinstance(msg, ResultMessage):
            result = msg
    if result is not None and result.is_error:
        blob = f"{result.api_error_status} {result.errors} {result.result}".lower()
        if result.api_error_status in (429, 529) or "limit" in blob or "quota" in blob or "usage" in blob:
            raise QuotaExhausted(blob[:300])
    out = result.structured_output if result else None
    if not isinstance(out, dict):
        out = {"answer": "error", "confidence": None, "explanation": (result.result if result else "") or "",
               "evidence_path": None}
    return dict(
        answer=out.get("answer"), confidence=out.get("confidence"), explanation=out.get("explanation"),
        evidence=out.get("evidence_path"), n_llm_calls=len(msg_ids), tool_calls=tool_calls,
        model=init.get("model"), api_key_source=init.get("apiKeySource"),
        num_turns=result.num_turns if result else None,
        result_subtype=result.subtype if result else None,
        result_errors=result.errors if result else None,
        usage=result.usage if result else None,
    )


class AuthBlocked(RuntimeError):
    """Login or network edge refused us (401/403, e.g. a Cloudflare challenge). Not an agent
    failure: the runner stops, nothing is saved, and the question is retried later."""


def answer_question(question: str, model: str = DEFAULT_MODEL) -> dict:
    try:
        return asyncio.run(_run(question, model))
    except Exception as e:  # the SDK raises ResultError for an error result
        msg = str(e).lower()
        if "authenticate" in msg or " 403" in msg or " 401" in msg or "cloudflare" in msg:
            raise AuthBlocked(str(e)[:300]) from e
        if "rate limit" in msg or "usage limit" in msg or " 429" in msg:
            raise QuotaExhausted(str(e)[:300]) from e
        raise


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or ("Was there new construction in the 3 km x 3 km area centred on "
                                   "52.3950°N, 13.7950°E (Grünheide, near Berlin) between summer 2019 "
                                   "and summer 2021?")
    print(json.dumps(answer_question(q), indent=1, default=str))
