"""Variant (3) backend: Claude through the Claude Agent SDK on the subscription login.
No tools, no settings files, short system prompt, JSON-schema output. One question to
the model per call (the SDK may use a second internal turn to deliver structured output)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage, query

from agents.sdk_agent.agent import DEFAULT_MODEL, PaidCredentialError, QuotaExhausted

ROOT = Path(__file__).resolve().parents[2]


class ClaudeSDK:
    def __init__(self, model: str | None = None):
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise PaidCredentialError("ANTHROPIC_API_KEY is set; refusing to run (zero-money rule)")
        self.model = model or DEFAULT_MODEL
        self.name = f"claude:{self.model}"
        self.n_calls = 0

    async def _call(self, system: str, prompt: str, json_schema: dict) -> str:
        cwd = ROOT / "runs" / "agent_cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        opts = ClaudeAgentOptions(model=self.model, system_prompt=system, tools=[], mcp_servers={},
                                  strict_mcp_config=True, setting_sources=[], permission_mode="dontAsk",
                                  output_format={"type": "json_schema", "schema": json_schema},
                                  max_turns=3, cwd=str(cwd))
        ids, result = set(), None
        async for m in query(prompt=prompt, options=opts):
            if isinstance(m, SystemMessage) and m.subtype == "init" and m.data.get("apiKeySource") != "none":
                raise PaidCredentialError(f"apiKeySource={m.data.get('apiKeySource')!r}")
            elif isinstance(m, AssistantMessage):
                ids.add(m.message_id or id(m))
            elif isinstance(m, ResultMessage):
                result = m
        self.n_calls += max(1, len(ids))
        if result is not None and result.is_error:
            blob = f"{result.api_error_status} {result.errors} {result.result}".lower()
            if result.api_error_status in (429, 529) or "limit" in blob or "quota" in blob:
                raise QuotaExhausted(blob[:300])
        import json
        if result is not None and result.structured_output is not None:
            return json.dumps(result.structured_output)
        return (result.result if result else "") or ""

    def raw_json(self, system: str, prompt: str, json_schema: dict) -> str:
        from agents.sdk_agent.agent import AuthBlocked
        try:
            return asyncio.run(self._call(system, prompt, json_schema))
        except (PaidCredentialError, QuotaExhausted):
            raise
        except Exception as e:
            msg = str(e).lower()
            if "authenticate" in msg or " 403" in msg or " 401" in msg or "cloudflare" in msg:
                raise AuthBlocked(str(e)[:300]) from e
            if "rate limit" in msg or "usage limit" in msg or " 429" in msg:
                raise QuotaExhausted(str(e)[:300]) from e
            raise
