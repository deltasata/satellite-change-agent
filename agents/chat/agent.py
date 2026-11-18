"""Claude holds the conversation; measure_change (code only) does every measurement.

One Conversation = one running Claude Code program (ClaudeSDKClient), which keeps all earlier
messages. Each message from the person goes through:

    scope_check (Qwen3 form)  --no-->  fixed reply; Claude never sees the message
         | yes
    Claude (may call measure_change, possibly several times)
         |
    answer and number check  --fails-->  Claude is asked once to rewrite  --fails again-->  the
         | passes                                                             function's own numbers
    reply to the person

Claude is reached only through the subscription login: the program stops if ANTHROPIC_API_KEY is set, or if
Claude Code reports a login other than the subscription (apiKeySource != "none").
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from agents.chat import checks

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-sonnet-5-5"  # the model used in the benchmark runs
TOOL = "mcp__eo__measure_change"

SYSTEM = f"""You answer questions about land-surface change (floods, construction, fire or vegetation loss)
at a place on Earth, measured in Sentinel-2 satellite images. Today is {date.today().isoformat()}.

You have one function, measure_change. Every answer must come from it: call it for every question,
and call it again, with changed inputs, for every follow-up that asks for a new or a checked
measurement. Never answer from your own knowledge.

Filling in the inputs:
- lat, lon and size_km from the question (size_km = side of the square area; default 5).
- kind: flood; construction (new buildings, roads, a large structure); vegetation_loss (fire, burn,
  clearing, damage to vegetation). A question about "any change" is about construction.
- An event "around" a day D: event_date = D. An event "in month M": before = the two months before M,
  after = M up to two weeks after it. "Between period A and period B": before = A, after = B.
  Seasons are meteorological: winter = December to February, spring = March to May, summer = June to
  August, autumn = September to November.
- "Double check" means: measure again with a different before period of similar length that ends
  just before the event, and compare.

Reply format: first line exactly "Answer: yes", "Answer: no" or "Answer: cannot tell", equal to the
answer of your last measure_change call. Then at most three sentences with the numbers from the
result (you may round them or write fractions as percentages). Do not compute new numbers yourself.
For a follow-up that compares two measurements, the first line is the answer of the latest one."""

REWRITE = ("CHECK BY CODE FAILED: {problems}. Rewrite your previous reply. Use only numbers that appear "
           "in the measure_change results, and make the first line equal to the answer of the last result. "
           "Do not call any function.")


class PaidLoginError(RuntimeError):
    pass


@dataclass
class Turn:
    message: str
    on_topic: bool
    scope_reason: str
    reply: str = ""
    claude_reply: str = ""             # Claude's first reply, before any rewrite
    calls: list = field(default_factory=list)   # (inputs, result) of each measure_change call
    check_problems: list = field(default_factory=list)
    rewritten: bool = False
    fallback: bool = False
    n_model_calls: int = 0
    seconds: float = 0.0


def measure_server_config() -> dict:
    keep = ("EO_DATA_CACHE", "MPLCONFIGDIR", "XDG_CACHE_HOME", "PATH")
    return {"type": "stdio", "command": sys.executable, "args": ["-m", "mcp_server.measure_server"],
            "env": {"PYTHONNOUSERSITE": "1", "PYTHONPATH": str(ROOT),
                    **{k: os.environ[k] for k in keep if k in os.environ}}}


class Conversation:
    """Use as: async with Conversation(scope_llm) as c: turn = await c.send("...")"""

    def __init__(self, scope_llm, model: str = MODEL, log=print):
        if os.environ.get("ANTHROPIC_API_KEY"):
            raise PaidLoginError("ANTHROPIC_API_KEY is set; refusing to run (zero-money rule)")
        from claude_agent_sdk import ClaudeAgentOptions

        folder = ROOT / "runs" / "agent_cwd"
        folder.mkdir(parents=True, exist_ok=True)
        self.options = ClaudeAgentOptions(
            model=model, system_prompt=SYSTEM, tools=[], mcp_servers={"eo": measure_server_config()},
            strict_mcp_config=True, allowed_tools=[TOOL], permission_mode="dontAsk",
            setting_sources=[], max_turns=8, cwd=str(folder))
        self.scope_llm, self.log = scope_llm, log
        self.turns: list[Turn] = []
        # Texts whose numbers Claude may use: the function description, today's date (it is in Claude's
        # instructions; added 17 November after the exam, where it caused the one false alarm), and,
        # as the conversation goes on, the person's messages and every measure_change result.
        self.sources: list[str] = [self._tool_description(), date.today().isoformat()]
        self.last_result: dict | None = None
        self.client = None
        self.started = False

    @staticmethod
    def _tool_description() -> str:
        from mcp_server.measure_server import measure_change
        return measure_change.__doc__ or ""

    async def __aenter__(self):
        from claude_agent_sdk import ClaudeSDKClient
        self.client = ClaudeSDKClient(options=self.options)
        await self.client.connect()
        return self

    async def __aexit__(self, *exc):
        await self.client.disconnect()

    async def _ask_claude(self, text: str, turn: Turn) -> str:
        from claude_agent_sdk import (AssistantMessage, ResultMessage, SystemMessage, TextBlock,
                                      ToolResultBlock, ToolUseBlock, UserMessage)

        await self.client.query(text)
        ids, pending, reply = set(), {}, ""
        async for m in self.client.receive_response():
            if isinstance(m, SystemMessage) and m.subtype == "init":
                if m.data.get("apiKeySource") != "none":
                    raise PaidLoginError(f"apiKeySource={m.data.get('apiKeySource')!r}")
                if not self.started:  # Claude Code reports this at the start of every turn; print it once
                    offered = [t for t in m.data.get("tools", []) if t.startswith("mcp__")]
                    self.log(f"      (login: Pro, apiKeySource 'none'; functions Claude is given: {offered})")
                    self.started = True
            elif isinstance(m, AssistantMessage):
                ids.add(m.message_id or id(m))
                for b in m.content:
                    if isinstance(b, ToolUseBlock):
                        pending[b.id] = b.input
                        self.log(f"      Claude calls measure_change({json.dumps(b.input)})")
                    elif isinstance(b, TextBlock):
                        reply = b.text  # the last text block is the reply
            elif isinstance(m, UserMessage) and isinstance(m.content, list):
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        raw = b.content if isinstance(b.content, str) else \
                            " ".join(c.get("text", "") for c in (b.content or []) if isinstance(c, dict))
                        self.sources.append(raw)
                        try:
                            result = json.loads(raw)
                        except json.JSONDecodeError:
                            result = {"error": raw}
                        turn.calls.append((pending.get(b.tool_use_id), result))
                        if "answer" in result:
                            self.last_result = result
                            self.log(f"      measure_change returned: {result['answer']}; "
                                     f"{result.get('rule') or result.get('reason')}")
                        else:
                            self.log(f"      measure_change refused: {raw[:200]}")
            elif isinstance(m, ResultMessage):
                if m.is_error:
                    raise RuntimeError(f"Claude Code ended with an error: {m.subtype} {m.errors}")
        turn.n_model_calls += len(ids)
        return reply.strip()

    def _problems(self, reply: str) -> list[str]:
        out = []
        bad = checks.unsupported_numbers(reply, self.sources)
        if bad:
            out.append(f"numbers not found in any result: {', '.join(bad)}")
        mismatch = checks.answer_mismatch(reply, self.last_result)
        if mismatch:
            out.append(mismatch)
        return out

    async def send(self, message: str) -> Turn:
        t0 = time.time()
        last = next((t for t in reversed(self.turns) if t.on_topic), None)  # the last answered question
        s = await asyncio.to_thread(checks.scope_check, self.scope_llm, message,
                                    last and last.message, last and last.reply)
        turn = Turn(message=message, on_topic=s.on_topic == "yes", scope_reason=s.reason)
        self.turns.append(turn)
        self.log(f"      scope check: {s.on_topic} ({s.reason})")
        if not turn.on_topic:
            turn.reply = checks.OUT_OF_SCOPE_REPLY
            turn.seconds = time.time() - t0
            return turn

        self.sources.append(message)
        reply = turn.claude_reply = await self._ask_claude(message, turn)
        problems = self._problems(reply)
        if problems:
            turn.check_problems.append(problems)
            self.log(f"      CHECK FAILED: {'; '.join(problems)} -> Claude is asked to rewrite")
            turn.rewritten = True
            reply = await self._ask_claude(REWRITE.format(problems="; ".join(problems)), turn)
            problems = self._problems(reply)
            if problems:
                turn.check_problems.append(problems)
                self.log(f"      CHECK FAILED AGAIN: {'; '.join(problems)} -> the function's own numbers are used")
                turn.fallback = True
                reply = checks.fallback_reply(self.last_result)
        turn.reply = reply
        turn.seconds = time.time() - t0
        return turn
