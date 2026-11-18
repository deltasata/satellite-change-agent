"""A conversation with the agent: Claude talks, and measures only through measure_change (fixed code).

    python examples/chat.py            # a prepared five-message conversation (about 2-4 minutes)
    python examples/chat.py --chat     # type your own messages; an empty line ends
    add -v to print Claude's first reply too, when the code checks made it rewrite

Needs: a Qwen3 model server on port 8101 for the off-topic check (scripts/serve_llm.sh) and a Claude
Code login with a subscription. The code refuses to run if ANTHROPIC_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VERBOSE = "-v" in sys.argv
logging.getLogger("httpx").setLevel(logging.WARNING)              # one line per request to Qwen3
logging.getLogger("claude_agent_sdk").setLevel(logging.WARNING)   # "Using bundled Claude Code CLI"

PREPARED = [
    "Was the 5 km x 5 km area centred on 44.5063°N, 11.8276°E (Conselice, Italy) flooded around "
    "2023-05-23, compared with the weeks before?",
    "Please double check this.",
    "Use March 2023 as the before period instead.",
    "And was there new construction in the same area between summer 2022 and summer 2023?",
    "How much do I weigh?",
]


async def main() -> None:
    from agents.chat.agent import Conversation
    from agents.llm_backends.local_llm import LocalLLM

    scope_llm = LocalLLM(thinking=False)
    print(f"  Off-topic check: {scope_llm.name}. Conversation: Claude through the Agent SDK, subscription login.\n")
    async with Conversation(scope_llm) as c:
        messages = PREPARED if "--chat" not in sys.argv else iter(lambda: input("\nYOU: ").strip(), "")
        for i, msg in enumerate(messages, 1):
            if "--chat" not in sys.argv:
                print(f"\n=== message {i}\nPERSON: {msg}")
            turn = await c.send(msg)
            if VERBOSE and turn.rewritten:
                print(f"      Claude's first reply was:\n      {turn.claude_reply}")
            print(f"REPLY ({turn.seconds:.0f} s, {turn.n_model_calls} Claude model calls"
                  f"{', rewritten' if turn.rewritten else ''}{', replaced by code' if turn.fallback else ''}):\n"
                  f"{turn.reply}")
        n = sum(t.n_model_calls for t in c.turns)
        k = sum(len(t.calls) for t in c.turns)
        print(f"\n  In all: {len(c.turns)} messages, {k} measure_change calls, {n} Claude model calls.")


if __name__ == "__main__":
    asyncio.run(main())
