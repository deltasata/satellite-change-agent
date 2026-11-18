"""Evaluation of the chat agent: Claude in conversation, measuring only through measure_change.

    python -m eval.a7_eval scope        # the off-topic check alone (Qwen3; no Claude): exam questions,
                                        # scope_questions.jsonl, and the follow-up messages in context
    python -m eval.a7_eval exam         # the 44-question exam, one new conversation per question
                                        # (Claude, subscription login; 30 s pause between questions; resumes)
    python -m eval.a7_eval followups    # the conversations of eval/followups.jsonl (Claude)

Results: runs/eval/a7-chat_sonnet-5-5/<id>.json (read by `python -m eval.metrics`) and
runs/a7/followups.json.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from agents.chat import checks
from agents.chat.agent import MODEL, Conversation
from agents.llm_backends.local_llm import LocalLLM

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "eval"
EXAM_DIR = ROOT / "runs" / "eval" / f"a7-chat_{MODEL.removeprefix('claude-')}"
PAUSE_S = 30  # rapid Claude sessions were once blocked


def jsonl(name: str) -> list[dict]:
    return [json.loads(x) for x in (HERE / name).read_text().splitlines() if x.strip()]


def part_scope() -> None:
    llm = LocalLLM(thinking=False)
    sets = {"exam (44)": [(q["question"], "yes") for q in jsonl("benchmark.jsonl")],
            "exam v2 (150)": [(q["question"], "yes") for q in jsonl("benchmark_v2.jsonl")],
            "scope_questions (first message)": [(q["question"], q["expected_on_topic"])
                                                for q in jsonl("scope_questions.jsonl")]}
    for name, qs in sets.items():
        wrong = [(e, s.on_topic, q, s.reason) for q, e in qs
                 if (s := checks.scope_check(llm, q)).on_topic != e]
        print(f"{name}: {len(qs) - len(wrong)}/{len(qs)} as expected")
        for e, got, q, why in wrong:
            print(f"   expected {e}, got {got}: {q}\n      reason: {why}")
    # Follow-ups: each message is judged with the conversation's first question as the previous one,
    # and a short stand-in answer (the real answer is only known after Claude runs).
    n = ok = 0
    for conv in jsonl("followups.jsonl"):
        first = conv["turns"][0]["message"]
        for t in conv["turns"][1:]:
            s = checks.scope_check(llm, t["message"], first, "Answer: yes. (measured numbers follow)")
            n += 1
            ok += (s.on_topic == "yes") == t["on_topic"]
            if (s.on_topic == "yes") != t["on_topic"]:
                print(f"   follow-up expected {t['on_topic']}, got {s.on_topic}: {t['message']}  ({s.reason})")
    print(f"follow-up messages, in context: {ok}/{n} as expected")


async def part_exam() -> None:
    EXAM_DIR.mkdir(parents=True, exist_ok=True)
    llm = LocalLLM(thinking=False)
    for q in jsonl("benchmark.jsonl"):
        out = EXAM_DIR / f"{q['id']}.json"
        if out.exists():
            continue
        t0 = time.time()
        rec = dict(id=q["id"], category=q["category"], expected=q["expected"])
        try:
            async with Conversation(llm, log=lambda *a: None) as c:
                turn = await c.send(q["question"])
            said = checks.stated_answer(turn.reply) if turn.on_topic else "out_of_scope"
            rec.update(answer=said or "no_answer", explanation=turn.reply, claude_reply=turn.claude_reply,
                       n_llm_calls=turn.n_model_calls, measure_calls=turn.calls, on_topic=turn.on_topic,
                       check_problems=turn.check_problems, rewritten=turn.rewritten, fallback=turn.fallback,
                       measured_answer=(c.last_result or {}).get("answer"), error=None)
        except Exception as e:  # a crash is recorded, never counted as an answer
            rec.update(answer="error", error=f"{type(e).__name__}: {e}"[:500], n_llm_calls=None)
            if "apiKeySource" in str(e) or "ANTHROPIC_API_KEY" in str(e):
                raise
        rec["elapsed_s"] = round(time.time() - t0, 1)
        out.write_text(json.dumps(rec, indent=1, default=str))
        print(f"{q['id']:32s} expected {q['expected']:12s} got {rec['answer']:12s} "
              f"calls {len(rec.get('measure_calls') or [])} checks {'FIRED' if rec.get('check_problems') else 'ok'} "
              f"{rec['elapsed_s']:.0f} s", flush=True)
        await asyncio.sleep(PAUSE_S)


def _turn_ok(t: dict, turn) -> list[str]:
    """Compare one turn with what was written down in advance. Returns the differences."""
    bad = []
    if turn.on_topic != t["on_topic"]:
        bad.append(f"on_topic {turn.on_topic}, expected {t['on_topic']}")
    if not t["on_topic"] or not turn.on_topic:
        return bad
    if "measure" in t and bool(turn.calls) != t["measure"]:
        bad.append(f"{len(turn.calls)} measure_change calls, expected {'some' if t['measure'] else 'none'}")
    inputs = turn.calls[-1][0] if turn.calls else {}
    for k in ("kind", "before_start", "before_end", "after_end", "event_date", "size_km"):
        if k in t and str(inputs.get(k)) != str(t[k]):
            bad.append(f"{k} = {inputs.get(k)}, expected {t[k]}")
    if "answer" in t and checks.stated_answer(turn.reply) != t["answer"]:
        bad.append(f"answer {checks.stated_answer(turn.reply)}, expected {t['answer']}")
    return bad


async def part_followups() -> None:
    llm = LocalLLM(thinking=False)
    report = []
    for conv in jsonl("followups.jsonl"):
        print(f"\n=== {conv['id']}")
        async with Conversation(llm, log=lambda *a: None) as c:
            prev_inputs = None
            for t in conv["turns"]:
                turn = await c.send(t["message"])
                bad = _turn_ok(t, turn)
                if t.get("inputs_change") and turn.calls and turn.calls[-1][0] == prev_inputs:
                    bad.append("the inputs did not change")
                if turn.calls:
                    prev_inputs = turn.calls[-1][0]
                report.append(dict(conversation=conv["id"], message=t["message"], expected=t, reply=turn.reply,
                                   calls=turn.calls, check_problems=turn.check_problems,
                                   n_model_calls=turn.n_model_calls, seconds=round(turn.seconds, 1), differences=bad))
                print(f"  {'OK  ' if not bad else 'DIFF'} {t['message'][:70]:70s} calls {len(turn.calls)} "
                      f"checks {'FIRED' if turn.check_problems else 'ok'}  {'; '.join(bad)}", flush=True)
        await asyncio.sleep(PAUSE_S)
    out = ROOT / "runs" / "a7" / "followups.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    n_ok = sum(not r["differences"] for r in report)
    print(f"\n{n_ok}/{len(report)} turns as written down in advance; details in {out}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else ""
    if part == "scope":
        part_scope()
    elif part == "exam":
        asyncio.run(part_exam())
    elif part == "followups":
        asyncio.run(part_followups())
    else:
        sys.exit(__doc__)
