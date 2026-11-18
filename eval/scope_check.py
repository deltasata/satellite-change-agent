"""Experiment: does the graph's first box (check_question) let every real question through and
stop the off-topic ones? Only that one form is filled, one model call per question.

    python -m eval.scope_check                  # Qwen3-8B on the model server, reasoning off
    python -m eval.scope_check --backend claude # Claude through the subscription login (one call per question)

Questions: the 44-question exam and the 150-question exam v2 (all should be 'yes'), and
eval/scope_questions.jsonl (off-topic ones should be 'no', reworded real ones 'yes').
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.common import DOMAIN_GUIDE
from agents.graph.graph import SCOPE_TASK, Scope
from agents.graph.run import make_backend
from agents.llm_backends.base import complete

HERE = Path(__file__).parent


def load(name: str, expected: str | None = None) -> list[tuple[str, str]]:
    rows = [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]
    return [(r["question"], expected or r["expected_on_topic"]) for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    args = ap.parse_args()
    llm = make_backend(args.backend)
    sets = {"exam (44)": load("benchmark.jsonl", "yes"),
            "exam v2 (150)": load("benchmark_v2.jsonl", "yes"),
            "scope_questions": load("scope_questions.jsonl")}
    print(f"model: {llm.name}")
    for name, qs in sets.items():
        wrong = []
        for q, expected in qs:
            s = complete(llm, DOMAIN_GUIDE + "\n\n" + SCOPE_TASK, f"Question: {q}", Scope)
            if s.on_topic != expected:
                wrong.append((expected, s.on_topic, q, s.reason))
        print(f"\n{name}: {len(qs) - len(wrong)}/{len(qs)} as expected")
        for expected, got, q, reason in wrong:
            print(f"  expected {expected}, got {got}: {q}\n     reason: {reason}")


if __name__ == "__main__":
    main()
