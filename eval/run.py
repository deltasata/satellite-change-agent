"""Evaluation runner: answer every benchmark question with one variant, save each result.

Usage:
  python -m eval.run --variant reference                 # no LLM, true parameters
  python -m eval.run --variant sdk --ids flood_ghana_yes0  # one question
  python -m eval.run --variant graph-local --limit 5

Results go to runs/eval/<variant>/<question id>.json. Existing results are skipped
(use --force to redo), so a run interrupted by a quota limit simply resumes.

Every variant implements:  answer_question(question_text) -> dict with keys
  answer ("yes" | "no" | "cannot_tell"), explanation, evidence (path or None),
  n_llm_calls, tool_calls (list)
The runner adds: id, expected, elapsed_s, error.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "benchmark.jsonl"


def load_questions(ids: list[str] | None = None, bench: Path = BENCH) -> list[dict]:
    qs = [json.loads(l) for l in Path(bench).read_text().splitlines() if l.strip()]
    return [q for q in qs if not ids or q["id"] in ids]


def results_root(bench) -> Path:
    """runs/eval for benchmark.jsonl, runs/eval_v2 for benchmark_v2.jsonl, ..."""
    stem = Path(bench).stem
    return ROOT / "runs" / ("eval" + stem.removeprefix("benchmark").replace("_", "_", 1))


def get_variant(name: str, **kw):
    """Return a function question_dict -> result dict."""
    if name == "reference":
        from eval.reference_pipeline import answer

        def run(q):
            r = answer(q)
            return dict(answer=r["answer"], explanation=r.get("reason") or json.dumps(r.get("change")),
                        evidence=r.get("evidence"), n_llm_calls=0, tool_calls=r["trace"])
        return run
    if name == "sdk":
        from agents.sdk_agent.agent import answer_question
        return lambda q: answer_question(q["question"], **kw)
    if name.startswith("graph-"):
        from agents.graph.run import answer_question
        backend = name.split("-", 1)[1]  # graph-local / graph-claude
        return lambda q: answer_question(q["question"], backend=backend, **kw)
    raise ValueError(f"unknown variant {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--bench", default=str(BENCH), help="benchmark file; v2 results go to runs/eval_v2/")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the output folder, e.g. a model name")
    ap.add_argument("--model", default=None, help="model name passed to the variant (if it takes one)")
    ap.add_argument("--sleep", type=float, default=0, help="pause (s) between questions: gentle on a subscription")
    args = ap.parse_args()

    kw = {"model": args.model} if args.model else {}
    run = get_variant(args.variant, **kw)
    out_dir = results_root(args.bench) / (args.variant + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = load_questions(args.ids, args.bench)[: args.limit]
    for i, q in enumerate(qs, 1):
        path = out_dir / f"{q['id']}.json"
        if path.exists() and not args.force:
            continue
        t0 = time.time()
        try:
            res = run(q)
            err = None
        except Exception as e:  # a crash is a result too: it is counted as a failure
            if type(e).__name__ in ("QuotaExhausted", "PaidCredentialError", "AuthBlocked"):
                print(f"STOP: {type(e).__name__}: {e}. Nothing saved for {q['id']}; rerun later to resume.")
                return
            res = dict(answer="error", explanation="", evidence=None, n_llm_calls=None, tool_calls=[])
            err = "".join(traceback.format_exception(e)).strip()[-1500:]  # full chain, incl. ExceptionGroup members
        rec = dict(id=q["id"], category=q["category"], expected=q["expected"], **res,
                   elapsed_s=round(time.time() - t0, 1), error=err)
        path.write_text(json.dumps(rec, indent=1, default=str))
        if args.sleep:
            time.sleep(args.sleep)
        ok = "✓" if rec["answer"] == q["expected"] else "✗"
        print(f"[{i}/{len(qs)}] {ok} {q['id']:30s} expected={q['expected']:11s} got={rec['answer']:11s} "
              f"{rec['elapsed_s']:6.1f}s llm={rec['n_llm_calls']} {err or ''}", flush=True)


if __name__ == "__main__":
    main()
