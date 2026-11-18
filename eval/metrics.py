"""Results table across variants, with CLUSTER bootstrap intervals.

Questions from the same flood event (or site) share weather, clouds and geography, so they
are not independent measurements. The bootstrap therefore resamples whole clusters (every
question has a `cluster` field: e.g. all Ghana flood chips form one cluster), which gives
honest, wider intervals than resampling individual questions.

  python -m eval.metrics --markdown                               # benchmark v1 (44)
  python -m eval.metrics --markdown --bench eval/benchmark_v2.jsonl
  python -m eval.metrics --compare graph-local_qwen3-8b graph-claude_sonnet-5-5 --bench ...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "eval" / "benchmark.jsonl"


def results_root(bench) -> Path:
    stem = Path(bench).stem
    return ROOT / "runs" / ("eval" + stem.removeprefix("benchmark"))


def load(variant_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(variant_dir.glob("*.json"))]


def clusters(bench=BENCH) -> dict[str, str]:
    """question id -> cluster; v1 file has no field, so derive (flood chips -> their event)."""
    out = {}
    for line in Path(bench).read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if "cluster" in q:
            out[q["id"]] = q["cluster"]
        elif q["id"].startswith("flood_"):
            out[q["id"]] = "flood_" + q["id"].split("_")[1]
        else:
            out[q["id"]] = q["id"]
    return out


def solvable_ids(bench=BENCH) -> set[str] | None:
    """Questions the no-LLM reference (true windows) answers correctly: the ceiling the tools allow."""
    d = results_root(bench) / "reference"
    if not d.exists():
        return None
    return {r["id"] for r in load(d) if r["answer"] == r["expected"]}


def cluster_bootstrap(values: dict[str, float], cl: dict[str, str], n_boot=4000, seed=0) -> np.ndarray:
    """Bootstrap distribution of the mean of `values` (id -> value), resampling clusters."""
    groups = defaultdict(list)
    for i, v in values.items():
        groups[cl.get(i, i)].append(v)
    g = list(groups.values())
    sums = np.array([sum(x) for x in g])
    ns = np.array([len(x) for x in g])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    return sums[idx].sum(1) / ns[idx].sum(1)


def summarize(recs: list[dict], n_boot: int = 4000, seed: int = 0, solvable: set[str] | None = None,
              cl: dict[str, str] | None = None) -> dict:
    cl = cl or {}
    correct = {r["id"]: float(r["answer"] == r["expected"]) for r in recs}
    boot = cluster_bootstrap(correct, cl, n_boot, seed) if recs else np.array([0.0])
    ct = [r for r in recs if r["expected"] == "cannot_tell"]
    ans = [r for r in recs if r["expected"] != "cannot_tell"]
    llm = [r["n_llm_calls"] for r in recs if r.get("n_llm_calls") is not None]
    fails = Counter()
    for r in recs:
        if r.get("error"):
            fails["crash"] += 1
        elif r["answer"] != r["expected"]:
            fails[f"{r['expected']}→{r['answer']}"] += 1
    sol = [r for r in recs if solvable and r["id"] in solvable]
    return dict(
        n=len(recs),
        accuracy=float(np.mean(list(correct.values()))) if recs else 0.0,
        accuracy_ci68=[float(x) for x in np.percentile(boot, [16, 84])],
        accuracy_ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
        accuracy_solvable=float(np.mean([r["answer"] == r["expected"] for r in sol])) if sol else None,
        n_solvable=len(sol),
        accuracy_answerable=float(np.mean([r["answer"] == r["expected"] for r in ans])) if ans else None,
        correct_abstentions=f"{sum(r['answer'] == 'cannot_tell' for r in ct)}/{len(ct)}",
        false_abstentions=f"{sum(r['answer'] == 'cannot_tell' for r in ans)}/{len(ans)}",
        avg_llm_calls=float(np.mean(llm)) if llm else None,
        avg_time_s=float(np.mean([r["elapsed_s"] for r in recs])) if recs else None,
        failure_modes=dict(fails.most_common()),
    )


def compare(a: str, b: str, bench=BENCH, n_boot=4000) -> dict:
    """Paired comparison B - A on the questions both answered: counts of fixed/broken questions,
    an exact sign test (assumes independent questions), and a cluster-bootstrap interval on the
    accuracy difference (does not)."""
    from scipy.stats import binomtest

    root, cl = results_root(bench), clusters(bench)
    A = {r["id"]: r for r in load(root / a)}
    B = {r["id"]: r for r in load(root / b)}
    ids = sorted(A.keys() & B.keys())
    d = {i: float(B[i]["answer"] == B[i]["expected"]) - float(A[i]["answer"] == A[i]["expected"]) for i in ids}
    fixed, broken = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    boot = cluster_bootstrap(d, cl, n_boot)
    return dict(n=len(ids), n_clusters=len({cl.get(i, i) for i in ids}), fixed=fixed, broken=broken,
                diff=float(np.mean(list(d.values()))),
                diff_ci95_cluster=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                sign_test_p=float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--bench", default=str(BENCH))
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare, bench=args.bench), indent=1))
        return
    root, cl = results_root(args.bench), clusters(args.bench)
    n_q = len(cl)
    solvable = solvable_ids(args.bench)
    rows = {d.name: summarize(load(d), solvable=solvable, cl=cl)
            for d in sorted(root.glob("*")) if d.is_dir() and any(d.glob("*.json"))}
    if args.markdown:
        print(f"Benchmark: {Path(args.bench).name}, {n_q} questions in {len(set(cl.values()))} clusters; "
              "intervals: cluster bootstrap")
        print()
        print("| Variant | n | Accuracy (68% CI) | Acc. on tool-solvable | Correct abstentions | False abstentions "
              "| Avg. LLM calls | Avg. time (s) | Failure modes |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k, s in rows.items():
            ci = s["accuracy_ci68"]
            fm = ", ".join(f"{a} ×{b}" for a, b in s["failure_modes"].items()) or "—"
            llm = f"{s['avg_llm_calls']:.1f}" if s["avg_llm_calls"] is not None else "—"
            sol = f"{s['accuracy_solvable']:.2f} ({s['n_solvable']})" if s["accuracy_solvable"] is not None else "—"
            print(f"| {k} | {s['n']} | {s['accuracy']:.2f} ({ci[0]:.2f}–{ci[1]:.2f}) | {sol} | "
                  f"{s['correct_abstentions']} | {s['false_abstentions']} | {llm} | {s['avg_time_s']:.0f} | {fm} |")
    else:
        print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
