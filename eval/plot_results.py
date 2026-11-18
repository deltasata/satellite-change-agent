"""Results figure: accuracy per variant (with 68 % bootstrap interval) and LLM calls per question.

Two panels, two measures, two separate y-scales: never one dual-axis chart.
  python -m eval.plot_results
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.metrics import load, solvable_ids, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, MUTED, S1, S2 = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b3", "#2a78d6", "#eb6834"
ORDER = ["reference", "sdk_sonnet-5-5", "graph-claude_sonnet-5-5",
         "graph-local_qwen3-14b-awq-v2", "graph-local_qwen3-14b-awq",
         "graph-local_qwen3-8b-thinking", "graph-local_qwen3-8b-v2", "graph-local_qwen3-8b",
         "graph-local_qwen3-4b-v2", "graph-local_qwen3-4b"]
LABEL = {"reference": "reference (no LLM, true windows)", "sdk_sonnet-5-5": "(1) Agent SDK + Claude Sonnet 5.5",
         "graph-claude_sonnet-5-5": "(3) LangGraph + Claude Sonnet 5.5",
         "graph-local_qwen3-14b-awq-v2": "(2) Qwen3-14B 4-bit, graph v2", "graph-local_qwen3-14b-awq": "(2) Qwen3-14B 4-bit, graph v1",
         "graph-local_qwen3-8b-thinking": "(2) Qwen3-8B thinking, graph v1", "graph-local_qwen3-8b-v2": "(2) Qwen3-8B, graph v2",
         "graph-local_qwen3-8b": "(2) Qwen3-8B, graph v1", "graph-local_qwen3-4b-v2": "(2) Qwen3-4B, graph v2",
         "graph-local_qwen3-4b": "(2) Qwen3-4B, graph v1"}


def main():
    sol = solvable_ids()
    runs = {}
    for name in ORDER:
        d = ROOT / "runs" / "eval" / name
        recs = load(d) if d.exists() else []
        if len(recs) >= 44:  # only complete runs
            runs[name] = summarize(recs, solvable=sol)
    names = list(runs)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 0.6 * len(names) + 1.6), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.08})
    y = np.arange(len(names))[::-1]
    acc = np.array([runs[n]["accuracy"] for n in names])
    lo = np.array([runs[n]["accuracy_ci68"][0] for n in names])
    hi = np.array([runs[n]["accuracy_ci68"][1] for n in names])
    col = [MUTED if n == "reference" else (S2 if n.endswith("-v2") else S1) for n in names]
    a1.barh(y, acc, color=col, height=0.6)
    a1.errorbar(acc, y, xerr=[acc - lo, hi - acc], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, v, h in zip(y, acc, hi):
        a1.text(h + 0.01, yi, f"{v:.2f}", va="center", fontsize=9)
    a1.set_yticks(y, [LABEL[n] for n in names])
    a1.set_xlim(0, 1.0)
    a1.set_xlabel("accuracy on 44 questions (bar: 68% bootstrap interval); orange = graph v2 (window dates computed in code)")
    calls = [runs[n]["avg_llm_calls"] or 0 for n in names]
    a2.barh(y, calls, color=col, height=0.6)
    for yi, v in zip(y, calls):
        a2.text(v + 0.05, yi, f"{v:.1f}", va="center", fontsize=9)
    a2.set_yticks(y, [""] * len(names))
    a2.set_xlabel("LLM calls per question")
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same tools, same instructions: who orchestrates, and which model", x=0.01, ha="left", fontsize=11)
    out = ROOT / "explain" / "figures" / "results_variants.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
