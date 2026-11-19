"""B4, Parts B and C: the answer path with the rule against the Part A winner (protocol: docs/METHODS.md section 7).

Part B: the 45 exam v2 questions whose reference kind is "construction" (15 construction, 26 stable, 4 any
change), with the true periods; everything as in `measure_change` now (OmniCloudMask clouds), only the change
detector differs. The other 105 answers do not depend on it and are taken from B3's run with OmniCloudMask
(runs/b3/exam_ocm.json). Accuracy over all 150; paired comparison over the exam's 83 groups. The winner is not
adopted if the whole interval of the accuracy difference lies below zero.

Part C: the 78 flood questions asked as construction questions (same place, size and true periods, kind
"construction"). Expected answer "no" for every one. Measured: the fraction answered "yes" (false alarms).
Paired comparison over the flood events. The winner is not adopted if the whole interval shows more false
alarms than the rule. Also the Conselice case, between summer 2022 and summer 2023 (after the May 2023
flood), reported alone.

The exam's Paris question lies partly inside OSCD's Paris area, so it is answered by the networks trained
without Paris and Bercy.

    python -m experiments.b4_exam run rule B && python -m experiments.b4_exam run fsd B     # or terramind
    python -m experiments.b4_exam run rule C && python -m experiments.b4_exam run fsd C
    python -m experiments.b4_exam report fsd
Writes runs/b4/exam_<detector>_<part>.json and report_BC.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs" / "b4"
INSIDE_OSCD = {"stable_paris_luxembourg": "paris_bercy"}
CONSELICE = dict(id="conselice_construction", cluster="conselice", expected="no",
                 reference=dict(kind="construction", lat=44.5063, lon=11.8276, size_km=5.0,
                                before=["2022-06-01", "2022-08-31"], after=["2023-06-01", "2023-08-31"]))
# (the question exactly as asked in the earlier conversation test: "new construction between summer 2022
# and summer 2023?", after the May 2023 flood; the rule answered "yes", 3.9 % changed)


def questions(part: str) -> list[dict]:
    qs = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    if part == "B":
        return [q for q in qs if q["reference"] and q["reference"].get("kind") == "construction"]
    out = []
    for q in qs:
        if q["category"] == "flood":
            out.append({**q, "expected": "no", "reference": {**q["reference"], "kind": "construction"}})
    return out + [CONSELICE]


def answer(q: dict, detector: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from core.measure import measure_change
    ref = q["reference"]
    change = detector if detector == "rule" or q["id"] not in INSIDE_OSCD else f"{detector}-without-{INSIDE_OSCD[q['id']]}"
    t0 = time.time()
    try:
        r = measure_change(lat=ref["lat"], lon=ref["lon"], kind=ref["kind"], size_km=ref["size_km"],
                           before_start=ref["before"][0], before_end=ref["before"][1],
                           after_start=ref["after"][0], after_end=ref["after"][1], change=change)
        out = dict(answer=r.answer, changed_fraction=(r.measurement or {}).get("changed_fraction"),
                   clear_before=r.n_clear_images_before, clear_after=r.n_clear_images_after, steps=r.steps)
    except ValueError as e:     # a period the function refuses; recorded, answered "cannot tell"
        out = dict(answer="cannot_tell", error=str(e)[:200])
    return dict(id=q["id"], cluster=q["cluster"], expected=q["expected"], change=change,
                seconds=round(time.time() - t0), **out)


def run(detector: str, part: str, workers: int = 8) -> None:
    qs = questions(part)
    dest = RUNS / f"exam_{detector}_{part}.json"
    done = {r["id"]: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    todo = [q for q in qs if q["id"] not in done]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(answer, todo, [detector] * len(todo)), 1):
            done[r["id"]] = r
            dest.write_text(json.dumps([done[q["id"]] for q in qs if q["id"] in done], indent=1))
            if k % 10 == 0:
                print(f"  {detector} {part}: {k}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{detector} {part}: {len(done)} questions, {time.time() - t0:.0f} s")


def _read(detector: str, part: str) -> dict:
    return {r["id"]: r for r in json.loads((RUNS / f"exam_{detector}_{part}.json").read_text())}


def report(winner: str) -> dict:
    from eval.metrics import cluster_bootstrap
    out = {"winner": winner}
    # Part B
    rule, net = _read("rule", "B"), _read(winner, "B")
    rest = {r["id"]: r for r in json.loads((ROOT / "runs" / "b3" / "exam_ocm.json").read_text())}
    allq = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    ids = [q["id"] for q in allq]
    cl = {q["id"]: q["cluster"] for q in allq}
    cat = {q["id"]: q["category"] for q in allq}
    a = {i: (rule.get(i) or rest[i])["answer"] for i in ids}
    b = {i: (net.get(i) or rest[i])["answer"] for i in ids}
    exp = {q["id"]: q["expected"] for q in allq}
    ra = {i: float(a[i] == exp[i]) for i in ids}
    rb = {i: float(b[i] == exp[i]) for i in ids}
    boot = cluster_bootstrap({i: rb[i] - ra[i] for i in ids}, cl, n_boot=10000)
    out["B"] = {"n": len(ids), "n_groups": len(set(cl.values())),
                "accuracy_rule": float(np.mean(list(ra.values()))), "accuracy_winner": float(np.mean(list(rb.values()))),
                "diff": float(np.mean(list(rb.values())) - np.mean(list(ra.values()))),
                "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                "the_45": {"rule": sum(ra[i] for i in rule), "winner": sum(rb[i] for i in rule), "n": len(rule)},
                "rule_reproduces_b3": sum(rule[i]["answer"] == rest[i]["answer"] for i in rule),
                "by_category": {c: {"rule": sum(ra[i] for i in rule if cat[i] == c), "winner": sum(rb[i] for i in rule if cat[i] == c),
                                    "n": sum(cat[i] == c for i in rule)} for c in sorted({cat[i] for i in rule})},
                "changed": [dict(id=i, expected=exp[i], rule=a[i], winner=b[i],
                                 cf_rule=rule[i].get("changed_fraction"), cf_winner=net[i].get("changed_fraction"))
                            for i in rule if a[i] != b[i]]}
    out["B"]["adopt"] = not (out["B"]["hi"] < 0)
    # Part C
    rule, net = _read("rule", "C"), _read(winner, "C")
    fl = [i for i in rule if i in net and i != "conselice_construction"]
    fa = {i: float(rule[i]["answer"] == "yes") for i in fl}
    fb = {i: float(net[i]["answer"] == "yes") for i in fl}
    bootc = cluster_bootstrap({i: fb[i] - fa[i] for i in fl}, {i: rule[i]["cluster"] for i in fl}, n_boot=10000)
    out["C"] = {"n": len(fl), "n_events": len({rule[i]["cluster"] for i in fl}),
                "false_alarms_rule": int(sum(fa.values())), "false_alarms_winner": int(sum(fb.values())),
                "cannot_tell_rule": sum(rule[i]["answer"] == "cannot_tell" for i in fl),
                "cannot_tell_winner": sum(net[i]["answer"] == "cannot_tell" for i in fl),
                "diff": float(np.mean(list(fb.values())) - np.mean(list(fa.values()))),
                "lo": float(np.percentile(bootc, 2.5)), "hi": float(np.percentile(bootc, 97.5)),
                "conselice": {"rule": [rule["conselice_construction"]["answer"], rule["conselice_construction"].get("changed_fraction")],
                              "winner": [net["conselice_construction"]["answer"], net["conselice_construction"].get("changed_fraction")]},
                "changed_fraction_median": {"rule": float(np.median([rule[i]["changed_fraction"] for i in fl if rule[i].get("changed_fraction") is not None])),
                                            "winner": float(np.median([net[i]["changed_fraction"] for i in fl if net[i].get("changed_fraction") is not None]))}}
    out["C"]["adopt"] = not (out["C"]["lo"] > 0)
    out["adopt"] = out["B"]["adopt"] and out["C"]["adopt"]
    (RUNS / "report_BC.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        print(json.dumps(report(sys.argv[2]), indent=1))
