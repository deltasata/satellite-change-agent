"""B5, Part B: the exam's 13 vegetation-loss (fire) questions, with the rule against the Part A winner
(protocol: docs/METHODS.md section 8).

Everything as in `measure_change` now (OmniCloudMask clouds, B4's construction network), true periods;
only the detector for vegetation-loss questions differs. The other 137 answers do not depend on it: the 52
construction-kind answers come from B4's run with TerraMind (runs/b4/exam_terramind_B.json), the rest from
B3's run (runs/b3/exam_ocm.json). Accuracy over all 150; paired comparison over the exam's 83 groups. The
winner is not adopted if the whole interval of the accuracy difference lies below zero. No exam fire is in
the networks' training (the fires at exam places were left out).

    python -m experiments.b5_exam run rule && python -m experiments.b5_exam run burn-unet   # or burn-terramind
    python -m experiments.b5_exam report burn-unet
Writes runs/b5/exam_<detector>.json and report_B.json.
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
RUNS = ROOT / "runs" / "b5"


def questions() -> list[dict]:
    qs = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    return [q for q in qs if q["reference"] and q["reference"].get("kind") == "vegetation_loss"]


def answer(q: dict, detector: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from core.measure import measure_change
    ref = q["reference"]
    t0 = time.time()
    try:
        r = measure_change(lat=ref["lat"], lon=ref["lon"], kind=ref["kind"], size_km=ref["size_km"],
                           before_start=ref["before"][0], before_end=ref["before"][1],
                           after_start=ref["after"][0], after_end=ref["after"][1], change=detector)
        out = dict(answer=r.answer, changed_fraction=(r.measurement or {}).get("changed_fraction"),
                   clear_before=r.n_clear_images_before, clear_after=r.n_clear_images_after,
                   detector=r.change_detector, steps=r.steps)
    except ValueError as e:
        out = dict(answer="cannot_tell", error=str(e)[:200])
    return dict(id=q["id"], cluster=q["cluster"], expected=q["expected"], seconds=round(time.time() - t0), **out)


def run(detector: str, workers: int = 7) -> None:
    qs = questions()
    dest = RUNS / f"exam_{detector}.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(answer, qs, [detector] * len(qs)))
    dest.write_text(json.dumps(res, indent=1))
    print(f"{detector}: {sum(r['answer'] == r['expected'] for r in res)} of {len(res)} right")
    for r in res:
        print(f"  {r['id']:24s} {r['answer']:12s} {r.get('changed_fraction')}")


def report(winner: str) -> dict:
    from eval.metrics import cluster_bootstrap
    allq = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    rest = {r["id"]: r for r in json.loads((ROOT / "runs" / "b3" / "exam_ocm.json").read_text())}
    rest.update({r["id"]: r for r in json.loads((ROOT / "runs" / "b4" / "exam_terramind_B.json").read_text())})
    rule = {r["id"]: r for r in json.loads((RUNS / "exam_rule.json").read_text())}
    net = {r["id"]: r for r in json.loads((RUNS / f"exam_{winner}.json").read_text())}
    ids = [q["id"] for q in allq]
    cl = {q["id"]: q["cluster"] for q in allq}
    exp = {q["id"]: q["expected"] for q in allq}
    ra = {i: float((rule.get(i) or rest[i])["answer"] == exp[i]) for i in ids}
    rb = {i: float((net.get(i) or rest[i])["answer"] == exp[i]) for i in ids}
    boot = cluster_bootstrap({i: rb[i] - ra[i] for i in ids}, cl, n_boot=10000)
    out = {"winner": winner, "n": len(ids), "n_groups": len(set(cl.values())),
           "accuracy_rule": float(np.mean(list(ra.values()))), "accuracy_winner": float(np.mean(list(rb.values()))),
           "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
           "fires": {"rule": int(sum(ra[i] for i in rule)), "winner": int(sum(rb[i] for i in rule)), "n": len(rule)},
           "per_question": [dict(id=i, rule=rule[i]["answer"], winner=net[i]["answer"], cf_rule=rule[i].get("changed_fraction"),
                                 cf_winner=net[i].get("changed_fraction")) for i in rule]}
    b3 = {r["id"]: r["answer"] for r in json.loads((ROOT / "runs" / "b3" / "exam_ocm.json").read_text())}
    out["rule_reproduces_b3"] = int(sum(rule[i]["answer"] == b3[i] for i in rule))
    out["diff"] = out["accuracy_winner"] - out["accuracy_rule"]
    out["adopt"] = not (out["hi"] < 0)
    (RUNS / "report_B.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report(sys.argv[2])
