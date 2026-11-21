"""B8, the exam step (protocol: docs/METHODS.md section 11): the exam's 13 vegetation-loss (fire) questions with
the adopted network (B7's U-TAE on every clear date, "burn-utae") against the network in use (B5's U-Net,
"burn-unet"), both run now on the same stored images. The other 137 answers do not depend on the burn detector
(as in experiments/b5_exam.py). Accuracy over all 150; paired comparison over the exam's 83 groups. Not adopted
if the whole interval of the accuracy change lies below zero.

    .venv/bin/python -m experiments.b8_exam run burn-unet && .venv/bin/python -m experiments.b8_exam run burn-utae
    .venv/bin/python -m experiments.b8_exam report
Writes runs/b8/exam_<detector>.json and runs/b8/report_exam.json.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.b5_exam import answer, questions  # noqa: E402

RUNS = ROOT / "runs" / "b8"


def run(detector: str, workers: int = 7) -> None:
    qs = questions()
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(answer, qs, [detector] * len(qs)))
    (RUNS / f"exam_{detector}.json").write_text(json.dumps(res, indent=1))
    print(f"{detector}: {sum(r['answer'] == r['expected'] for r in res)} of {len(res)} right")
    for r in res:
        print(f"  {r['id']:24s} {r['expected']:12s} {r['answer']:12s} {r.get('changed_fraction')}")


def report() -> dict:
    from eval.metrics import cluster_bootstrap
    allq = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    rest = {r["id"]: r for r in json.loads((ROOT / "runs" / "b3" / "exam_ocm.json").read_text())}
    rest.update({r["id"]: r for r in json.loads((ROOT / "runs" / "b4" / "exam_terramind_B.json").read_text())})
    old = {r["id"]: r for r in json.loads((RUNS / "exam_burn-unet.json").read_text())}
    new = {r["id"]: r for r in json.loads((RUNS / "exam_burn-utae.json").read_text())}
    ids = [q["id"] for q in allq]
    cl = {q["id"]: q["cluster"] for q in allq}
    exp = {q["id"]: q["expected"] for q in allq}
    ra = {i: float((old.get(i) or rest[i])["answer"] == exp[i]) for i in ids}
    rb = {i: float((new.get(i) or rest[i])["answer"] == exp[i]) for i in ids}
    boot = cluster_bootstrap({i: rb[i] - ra[i] for i in ids}, cl, n_boot=10000)
    out = {"n": len(ids), "n_groups": len(set(cl.values())),
           "accuracy_unet": float(np.mean(list(ra.values()))), "accuracy_utae": float(np.mean(list(rb.values()))),
           "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
           "fires_right": {"unet": int(sum(ra[i] for i in old)), "utae": int(sum(rb[i] for i in old)), "n": len(old)},
           "per_question": [dict(id=i, expected=exp[i], unet=old[i]["answer"], utae=new[i]["answer"],
                                 cf_unet=old[i].get("changed_fraction"), cf_utae=new[i].get("changed_fraction"))
                            for i in old]}
    out["diff"] = out["accuracy_utae"] - out["accuracy_unet"]
    out["adopt"] = not (out["hi"] < 0)
    (RUNS / "report_exam.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        report()
