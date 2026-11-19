"""B3, Part B: the whole answer path on exam v2 (150 questions), ESA's SCL against OmniCloudMask.
Protocol: docs/METHODS.md section 6.

Everything is `measure_change` as it is now (B1 U-Net for flood water, B2 radar when the optical images
cannot tell), with the true periods of each question (question["reference"]); only `cloud` differs. For a
question about a Sen1Floods11 flood E, the water and radar models are those that left E out (Paraguay has
no radar model of its own and gets the one trained on all events, as in B2).

Measured: accuracy against the stored expected answers, and the number of "cannot tell". Paired
comparison, resampling the exam's 83 groups of related questions (cluster bootstrap).
The protocol's condition: OmniCloudMask is not adopted if the whole interval of the accuracy difference
lies below zero.

    source activate.sh
    python -m experiments.b3_exam run scl          # answers, runs/b3/exam_scl.json (8 questions at a time)
    python -m experiments.b3_exam run ocm
    python -m experiments.b3_exam report           # runs/b3/report_B.json
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
RUNS = ROOT / "runs" / "b3"
RADAR_FOLDS = {"Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Somalia", "Spain", "Sri-Lanka", "USA"}


def questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]


def answer(q: dict, cloud: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from core.measure import measure_change
    ref = q["reference"]
    extra = {}
    if q["cluster"].startswith("flood_"):
        ev = q["cluster"].removeprefix("flood_")
        extra = dict(water=f"unet-without-{ev}", radar=ev if ev in RADAR_FOLDS else "ALL")
    t0 = time.time()
    try:
        r = measure_change(lat=ref["lat"], lon=ref["lon"], kind=ref["kind"], size_km=ref["size_km"],
                           before_start=ref["before"][0], before_end=ref["before"][1],
                           after_start=ref["after"][0], after_end=ref["after"][1], cloud=cloud, **extra)
        out = dict(answer=r.answer, sensor=r.sensor, clear_before=r.n_clear_images_before,
                   clear_after=r.n_clear_images_after, steps=r.steps)
    except ValueError as e:     # e.g. a period the function refuses; recorded, answered "cannot tell"
        out = dict(answer="cannot_tell", error=str(e)[:200])
    return dict(id=q["id"], cluster=q["cluster"], expected=q["expected"], seconds=round(time.time() - t0), **out)


def run(cloud: str, workers: int = 8) -> None:
    qs = questions()
    dest = RUNS / f"exam_{cloud}.json"
    done = {r["id"]: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    todo = [q for q in qs if q["id"] not in done]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(answer, todo, [cloud] * len(todo)), 1):
            done[r["id"]] = r
            dest.write_text(json.dumps([done[q["id"]] for q in qs if q["id"] in done], indent=1))
            if k % 10 == 0:
                print(f"  {cloud}: {k}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{cloud}: {len(done)} questions, {time.time() - t0:.0f} s")


def report() -> dict:
    from eval.metrics import cluster_bootstrap
    a = {r["id"]: r for r in json.loads((RUNS / "exam_scl.json").read_text())}
    b = {r["id"]: r for r in json.loads((RUNS / "exam_ocm.json").read_text())}
    ids = [q["id"] for q in questions() if q["id"] in a and q["id"] in b]
    cl = {i: a[i]["cluster"] for i in ids}
    ra = {i: float(a[i]["answer"] == a[i]["expected"]) for i in ids}
    rb = {i: float(b[i]["answer"] == b[i]["expected"]) for i in ids}
    boot = cluster_bootstrap({i: rb[i] - ra[i] for i in ids}, cl, n_boot=10000)
    changed = [dict(id=i, expected=a[i]["expected"], scl=a[i]["answer"], ocm=b[i]["answer"],
                    clear_scl=[a[i].get("clear_before"), a[i].get("clear_after")],
                    clear_ocm=[b[i].get("clear_before"), b[i].get("clear_after")])
               for i in ids if a[i]["answer"] != b[i]["answer"]]
    out = {"n": len(ids), "n_groups": len(set(cl.values())),
           "accuracy_scl": float(np.mean(list(ra.values()))), "accuracy_ocm": float(np.mean(list(rb.values()))),
           "diff": float(np.mean(list(rb.values())) - np.mean(list(ra.values()))),
           "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
           "cannot_tell_scl": sum(a[i]["answer"] == "cannot_tell" for i in ids),
           "cannot_tell_ocm": sum(b[i]["answer"] == "cannot_tell" for i in ids),
           "radar_answers_scl": sum(a[i].get("sensor") == "radar" for i in ids),
           "radar_answers_ocm": sum(b[i].get("sensor") == "radar" for i in ids),
           "changed": changed,
           "errors": {k: [i for i in ids if d[i].get("error")] for k, d in (("scl", a), ("ocm", b))}}
    out["adopt"] = not (out["hi"] < 0)
    (RUNS / "report_B.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        r = report()
        print(json.dumps({k: v for k, v in r.items() if k != "changed"}, indent=1))
        for c in r["changed"]:
            print("  ", c)
