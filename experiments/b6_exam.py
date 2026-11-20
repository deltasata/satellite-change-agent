"""B6, step 0: the whole exam (150 questions) with today's answer path, keeping every measured number
(protocol: docs/METHODS.md section 9).

Everything is `measure_change` as it is now: OmniCloudMask clouds (B3), the U-Net for flood water (B1),
radar TerraMind when optical images cannot tell (B2), the Siamese TerraMind for construction (B4) and the
burn U-Net for vegetation loss (B5); the true periods of each question. As in B3 and B4, a question about
a Sen1Floods11 flood is answered by the water and radar networks that left that flood out, and the one
question inside an OSCD city (Paris) by the change networks that left Paris and Bercy out.

The answers are those of today's fixed thresholds; B6 then asks whether thresholds fitted on other
questions, or a small classifier, answer better (experiments/b6_fit.py).

    source activate.sh
    python -m experiments.b6_exam          # runs/b6/exam.json, 8 questions at a time; resumes if stopped
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs" / "b6"
RADAR_FOLDS = {"Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Somalia", "Spain", "Sri-Lanka", "USA"}
INSIDE_OSCD = {"stable_paris_luxembourg": "paris_bercy"}


def questions() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]


def extra_arguments(q: dict) -> dict:
    """The left-out networks for questions about places the networks were trained on."""
    if q["cluster"].startswith("flood_"):
        ev = q["cluster"].removeprefix("flood_")
        return dict(water=f"unet-without-{ev}", radar=ev if ev in RADAR_FOLDS else "ALL")
    if q["id"] in INSIDE_OSCD:
        from core.measure import CHANGE_DETECTOR
        return dict(change=f"{CHANGE_DETECTOR}-without-{INSIDE_OSCD[q['id']]}")
    return {}


def answer(q: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from core.measure import measure_change
    ref = q["reference"]
    t0 = time.time()
    try:
        r = measure_change(lat=ref["lat"], lon=ref["lon"], kind=ref["kind"], size_km=ref["size_km"],
                           before_start=ref["before"][0], before_end=ref["before"][1],
                           after_start=ref["after"][0], after_end=ref["after"][1], **extra_arguments(q))
        out = dict(answer=r.answer, sensor=r.sensor, measurement=r.measurement, reason=r.reason,
                   clear_before=r.n_clear_images_before, clear_after=r.n_clear_images_after,
                   images_before=r.n_images_before, images_after=r.n_images_after,
                   water_detector=r.water_detector, change_detector=r.change_detector, steps=r.steps)
    except ValueError as e:     # a period the function refuses; recorded, answered "cannot tell"
        out = dict(answer="cannot_tell", error=str(e)[:200])
    return dict(id=q["id"], cluster=q["cluster"], category=q["category"], kind=(q["reference"] or {}).get("kind"),
                expected=q["expected"], seconds=round(time.time() - t0), **out)


def run(workers: int = 8) -> None:
    qs = [q for q in questions() if q["reference"]]
    RUNS.mkdir(parents=True, exist_ok=True)
    dest = RUNS / "exam.json"
    done = {r["id"]: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    todo = [q for q in qs if q["id"] not in done]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(answer, todo), 1):
            done[r["id"]] = r
            dest.write_text(json.dumps([done[q["id"]] for q in qs if q["id"] in done], indent=1, default=str))
            if k % 10 == 0:
                print(f"  {k}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"{len(done)} questions, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    run()
