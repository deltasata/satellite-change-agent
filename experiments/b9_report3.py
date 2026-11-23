"""B9, avenue 3: the table and the decision (protocol: docs/METHODS.md section 12).

B2's Part A exactly (experiments/b2_report.py): each Sen1Floods11 event's tiles scored by the runs that left
that event out, mean of 3 seeds' water probability > 0.5, pooled water IoU over the scored pixels; candidate
M (B2's TerraMind with MMFlood added) minus B2's TerraMind, 95 % interval from resampling the 10 events
10 000 times. M passes if the whole interval is above zero.

    .venv-ml/bin/python -m experiments.b9_report3
Writes runs/b9/report_avenue3.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import sen1floods11 as s1f  # noqa: E402
from experiments import b2_common as B  # noqa: E402
from experiments.b2_report import counts_from_prob, run_dir  # noqa: E402

M = ROOT / "runs" / "b9" / "radar"


def main() -> dict:
    chips = sorted(json.loads((B.RUNS / "counts_rtc_vh_fit.json").read_text()))
    events = sorted({s1f.event(c) for c in chips})
    probs = {"ref": {s: {} for s in range(3)}, "M": {s: {} for s in range(3)}}
    for e in events:
        for s in range(3):
            probs["ref"][s].update(dict(np.load(run_dir("terramind", "s1", "rtc", e, s) / "prob.npz")))
            probs["M"][s].update(dict(np.load(M / f"m_{e}_s{s}" / "prob.npz")))
    rows = {k: counts_from_prob([v[s] for s in range(3)], chips, "s1", "rtc") for k, v in probs.items()}
    seeds = {k: [B.pooled_iou(counts_from_prob([v[s]], chips, "s1", "rtc"), chips) for s in range(3)] for k, v in probs.items()}
    d, lo, hi, n = B.bootstrap_difference(rows["ref"], rows["M"], chips)
    stored = json.loads((B.RUNS / "report_A.json").read_text())["terramind"]["iou"]
    rep = {"n_tiles": len(chips), "n_events": len(events),
           "ref_iou": B.pooled_iou(rows["ref"], chips), "ref_iou_stored_B2": stored,
           "M_iou": B.pooled_iou(rows["M"], chips), "M_minus_ref": [d, lo, hi], "passes": lo > 0, "seeds": seeds,
           "per_event": {e: {k: B.pooled_iou(rows[k], [c for c in chips if s1f.event(c) == e]) for k in rows} for e in events}}
    (ROOT / "runs" / "b9" / "report_avenue3.json").write_text(json.dumps(rep, indent=1))
    print(f"{len(chips)} tiles, {len(events)} events. B2 TerraMind {rep['ref_iou']:.3f} (stored {stored:.3f}); "
          f"with MMFlood {rep['M_iou']:.3f}; difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> passes: {rep['passes']}")
    print("single seeds:", {k: [round(x, 3) for x in v] for k, v in seeds.items()})
    for e, v in rep["per_event"].items():
        print(f"  {e:10s} B2 {v['ref']:.3f}  M {v['M']:.3f}  {v['M'] - v['ref']:+.3f}")
    return rep


if __name__ == "__main__":
    main()
