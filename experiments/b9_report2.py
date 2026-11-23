"""B9, avenue 2: the table and the decision (protocol: docs/METHODS.md section 12).

Reference W (adopted in avenue 1): B4's TerraMind runs minus the water of the after image. Candidates: S+W
(SpaceNet 7 added) and SN+W (SpaceNet 7 and avenue 1's flooded places added), each minus the water.
Part A (decides): OSCD urban change, B4's cities and scored pixels; candidate minus W, 22 groups resampled;
passes if the whole 95 % interval is above zero. Part C (condition): false "yes" on the 78 flood questions
(S+W from its ALL runs, which saw no flood; SN+W from the runs that left each event out); not adopted if the
interval's upper end of (candidate - W) is above +0.05.

    .venv/bin/python -m experiments.b9_report2
Writes runs/b9/report_avenue2.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data.oscd_pc import group  # noqa: E402
from experiments import b4_common as C  # noqa: E402
from experiments import b9_report as R  # noqa: E402

CANDS = {"SW": ("s", "S+W: SpaceNet 7 added"), "SNW": ("sn", "SN+W: SpaceNet 7 and flooded places added")}
MARGIN_C = 0.05


def part_a() -> dict:
    cms = {k: {} for k in ("W", *CANDS)}
    for c in C.kept():
        z = C.inputs(c)
        wat = R.water_after(z["after"], "unet")
        ref = R._mean_prob([C.RUNS / "runs" / f"terramind_{group(c)}_s{s}" / "prob.npz" for s in range(3)], c) > 0.5
        cms["W"][c] = C.confusion(ref & ~wat, z)
        for k, (m, _) in CANDS.items():
            p = R._mean_prob([R.RUNS / "runs" / f"{m}_{group(c)}_s{s}" / "prob.npz" for s in range(3)], c) > 0.5
            cms[k][c] = C.confusion(p & ~wat, z)
    out = {}
    for k, cm in cms.items():
        s = C.summary(cm)
        out[k] = {q: s[q] for q in ("iou", "precision", "recall", "per_city")}
        if k != "W":
            out[k]["minus_W"] = list(C.bootstrap_difference(cms["W"], cm))[:3]
            out[k]["passes"] = out[k]["minus_W"][1] > 0
    return out


def part_c() -> dict:
    from eval.metrics import cluster_bootstrap
    ref = np.load(R.RUNS / "ref_partc.npz")
    qs = [q for q in R.questions() if q["id"] != "conselice_construction"]
    fy = {k: {} for k in ("W", *CANDS)}
    for q in qs:
        f = R.PARTC / f"{q['id']}.npz"
        if not f.exists():
            for k in fy:
                fy[k][q["id"]] = 0.0
            continue
        z = np.load(f)
        u, ev = z["usable"], R.event(q)
        wat = R.water_after(z["after"], f"unet-without-{ev}")
        maps = {"W": ref[q["id"]].astype(np.float32) / 255 > 0.5,
                "SW": R._mean_prob([R.RUNS / "runs" / f"s_ALL_s{s}" / "prob.npz" for s in range(3)], q["id"]) > 0.5,
                "SNW": R._mean_prob([R.RUNS / "runs" / f"sn_ev_{ev}_s{s}" / "prob.npz" for s in range(3)], q["id"]) > 0.5}
        for k, m in maps.items():
            fy[k][q["id"]] = float((((m & ~wat) & u).sum() / u.sum()) >= R.MIN_FRACTION)
    cl = {q["id"]: q["cluster"] for q in qs}
    out = {"W": {"false_yes": int(sum(fy["W"].values()))}}
    for k in CANDS:
        b = cluster_bootstrap({i: fy[k][i] - fy["W"][i] for i in fy["W"]}, cl, n_boot=10000)
        d = float(np.mean(list(fy[k].values())) - np.mean(list(fy["W"].values())))
        out[k] = {"false_yes": int(sum(fy[k].values())), "minus_W": [d, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        out[k]["fails_margin"] = out[k]["minus_W"][2] > MARGIN_C
    return out


def main() -> dict:
    a, c = part_a(), part_c()
    ok = [k for k in CANDS if a[k]["passes"] and not c[k]["fails_margin"]]
    dec = {"passing": ok, "adopted": max(ok, key=lambda k: a[k]["iou"]) if ok else None}
    rep = {"part_a": a, "part_c": c, "decision": dec}
    (R.RUNS / "report_avenue2.json").write_text(json.dumps(rep, indent=1))
    print(f"{'detector':46s} {'OSCD IoU':>8s} {'P':>6s} {'R':>6s}  minus W [95 %]            false yes  minus W [95 %]")
    print(f"{'W (adopted)':46s} {a['W']['iou']:8.3f} {a['W']['precision']:6.3f} {a['W']['recall']:6.3f}  {'':26s} {c['W']['false_yes']:3d}/78")
    for k, (_, name) in CANDS.items():
        x, y = a[k]["minus_W"], c[k]["minus_W"]
        print(f"{name:46s} {a[k]['iou']:8.3f} {a[k]['precision']:6.3f} {a[k]['recall']:6.3f}  {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]  "
              f"{c[k]['false_yes']:3d}/78  {y[0]:+.3f} [{y[1]:+.3f}, {y[2]:+.3f}]")
    print("Decision:", json.dumps(dec))
    return rep


if __name__ == "__main__":
    main()
