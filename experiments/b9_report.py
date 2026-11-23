"""B9, avenue 1: the table and the decision (protocol: docs/METHODS.md section 12).

Part C (decides): the 78 flood questions asked as construction questions, scored from the images the
pipeline built (experiments/b9_flood_data.py). A question the pipeline could not answer (too few clear
pixels) is "cannot tell" for every detector. Otherwise "yes" if the changed pixels are at least 3 % of the
usable ones. False "yes" fraction over the 78; each candidate minus the reference, 95 % interval from
resampling the 11 flood events 10 000 times; a candidate passes if the whole interval is below zero.
Part A (condition): B4's 24 OSCD cities and scored pixels; IoU minus the reference, 22 groups resampled;
not adopted if the interval's lower end is below -0.03.

Detectors: "ref" B4's TerraMind (its ALL runs on Part C, which never saw a flood; its leave-one-group-out runs
on Part A); "W" ref minus the water of the after image (B1's water U-Net; for a flood question the one
trained without that event); "N" B9's runs with flooded places as "no change"; "NW" N minus the water.

    .venv-ml/bin/python -m experiments.b9_report ref      # B4's networks on the recorded Part C images (GPU)
    .venv-ml/bin/python -m experiments.b9_report          # the table and the decision
Writes runs/b9/ref_partc.npz, runs/b9/report_avenue1.json.
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

RUNS = ROOT / "runs" / "b9"
PARTC = ROOT / "data_cache" / "b9" / "partc"
DETECTORS = {"ref": "B4's TerraMind (in use)", "W": "W: water of the after image removed",
             "N": "N: flooded places as 'no change' in training", "NW": "N+W"}
SIMPLER = ["W", "N", "NW"]
MIN_FRACTION, MARGIN_A = 0.03, -0.03


def questions() -> list[dict]:
    return json.loads((ROOT / "data_cache" / "b9" / "partc.json").read_text())


def event(q: dict) -> str | None:
    return q["cluster"].removeprefix("flood_") if q["cluster"].startswith("flood_") else None


def ref_partc() -> None:
    import torch
    from experiments import b4_train as T
    dev = torch.device("cuda")
    names = [f.stem for f in sorted(PARTC.glob("*.npz"))]
    probs = {n: [] for n in names}
    for s in range(3):
        net = T.Net("terramind")
        st = torch.load(C.RUNS / "runs" / f"terramind_ALL_s{s}" / "weights.pt", map_location="cpu")
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in st.items()})
        net.to(dev).eval()
        for n in names:
            z = np.load(PARTC / f"{n}.npz")
            probs[n].append(T.predict(net, {"before": z["before"], "after": z["after"]}, dev))
    np.savez_compressed(RUNS / "ref_partc.npz", **{n: np.round(np.mean(p, 0) * 255).astype(np.uint8) for n, p in probs.items()})


def _mean_prob(files: list[Path], name: str) -> np.ndarray:
    return np.mean([np.load(f)[name].astype(np.float32) / 255 for f in files], axis=0)


def water_after(after: np.ndarray, detector: str) -> np.ndarray:
    from core.models.water import water_mask
    return water_mask(after, detector)


def part_c() -> dict:
    ref = np.load(RUNS / "ref_partc.npz")
    rows = {d: {} for d in DETECTORS}
    detail = []
    for q in questions():
        qid, ev = q["id"], event(q)
        f = PARTC / f"{qid}.npz"
        if not f.exists():
            for d in DETECTORS:
                rows[d][qid] = ("cannot_tell", None)
            continue
        z = np.load(f)
        u = z["usable"]
        group_ = f"ev_{ev}" if ev else "ALL"
        maps = {"ref": ref[qid].astype(np.float32) / 255 > 0.5,
                "N": _mean_prob([RUNS / "runs" / f"n_{group_}_s{s}" / "prob.npz" for s in range(3)], qid) > 0.5}
        wat = water_after(z["after"], f"unet-without-{ev}" if ev else "unet")
        maps["W"], maps["NW"] = maps["ref"] & ~wat, maps["N"] & ~wat
        for d, m in maps.items():
            cf = float((m & u).sum() / u.sum())
            rows[d][qid] = ("yes" if cf >= MIN_FRACTION else "no", round(cf, 4))
        detail.append({"id": qid, "water_after_fraction": round(float((wat & u).sum() / u.sum()), 4),
                       **{d: rows[d][qid][1] for d in DETECTORS}})
    from eval.metrics import cluster_bootstrap
    ids = [q["id"] for q in questions() if q["id"] != "conselice_construction"]
    cl = {q["id"]: q["cluster"] for q in questions()}
    out = {"n": len(ids), "n_events": len({cl[i] for i in ids}), "by_detector": {}, "per_question": detail,
           "conselice": {d: rows[d]["conselice_construction"] for d in DETECTORS}}
    fy = {d: {i: float(rows[d][i][0] == "yes") for i in ids} for d in DETECTORS}
    for d in DETECTORS:
        r = {"false_yes": int(sum(fy[d].values())), "false_yes_fraction": float(np.mean(list(fy[d].values()))),
             "cannot_tell": sum(rows[d][i][0] == "cannot_tell" for i in ids),
             "per_event": {e: int(sum(fy[d][i] for i in ids if cl[i] == e)) for e in sorted({cl[i] for i in ids})}}
        if d != "ref":
            boot = cluster_bootstrap({i: fy[d][i] - fy["ref"][i] for i in ids}, cl, n_boot=10000)
            r["minus_ref"] = [r["false_yes_fraction"] - out["by_detector"]["ref"]["false_yes_fraction"],
                              float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            r["passes"] = r["minus_ref"][2] < 0
        out["by_detector"][d] = r
    out["_fy"] = fy
    out["_cl"] = cl
    return out


def part_a() -> dict:
    cms = {d: {} for d in DETECTORS}
    for c in C.kept():
        z = C.inputs(c)
        ref = _mean_prob([C.RUNS / "runs" / f"terramind_{group(c)}_s{s}" / "prob.npz" for s in range(3)], c) > 0.5
        n = _mean_prob([RUNS / "runs" / f"n_{group(c)}_s{s}" / "prob.npz" for s in range(3)], c) > 0.5
        wat = water_after(z["after"], "unet")
        for d, m in {"ref": ref, "N": n, "W": ref & ~wat, "NW": n & ~wat}.items():
            cms[d][c] = C.confusion(m, z)
    stored = C.read("terramind")
    out = {"ref_equals_b4": all(cms["ref"][c] == stored[c] for c in stored), "by_detector": {}}
    for d in DETECTORS:
        s = C.summary(cms[d])
        r = {k: s[k] for k in ("iou", "f1", "precision", "recall", "per_city")}
        if d != "ref":
            r["minus_ref"] = list(C.bootstrap_difference(cms["ref"], cms[d]))[:3]
            r["fails_margin"] = r["minus_ref"][1] < MARGIN_A
        out["by_detector"][d] = r
    out["_cms"] = cms
    return out


def decide(c: dict, a: dict) -> dict:
    from eval.metrics import cluster_bootstrap
    ok = [d for d in SIMPLER if c["by_detector"][d]["passes"] and not a["by_detector"][d]["fails_margin"]]
    if not ok:
        return {"passing": [], "adopted": None}
    best = min(ok, key=lambda d: (c["by_detector"][d]["false_yes"], SIMPLER.index(d)))
    choice, notes = best, {}
    for d in SIMPLER:
        if d == best or d not in ok or SIMPLER.index(d) > SIMPLER.index(best):
            continue
        boot = cluster_bootstrap({i: c["_fy"][best][i] - c["_fy"][d][i] for i in c["_fy"][d]}, c["_cl"], n_boot=10000)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        notes[d] = [lo, hi]
        if hi >= 0:                    # the best one's advantage over the simpler one reaches zero
            choice = d
            break
    return {"passing": ok, "best": best, "best_minus_simpler": notes, "adopted": choice}


def main() -> dict:
    c, a = part_c(), part_a()
    dec = decide(c, a)
    rep = {"part_c": {k: v for k, v in c.items() if not k.startswith("_")},
           "part_a": {k: v for k, v in a.items() if not k.startswith("_")}, "decision": dec}
    (RUNS / "report_avenue1.json").write_text(json.dumps(rep, indent=1))
    print(f"Part C: {c['n']} flood questions asked as construction ({c['n_events']} events)")
    print(f"{'detector':48s} {'false yes':>9s}  minus ref [95 %]           OSCD IoU  minus ref [95 %]")
    for d, name in DETECTORS.items():
        r, s = c["by_detector"][d], a["by_detector"][d]
        t1 = f"{r['minus_ref'][0]:+.3f} [{r['minus_ref'][1]:+.3f}, {r['minus_ref'][2]:+.3f}]" if d != "ref" else " " * 24
        t2 = f"{s['minus_ref'][0]:+.3f} [{s['minus_ref'][1]:+.3f}, {s['minus_ref'][2]:+.3f}]" if d != "ref" else ""
        print(f"{name:48s} {r['false_yes']:4d}/{c['n']}  {t1}   {s['iou']:.3f}   {t2}")
    print(f"\nConselice: {c['conselice']}")
    print(f"B4's Part A reproduced: {a['ref_equals_b4']}")
    print(f"Decision: {json.dumps(dec)}")
    return rep


if __name__ == "__main__":
    if sys.argv[1:] == ["ref"]:
        ref_partc()
    else:
        main()
