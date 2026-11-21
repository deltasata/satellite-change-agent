"""B8: the table, the decision, the false alarms, and the comparison with TerraMind-base-Fire (protocol:
docs/METHODS.md section 11).

Part A, the new fires of 2026 (decides). Per detector, the map of "burnt" (networks: mean of 3 seeds > 0.5;
TerraMind-base-Fire: its own map) on every new patch the B5 copy kept, scored on B5's pixels (labelled, at
least 2 clear observations before and 1 after in the medians). Pooled IoU; each candidate minus B5's U-Net,
95 % interval from resampling the fires 10 000 times. A candidate passes if the whole interval is above zero;
of those passing, the highest IoU, unless a candidate with fewer parameters passes and the best one's
advantage over it has an interval reaching zero (then the smaller one).
Part C, false alarms. New patches with no burnt pixel, at least 90 % labelled and at least half scored;
"yes" when at least 3 % of the scored pixels are marked. A candidate is not adopted if the whole interval of
(its false "yes" - the U-Net's) is above zero.
Reported only: every detector on the 42 held-out ImpactMesh fires, minus TerraMind-base-Fire's 0.864.

    python -m experiments.b8_report
Writes runs/b8/report.json and runs/b8/cm_<detector>_<set>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b5_common as C  # noqa: E402
from experiments.b5_report import _boot  # noqa: E402  (fire bootstrap of a per-patch difference)
from experiments.b8_predict import load_es, patches  # noqa: E402

RUNS = ROOT / "runs" / "b8"
NAMES = {"unet": "U-Net, two medians (B5, in use)", "utae": "U-TAE, every clear optical date (B7)",
         "utae_rd": "U-TAE with radar and elevation (B8)", "tmfire": "TerraMind-base-Fire, as published"}
MIN_FRACTION = 0.03


def tmfire_maps(which: str) -> dict:
    import rasterio
    out = {}
    for f in sorted((RUNS / f"tmfire_{which}").glob("*_prediction.tif")):
        with rasterio.open(f) as r:
            out[f.name.replace("_prediction.tif", "")] = r.read(1) == 1
    return out


def maps_new(model: str, names: list[str]) -> tuple[dict, list[dict]]:
    """Mean-of-seeds maps and the single seeds' maps on the new patches."""
    if model == "tmfire":
        m = tmfire_maps("cems2026")
        return {n: m[n] for n in names}, []
    runs = [np.load(RUNS / "cems2026" / f"{model}_s{s}.npz") for s in range(3)]
    single = [{n: z[n] > 127.5 for n in names} for z in runs]
    mean = {n: np.mean([z[n].astype(np.float32) / 255 for z in runs], axis=0) > 0.5 for n in names}
    return mean, single


def params(model: str) -> int:
    """Number of parameters, from each run's own record (TerraMind-base-Fire: counted in its checkpoint)."""
    f = {"unet": C.RUNS / "runs" / "unet_s0" / "result.json", "utae": ROOT / "runs" / "b7" / "runs" / "utae_s0" / "result.json",
         "utae_rd": RUNS / "runs" / "utae_rd_s0" / "result.json", "tmfire": RUNS / "tmfire_params.json"}[model]
    return json.loads(f.read_text())["parameters"]


def part_a(names: list[str], z: dict) -> dict:
    out, cms, maps = {}, {}, {}
    for m in NAMES:
        mean, single = maps_new(m, names)
        maps[m] = mean
        cms[m] = {n: C.confusion(mean[n], z[n]) for n in names}
        (RUNS / f"cm_{m}_cems2026.json").write_text(json.dumps(cms[m]))
        s = C.summary(cms[m])
        out[m] = {"name": NAMES[m], **{q: s[q] for q in ("iou", "f1", "precision", "recall", "mean_fire_iou")},
                  "per_fire": s["per_fire"],
                  "single_seeds": [C.iou(np.sum([C.confusion(x[n], z[n]) for n in names], axis=0)) for x in single],
                  "minus_unet": C.bootstrap_difference(cms["unet"], cms[m]) if m != "unet" else None}
    return out, cms, maps


def decide(a: dict, cms: dict, sizes: dict) -> dict:
    passing = [m for m in NAMES if m != "unet" and a[m]["minus_unet"][1] > 0]
    if not passing:
        return {"passing": [], "adopted": None}
    best = max(passing, key=lambda m: a[m]["iou"])
    choice, notes = best, {}
    for m in sorted(passing, key=lambda m: sizes[m]):
        if m == best or sizes[m] >= sizes[best]:
            continue
        d = C.bootstrap_difference(cms[m], cms[best])          # best minus the smaller one
        notes[m] = d
        if d[1] <= 0:
            choice = m
            break
    return {"passing": passing, "best": best, "smaller_vs_best": notes, "adopted": choice}


def part_c(names: list[str], z: dict, maps: dict) -> dict:
    no_q, yes_q, ans = [], [], {m: {} for m in NAMES}
    for n in names:
        s = C.scored(z[n])
        if s.sum() == 0:
            continue
        lab = z[n]["label"]
        if (lab == 1).sum() == 0 and (lab >= 0).mean() >= 0.9 and s.mean() >= 0.5:
            no_q.append(n)
        elif ((lab == 1) & s).sum() / s.sum() >= 0.15:
            yes_q.append(n)
        else:
            continue
        for m in NAMES:
            ans[m][n] = bool((maps[m][n] & s).sum() / s.sum() >= MIN_FRACTION)
    res = {"n_no": len(no_q), "n_yes": len(yes_q), "by_detector": {}}
    for m in NAMES:
        r = {"false_yes": float(np.mean([ans[m][n] for n in no_q])), "hit": float(np.mean([ans[m][n] for n in yes_q]))}
        if m != "unet":
            r["false_yes_diff"] = _boot({n: float(ans[m][n]) - float(ans["unet"][n]) for n in no_q})
            r["hit_diff"] = _boot({n: float(ans[m][n]) - float(ans["unet"][n]) for n in yes_q})
            r["fail"] = r["false_yes_diff"][1] > 0
        res["by_detector"][m] = r
    return res


def heldout() -> dict:
    """Reported only: the 42 held-out fires, every detector minus TerraMind-base-Fire with ImpactMesh's own files."""
    ref = C.read("tmfire")
    rows = {"tmfire_own_files": ref, "unet": C.read("unet"),
            "utae": json.loads((ROOT / "runs" / "b7" / "cm_utae.json").read_text()),
            "tmfire_our_inputs": json.loads((RUNS / "cm_tmfire_ourinputs_heldout.json").read_text())}
    loaded = {n: C.load(n) for n in ref}
    runs = [np.load(RUNS / "runs" / f"utae_rd_s{s}" / "prob_heldout.npz") for s in range(3)]
    rows["utae_rd"] = {n: C.confusion(np.mean([r[n].astype(np.float32) / 255 for r in runs], axis=0) > 0.5, loaded[n])
                       for n in ref}
    (RUNS / "cm_utae_rd_heldout.json").write_text(json.dumps(rows["utae_rd"]))
    out = {}
    for k, cm in rows.items():
        cm = {n: cm[n] for n in ref if n in cm}
        out[k] = {"iou": C.iou(np.sum(list(cm.values()), axis=0)), "n": len(cm),
                  "minus_tmfire": C.bootstrap_difference(ref, cm) if k != "tmfire_own_files" else None}
    return out


def main() -> dict:
    names = patches()
    z = {n: load_es(n) for n in names}
    a, cms, maps = part_a(names, z)
    sizes = {m: params(m) for m in NAMES}
    dec = decide(a, cms, sizes)
    c = part_c(names, z, maps)
    if dec["adopted"] and c["by_detector"][dec["adopted"]].get("fail"):
        dec["adopted_after_false_alarms"] = None
    else:
        dec["adopted_after_false_alarms"] = dec["adopted"]
    h = heldout()
    rep = {"n_patches": len(names), "n_fires": len({C.fire(n) for n in names}), "part_a": a, "parameters": sizes,
           "decision": dec, "part_c": c, "heldout_vs_tmfire": h}
    (RUNS / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"New fires of 2026: {rep['n_patches']} patches from {rep['n_fires']} fires\n")
    print(f"{'detector':42s} {'IoU':>6s} {'prec.':>6s} {'recall':>6s}  minus the U-Net [95 %]      false yes")
    for m in NAMES:
        r, d = a[m], a[m]["minus_unet"]
        txt = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else " " * 24
        print(f"{r['name']:42s} {r['iou']:6.3f} {r['precision']:6.3f} {r['recall']:6.3f}  {txt}  "
              f"{c['by_detector'][m]['false_yes']:.1%}")
    print(f"\nDecision: {json.dumps(dec)}")
    print(f"\n42 held-out ImpactMesh fires (reported only), minus TerraMind-base-Fire with its own files:")
    for k, r in h.items():
        d = r["minus_tmfire"]
        print(f"  {k:20s} {r['iou']:.3f}  " + (f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""))
    return rep


if __name__ == "__main__":
    main()
