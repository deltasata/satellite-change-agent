"""B7 Part A (the table and the decision) and Part C (false alarms on unburnt land), protocol:
docs/METHODS.md section 10.

Part A. Per held-out patch, the mean of the 3 seeds' probabilities; "burnt" where it is above 0.5; scored on
B5's pixels (labelled, at least 2 clear observations before and 1 after in the medians). U-TAE replaces B5's
U-Net if the 95 % interval of (U-TAE's pooled IoU - the U-Net's), resampling the 42 fires, lies wholly above
zero. The control (U-TAE on the two medians) and B5's other detectors are reported.

Part C. B5's "no" patches (no burnt pixel, >= 90 % labelled, >= 50 % scored) and "yes" patches (>= 15 % of
the scored pixels burnt), answered "yes" when at least 3 % of the scored pixels are marked. U-TAE is not
adopted if the whole interval of (its false "yes" - the U-Net's) lies above zero.

    python -m experiments.b7_report
Writes runs/b7/report_A.json, report_C.json and cm_<detector>.json.
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

RUNS = ROOT / "runs" / "b7"
NEW = {"utae": "U-TAE, every clear date", "utae2": "U-TAE on the two medians (control)"}
B5 = {"rule": "the rule (change ratio > 4)", "dnbr_fitted": "dNBR, threshold fitted",
      "unet": "U-Net, two medians (B5, adopted)", "terramind": "TerraMind, Siamese (B5)"}
MIN_FRACTION = 0.03


def mean_prob(model: str) -> tuple[dict, list, int]:
    runs = sorted((RUNS / "runs").glob(f"{model}_s*"))
    loaded = [np.load(r / "prob.npz") for r in runs]
    params = json.loads((runs[0] / "result.json").read_text())["parameters"]
    names = loaded[0].files
    mean = {n: np.mean([z[n].astype(np.float32) / 255 for z in loaded], axis=0) for n in names}
    single = [{n: z[n].astype(np.float32) / 255 for n in names} for z in loaded]
    return mean, single, params


def b5_map(model: str, name: str) -> np.ndarray:
    return np.mean([np.load(C.RUNS / "runs" / f"{model}_s{s}" / "prob.npz")[name].astype(np.float32) / 255
                    for s in range(3)], axis=0) > 0.5


def main() -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    rows, maps = {}, {}
    for k, name in B5.items():
        rows[k] = {"name": name, "cm": C.read(k)}
    loaded = {n: C.load(n) for n in C.built("test")}
    for m, name in NEW.items():
        mean, single, params = mean_prob(m)
        cms = {n: C.confusion(p > 0.5, loaded[n]) for n, p in mean.items()}
        maps[m] = {n: p > 0.5 for n, p in mean.items()}
        (RUNS / f"cm_{m}.json").write_text(json.dumps(cms))
        rows[m] = {"name": name, "cm": cms, "params": params,
                   "single_seeds": [C.iou(np.sum([C.confusion(s[n] > 0.5, loaded[n]) for n in s], axis=0)) for s in single]}
    ref = rows["unet"]["cm"]
    print(f"{len(ref)} held-out patches from {len({C.fire(n) for n in ref})} fires\n")
    print(f"{'detector':40s} {'IoU':>6s} {'F1':>6s} {'prec.':>6s} {'recall':>6s} {'fire mean':>9s}   minus B5's U-Net [95 %]")
    out = {}
    for k, r in rows.items():
        s = C.summary(r["cm"])
        d = C.bootstrap_difference(ref, r["cm"]) if k != "unet" else None
        out[k] = {"name": r["name"], **{q: s[q] for q in ("iou", "f1", "precision", "recall", "mean_fire_iou")},
                  "per_fire": s["per_fire"], "minus_unet": d,
                  **{q: r[q] for q in ("params", "single_seeds") if q in r}}
        txt = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
        print(f"{r['name']:40s} {s['iou']:6.3f} {s['f1']:6.3f} {s['precision']:6.3f} {s['recall']:6.3f} {s['mean_fire_iou']:9.3f}   {txt}")
    out["passes"] = out["utae"]["minus_unet"][1] > 0
    out["utae_minus_control"] = C.bootstrap_difference(rows["utae2"]["cm"], rows["utae"]["cm"])
    print(f"\nU-TAE passes Part A: {out['passes']}; U-TAE minus its control on medians: "
          f"{out['utae_minus_control'][0]:+.3f} [{out['utae_minus_control'][1]:+.3f}, {out['utae_minus_control'][2]:+.3f}]")
    (RUNS / "report_A.json").write_text(json.dumps(out, indent=1))

    # Part C
    no_q, yes_q, ans = [], [], {k: {} for k in ("unet", "utae", "utae2")}
    for n, z in loaded.items():
        m = C.scored(z)
        if m.sum() == 0:
            continue
        lab = z["label"]
        if (lab == 1).sum() == 0 and (lab >= 0).mean() >= 0.9 and m.mean() >= 0.5:
            no_q.append(n)
        elif ((lab == 1) & m).sum() / m.sum() >= 0.15:
            yes_q.append(n)
        else:
            continue
        pred = {"unet": b5_map("unet", n), "utae": maps["utae"][n], "utae2": maps["utae2"][n]}
        for k, p in pred.items():
            ans[k][n] = bool((p & m).sum() / m.sum() >= MIN_FRACTION)
    res = {"n_no": len(no_q), "n_yes": len(yes_q),
           "by_detector": {k: {"false_yes": float(np.mean([a[n] for n in no_q])), "hit": float(np.mean([a[n] for n in yes_q]))}
                           for k, a in ans.items()},
           "false_yes_diff": _boot({n: float(ans["utae"][n]) - float(ans["unet"][n]) for n in no_q}),
           "hit_diff": _boot({n: float(ans["utae"][n]) - float(ans["unet"][n]) for n in yes_q})}
    res["fail"] = res["false_yes_diff"][1] > 0
    print(json.dumps(res, indent=1))
    (RUNS / "report_C.json").write_text(json.dumps(res, indent=1))
    return out


if __name__ == "__main__":
    main()
