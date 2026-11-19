"""B4 Part A, the table and the decision (protocol: docs/METHODS.md section 7).

Networks: per city, the mean of the 3 seeds' probabilities from the runs that left the city's group out;
"changed" where the mean is above 0.5. Every detector is scored on the same pixels (the rule's usable
pixels). Decision: a detector replaces the rule if the 95 % interval of (its pooled IoU - the rule's),
from resampling the groups of cities, lies wholly above zero; among several, the highest pooled IoU; if two
lie within each other's interval, the cheaper one (parameters; within 5 % counts as equal).

    python -m experiments.b4_report
Writes runs/b4/report_A.json and cm_<network>.json.
"""

from __future__ import annotations

import json

import numpy as np

from core.data.oscd_pc import group
from experiments import b4_common as C

NETWORKS = {"fsd": "FC-Siam-diff, from nothing", "terramind": "TerraMind-1.0-base, Siamese, fine-tuned"}


def network_cms(model: str) -> tuple[dict, dict, int]:
    cms, seeds, params = {}, {}, 0
    for c in C.kept():
        probs = []
        for s in range(3):
            d = C.RUNS / "runs" / f"{model}_{group(c)}_s{s}"
            probs.append(np.load(d / "prob.npz")[c].astype(np.float32) / 255)
            params = json.loads((d / "result.json").read_text())["parameters"]
        z = C.inputs(c)
        cms[c] = C.confusion(np.mean(probs, axis=0) > 0.5, z)
        seeds[c] = [C.confusion(p > 0.5, z) for p in probs]
    single = [C.iou(np.sum([seeds[c][s] for c in seeds], axis=0)) for s in range(3)]
    return cms, {"single_seeds": single}, params


def main() -> None:
    rule, fitted = C.read("rule"), C.read("rule_fitted")
    rows = {"rule": {"name": "the rule (ratio > 4)", "cm": rule, "params": 0},
            "rule_fitted": {"name": "the rule, threshold fitted", "cm": fitted, "params": 0}}
    for m, name in NETWORKS.items():
        cms, extra, params = network_cms(m)
        C.save(m, cms)
        rows[m] = {"name": name, "cm": cms, "params": params, **extra}
    print(f"{len(rule)} cities in {len({group(c) for c in rule})} groups\n")
    print(f"{'detector':42s} {'IoU':>6s} {'F1':>6s} {'prec.':>6s} {'recall':>6s}   minus the rule [95 %]")
    out = {}
    for k, r in rows.items():
        s = C.summary(r["cm"])
        d = C.bootstrap_difference(rule, r["cm"]) if k != "rule" else None
        out[k] = {"name": r["name"], **{q: s[q] for q in ("iou", "f1", "precision", "recall")},
                  "per_city": s["per_city"], "minus_rule": d, "params": r["params"],
                  **({"single_seeds": r["single_seeds"]} if "single_seeds" in r else {})}
        txt = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
        print(f"{r['name']:42s} {s['iou']:6.3f} {s['f1']:6.3f} {s['precision']:6.3f} {s['recall']:6.3f}   {txt}")

    passing = [k for k in NETWORKS if out[k]["minus_rule"][1] > 0]
    decision = {"passing": passing}
    if passing:
        best = max(passing, key=lambda k: out[k]["iou"])
        for k in passing:
            if k == best:
                continue
            d = C.bootstrap_difference(rows[k]["cm"], rows[best]["cm"])
            decision[f"{best}_minus_{k}"] = d
            if d[1] <= 0 and out[k]["params"] < 0.95 * out[best]["params"]:
                best = k    # within each other's interval: the cheaper one
        decision["winner"] = best
        decision["winner_minus_rule_fitted"] = C.bootstrap_difference(fitted, rows[best]["cm"])
    else:
        decision["winner"] = None
    out["decision"] = decision
    print("\ndecision:", json.dumps(decision))
    print("\nper city IoU:")
    for c in rule:
        print(f"  {c:12s} " + "  ".join(f"{k} {out[k]['per_city'][c]:.3f}" for k in rows))
    (C.RUNS / "report_A.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
