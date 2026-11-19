"""B4 Part A, the rules (protocol: docs/METHODS.md section 7, detectors 1 and 2).

1. The rule as in the pipeline: "changed" where the ratio map (change of the median over the before
   images' date-to-date scatter, root mean square over 5 bands, 3 x 3 median filter) is above 4.
2. The rule with a fitted threshold: for each left-out group, the threshold from {2, 3, 4, 5, 6, 8} with
   the highest pooled IoU on the cities of the other groups.

    python -m experiments.b4_rules
Writes runs/b4/cm_rule.json, cm_rule_fitted.json and rule_choice.json.
"""

from __future__ import annotations

import json

import numpy as np

from core.data.oscd_pc import group
from experiments import b4_common as C


def main() -> None:
    cities = C.kept()
    z = {c: C.inputs(c) for c in cities}
    cm = {t: {c: C.confusion(np.nan_to_num(z[c]["score"], nan=0.0) > t, z[c]) for c in cities}
          for t in C.RULE_THRESHOLDS}
    C.save("rule", cm[4.0])
    fitted, choice = {}, {}
    for g in sorted({group(c) for c in cities}):
        train = [c for c in cities if group(c) != g]
        scores = {t: C.iou(np.sum([cm[t][c] for c in train], axis=0)) for t in C.RULE_THRESHOLDS}
        best = max(scores, key=scores.get)
        choice[g] = {"threshold": best, "train_iou": scores}
        for c in cities:
            if group(c) == g:
                fitted[c] = cm[best][c]
    C.save("rule_fitted", fitted)
    (C.RUNS / "rule_choice.json").write_text(json.dumps(choice, indent=1))
    for name, d in (("rule (ratio > 4)", cm[4.0]), ("rule, fitted threshold", fitted)):
        s = C.summary(d)
        print(f"{name:24s} IoU {s['iou']:.3f}  F1 {s['f1']:.3f}  precision {s['precision']:.3f}  recall {s['recall']:.3f}")
    print("thresholds chosen:", sorted({v['threshold'] for v in choice.values()}))


if __name__ == "__main__":
    main()
