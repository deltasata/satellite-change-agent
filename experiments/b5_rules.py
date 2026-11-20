"""B5 Part A, the rules (protocol: docs/METHODS.md section 8, detectors 1 and 2).

1. The rule as in the pipeline: "changed" where the change rule's ratio map is above 4.
2. Fitted rules, thresholds chosen on the training fires' scored pixels (pooled IoU of "burnt"):
   the change ratio > t, t from {2, 3, 4, 5, 6, 8}; and dNBR > t, t from {0.05, 0.10, 0.15, 0.20, 0.27,
   0.35, 0.44}, dNBR = NBR(before median) - NBR(after median), NBR = (B08 - B12) / (B08 + B12).

    python -m experiments.b5_rules
Writes runs/b5/cm_rule.json, cm_ratio_fitted.json, cm_dnbr_fitted.json and rule_choice.json.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments import b5_common as C


def _cms(name: str) -> dict:
    z = C.load(name)
    s, d = np.nan_to_num(z["score"], nan=0.0), C.dnbr(z)
    return {"ratio": {t: C.confusion(s > t, z) for t in C.RATIO_THRESHOLDS},
            "dnbr": {t: C.confusion(d > t, z) for t in C.DNBR_THRESHOLDS}}


def all_cms(names: list[str]) -> dict:
    with ProcessPoolExecutor(32) as pool:
        return dict(zip(names, pool.map(_cms, names, chunksize=64)))


def main() -> None:
    train, test = all_cms(C.built("train")), all_cms(C.built("test"))
    choice = {}
    for kind, ts in (("ratio", C.RATIO_THRESHOLDS), ("dnbr", C.DNBR_THRESHOLDS)):
        scores = {t: C.iou(np.sum([v[kind][t] for v in train.values()], axis=0)) for t in ts}
        choice[kind] = {"threshold": max(scores, key=scores.get), "train_iou": scores}
    C.save("rule", {n: v["ratio"][4.0] for n, v in test.items()})
    C.save("ratio_fitted", {n: v["ratio"][choice["ratio"]["threshold"]] for n, v in test.items()})
    C.save("dnbr_fitted", {n: v["dnbr"][choice["dnbr"]["threshold"]] for n, v in test.items()})
    choice["n_train_patches"], choice["n_test_patches"] = len(train), len(test)
    (C.RUNS / "rule_choice.json").write_text(json.dumps(choice, indent=1))
    print(json.dumps(choice, indent=1))
    for name in ("rule", "ratio_fitted", "dnbr_fitted"):
        s = C.summary(C.read(name))
        print(f"{name:14s} IoU {s['iou']:.3f}  F1 {s['f1']:.3f}  precision {s['precision']:.3f}  recall {s['recall']:.3f}")


if __name__ == "__main__":
    main()
