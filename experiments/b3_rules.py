"""B3, the rules and the stored masks (protocol: docs/METHODS.md section 6).

  scl          the pipeline's rule: unusable where SCL is 3, 8, 9 or 10.        test: es and gee
  scl_best     the stronger SCL rule: SCL 2 and / or 7 added, the unusable area widened by k = 0, 1, 2, 3
               or 5 pixels; the combination with the highest pooled IoU on the validation part (gee).
  published masks stored in the data set (reported only, test: gee): unetmobv2_v2 and sensei_v2 give the
               4 classes; s2cloudless, cloudscore_cs_v1 and cloudscore_cs_cdf_v1 give a number per pixel,
               turned into usable / unusable by the threshold (and direction) that is best on the
               validation part (Cloud Score+ in 256 levels, see `level`).

    python -m experiments.b3_rules         # writes runs/b3/cm_<kind>_<name>.json and runs/b3/rule_choice.json
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

from experiments import b3_common as C

VARIANT_CLASSES = {"base": (3, 8, 9, 10), "+2": (2, 3, 8, 9, 10), "+7": (3, 7, 8, 9, 10), "+2+7": (2, 3, 7, 8, 9, 10)}
WIDEN = (0, 1, 2, 3, 5)
CLASS_MASKS = ("unetmobv2_v2", "sensei_v2")
VALUE_MASKS = ("s2cloudless", "cloudscore_cs_v1", "cloudscore_cs_cdf_v1")


def level(name: str, a: np.ndarray) -> np.ndarray:
    """A stored value -> one of 256 levels: s2cloudless (0-100) as it is; Cloud Score+ (stored up to about
    65 000) in steps of 256, which is fine enough for a reported-only comparison."""
    a = a.astype(np.int64)
    return np.clip(a // 256 if name.startswith("cloudscore") else a, 0, 255)


def _validation_variants(i: str) -> dict:
    y, m = C.scored(i, "validation")
    s = C.scl(i, "validation", "gee")
    return {f"{c}|{k}": C.confusion(C.scl_classes(s, VARIANT_CLASSES[c], k), y, m)
            for c, k in product(VARIANT_CLASSES, WIDEN)}


def _validation_values(i: str) -> dict:
    """Per value mask: histogram of its values over scored pixels, separately for usable and unusable label."""
    y, m = C.scored(i, "validation")
    d = C.load(i, "validation")
    out = {}
    for name in VALUE_MASKS:
        v = level(name, d[name][m])
        bad = y[m] >= 1
        out[name] = (np.bincount(v[bad], minlength=256), np.bincount(v[~bad], minlength=256))
    return out


def best_threshold(hist_bad: np.ndarray, hist_good: np.ndarray) -> tuple[str, int, float]:
    """The direction and threshold t with the highest pooled IoU of "unusable": either unusable where
    value >= t, or unusable where value < t. Every value that occurs is tried."""
    cb, cg = np.cumsum(hist_bad), np.cumsum(hist_good)      # counts with value <= v
    nb, ng = cb[-1], cg[-1]
    best = ("high", 0, -1.0)
    for t in np.flatnonzero((hist_bad + hist_good) > 0):
        below_b, below_g = (cb[t - 1] if t else 0), (cg[t - 1] if t else 0)
        for direction, tp, fp in (("high", nb - below_b, ng - below_g), ("low", below_b, below_g)):
            fn = nb - tp
            iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
            if iou > best[2]:
                best = (direction, int(t), float(iou))
    return best


def _test(i: str, choice: dict) -> dict:
    y, m = C.scored(i, "test")
    d = C.load(i, "test")
    out = {}
    for kind in ("es", "gee"):
        s = C.scl(i, "test", kind)
        out[("scl", kind)] = C.confusion(C.scl_classes(s), y, m)
        out[("scl_best", kind)] = C.confusion(
            C.scl_classes(s, VARIANT_CLASSES[choice["scl_best"]["classes"]], choice["scl_best"]["widen"]), y, m)
    for name in CLASS_MASKS:
        out[(name, "gee")] = C.confusion(np.clip(d[name].astype(np.int64), 0, 3), y, m)
    for name in VALUE_MASKS:
        direction, t = choice[name]["direction"], choice[name]["t"]
        v = level(name, d[name])
        bad = v >= t if direction == "high" else v < t
        out[(name, "gee")] = C.confusion(bad.astype(np.int64), y, m)
    return out


def main() -> None:
    val, test = C.ids("validation"), C.ids("test")
    print(f"validation {len(val)} images, test {len(test)} images (with an Earth Search copy)")
    with ProcessPoolExecutor(32) as pool:
        vv = list(pool.map(_validation_variants, val, chunksize=8))
        vh = list(pool.map(_validation_values, val, chunksize=8))
    scores = {key: C.iou_unusable(np.sum([v[key] for v in vv], axis=0)) for key in vv[0]}
    best = max(scores, key=scores.get)
    cls, k = best.split("|")
    choice = {"validation_iou": scores, "scl_best": {"classes": cls, "widen": int(k), "validation_iou": scores[best]}}
    for name in VALUE_MASKS:
        hb = np.sum([h[name][0] for h in vh], axis=0)
        hg = np.sum([h[name][1] for h in vh], axis=0)
        direction, t, iou = best_threshold(hb, hg)
        choice[name] = {"direction": direction, "t": t, "validation_iou": iou}
    print("stronger SCL rule chosen on validation:", choice["scl_best"])
    print("base rule on validation:", round(scores["base|0"], 4))
    C.RUNS.mkdir(parents=True, exist_ok=True)
    (C.RUNS / "rule_choice.json").write_text(json.dumps(choice, indent=1))
    with ProcessPoolExecutor(32) as pool:
        res = list(pool.map(_test, test, [choice] * len(test), chunksize=8))
    for key in res[0]:
        name, kind = key
        C.save(name, kind, {i: r[key] for i, r in zip(test, res)})
    print("written:", sorted({f"{k[1]}_{k[0]}" for k in res[0]}))


if __name__ == "__main__":
    main()
