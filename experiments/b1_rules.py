"""B1, the rules: NDWI > 0 (the pipeline's rule), and NDWI > t, MNDWI > t with t fitted on the other events.

For each tile, the scored pixels' index values are put into 400 bins from -1 to 1, separately for
water and dry pixels. A threshold t then gives the counts by summing bins, so fitting t on 9 or 10
events is a sum, not a new pass over the images.

    python -m experiments.b1_rules           # both image kinds; writes runs/b1/counts_<kind>_<rule>.json
"""

from __future__ import annotations

import json

import numpy as np

from core.data import sen1floods11 as s1f
from experiments import b1_common as C

EDGES = np.linspace(-1, 1, 401)
GRID = EDGES[100:300]          # thresholds tried: -0.5 .. +0.5 in steps of 0.005


def nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def index(x, name):
    blue, green, red, nir, nir08, swir16, swir22 = x
    return nd(green, nir) if name == "ndwi" else nd(green, swir16)


def histograms(kind: str):
    """chip -> {index: (water histogram, dry histogram)}"""
    out = {}
    for c in s1f.chips():
        s = C.scored(c, kind)
        if s is None:
            continue
        x, y, m = s
        out[c] = {}
        for name in ("ndwi", "mndwi"):
            v = np.clip(np.nan_to_num(index(x, name), nan=-1.0), -1, 1 - 1e-9)
            out[c][name] = (np.histogram(v[m & (y == 1)], EDGES)[0], np.histogram(v[m & (y == 0)], EDGES)[0])
    return out


def counts_at(hw, hd, t):
    k = np.searchsorted(EDGES, t, side="right") - 1   # bins from k upwards count as water (v >= EDGES[k])
    tp, fn = hw[k:].sum(), hw[:k].sum()
    fp, tn = hd[k:].sum(), hd[:k].sum()
    return [int(tp), int(fp), int(fn), int(tn)]


def fit_threshold(H, chips, name):
    hw = np.sum([H[c][name][0] for c in chips], axis=0)
    hd = np.sum([H[c][name][1] for c in chips], axis=0)
    return max(GRID, key=lambda t: C.iou(counts_at(hw, hd, t)))


def run(kind: str) -> dict:
    H = histograms(kind)
    chips = sorted(H)
    events = sorted({s1f.event(c) for c in chips})
    res = {"ndwi0": {c: counts_at(*H[c]["ndwi"], 0.0) for c in chips}}
    fitted = {}
    for name in ("ndwi", "mndwi"):
        res[f"{name}_fit"] = {}
        for e in events:
            train = [c for c in chips if s1f.event(c) != e]
            t = fit_threshold(H, train, name)
            fitted.setdefault(name, {})[e] = round(float(t), 3)
            for c in chips:
                if s1f.event(c) == e:
                    res[f"{name}_fit"][c] = counts_at(*H[c][name], t)
    for rule, per_chip in res.items():
        C.save_counts(rule, kind, per_chip)
    (C.RUNS / f"thresholds_{kind}.json").write_text(json.dumps(fitted, indent=1))
    return res


if __name__ == "__main__":
    for kind in ("l1c", "l2a"):
        res = run(kind)
        chips = sorted(res["ndwi0"])
        print(f"{kind}: {len(chips)} tiles, {len({s1f.event(c) for c in chips})} events")
        for rule, pc in res.items():
            print(f"   {rule:10s} pooled IoU {C.pooled_iou(pc, chips):.3f}")
