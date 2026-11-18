"""B2, the radar rules: water where backscatter is low (smooth water reflects the radar pulse away).

  vh_fit   VH < t, t fitted on the training events (Part A: leaving one event out; Part B: official
           training part). t maximises the pooled water IoU of the training tiles, grid -30..-10 dB, 0.1 dB.
  vv_fit   the same with VV.
  otsu_vh  Otsu's threshold per tile on VH: the t that best splits the tile's own histogram into two
           groups (largest variance between the groups). Nothing is fitted; it is Bonafilia et al.'s baseline.

    python -m experiments.b2_rules           # writes runs/b2/counts_<kind>_<rule>.json and thresholds_<kind>.json
"""

from __future__ import annotations

import json

import numpy as np

from core.data import sen1floods11 as s1f
from experiments import b2_common as B

EDGES = np.round(np.arange(-60.0, 10.0001, 0.1), 1)       # histogram bin edges in dB
GRID = np.round(np.arange(-30.0, -9.9999, 0.1), 1)          # thresholds tried


def otsu(v: np.ndarray) -> float:
    h, e = np.histogram(v, bins=256)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def tile_data(kind: str):
    """Per tile: histograms of VV and VH for water and dry scored pixels, and the Otsu counts."""
    out = {}
    for c in s1f.chips():
        r = B.radar(c, kind)
        if r is None:
            continue
        y = B.label(c)
        m = np.isfinite(r).all(axis=0) & (y >= 0)
        d = {}
        for k, band in enumerate(("vv", "vh")):
            d[band] = [np.histogram(r[k][m & (y == cls)], EDGES)[0] for cls in (1, 0)]
        vh = r[1][np.isfinite(r[1])]
        t = otsu(vh) if vh.size > 100 else -20.0
        d["otsu_t"] = t
        d["otsu"] = B.counts(r[1] < t, y, m)
        out[c] = d
    return out


def counts_at(d: dict, band: str, t: float) -> list[int]:
    """[TP, FP, FN, TN] of 'band < t' from the histograms (t lies on a bin edge)."""
    k = int(round((t - EDGES[0]) / 0.1))
    hw, hd = d[band]
    tp, fn = int(hw[:k].sum()), int(hw[k:].sum())
    fp, tn = int(hd[:k].sum()), int(hd[k:].sum())
    return [tp, fp, fn, tn]


def fit(data: dict, chips: list[str], band: str) -> float:
    scores = [B.iou(np.array([counts_at(data[c], band, t) for c in chips])) for t in GRID]
    return float(GRID[int(np.nanargmax(scores))])


def main() -> None:
    B.RUNS.mkdir(parents=True, exist_ok=True)
    # Part A: our radar copy, leave one event out
    data = tile_data("rtc")
    events = sorted({s1f.event(c) for c in data})
    thresholds = {}
    for band in ("vv", "vh"):
        per = {}
        for e in events:
            t = fit(data, [c for c in data if s1f.event(c) != e], band)
            thresholds[f"{band}_without_{e}"] = t
            per.update({c: counts_at(data[c], band, t) for c in data if s1f.event(c) == e})
        thresholds[f"{band}_ALL"] = fit(data, list(data), band)
        (B.RUNS / f"counts_rtc_{band}_fit.json").write_text(json.dumps(per))
    (B.RUNS / "counts_rtc_otsu_vh.json").write_text(json.dumps({c: data[c]["otsu"] for c in data}))
    thresholds["otsu_per_tile"] = {c: data[c]["otsu_t"] for c in data}
    (B.RUNS / "thresholds_rtc.json").write_text(json.dumps(thresholds, indent=1))
    print("Part A thresholds:", {k: v for k, v in thresholds.items() if k != "otsu_per_tile"})
    # Part B: the data set's own radar, official training part
    data = tile_data("grd")
    train = [c for c in B.official_parts()["train"] if c in data]
    tb = {}
    for band in ("vv", "vh"):
        t = fit(data, train, band)
        tb[band] = t
        (B.RUNS / f"counts_grd_{band}_fit.json").write_text(json.dumps({c: counts_at(data[c], band, t) for c in data}))
    (B.RUNS / "counts_grd_otsu_vh.json").write_text(json.dumps({c: data[c]["otsu"] for c in data}))
    (B.RUNS / "thresholds_grd.json").write_text(json.dumps(tb, indent=1))
    print("Part B thresholds (official training part):", tb)


if __name__ == "__main__":
    main()
