"""B3 tables (protocol: docs/METHODS.md section 6).

    python -m experiments.b3_report A        # Part A: the cloud detectors on the test part (decision)

Decision (Part A, on the Earth Search copy, "es"): a detector replaces the SCL rule if the 95 % interval
of (detector minus rule) of the pooled "unusable" IoU lies wholly above zero (resampling whole places,
10 000 times). Among several, the highest pooled IoU; if the next one lies within the interval of their
difference (the interval includes zero), the cheaper one. Checked once more without the 16 test images
taken on the same day and tile as OmniCloudMask training images.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from experiments import b3_common as C

NETWORKS = ("unet18", "terramind")
CANDIDATES = ("ocm", "unet18", "terramind")   # detectors 3-5
# Cost = parameters run per image, measured: OmniCloudMask 14.4 M (two networks, 7.7 M + 6.8 M, both run),
# U-Net 14.3 M, TerraMind-base with decoder about 100 M. Costs within 5 % count as equal.
COST = {"ocm": 14.4, "unet18": 14.3, "terramind": 100.0}
PUBLISHED = ("unetmobv2_v2", "sensei_v2", "s2cloudless", "cloudscore_cs_v1", "cloudscore_cs_cdf_v1")


def network_cms(model: str, kind: str, test: list[str]) -> tuple[dict, list[float]]:
    """Ensemble of the 3 seeds (mean probability, most probable class) and the single seeds' IoU."""
    probs = [np.load(C.RUNS / "runs" / f"{model}_s{s}" / f"prob_{kind}.npz") for s in range(3)]
    ens, single = {}, [[] for _ in probs]
    for i in test:
        y, m = C.scored(i, "test")
        ps = [p[i].astype(np.float32) for p in probs]
        ens[i] = C.confusion(np.mean(ps, axis=0).argmax(0), y, m)
        for s, p in enumerate(ps):
            single[s].append(C.confusion(p.argmax(0), y, m))
    return ens, [C.iou_unusable(np.sum(c, axis=0)) for c in single]


def part_a() -> dict:
    test = C.ids("test")
    cms = {(n, k): C.read(n, k) for n in ("scl", "scl_best", "ocm") for k in ("es", "gee")}
    singles = {}
    for model in NETWORKS:
        for kind in ("es", "gee"):
            cms[(model, kind)], singles[(model, kind)] = network_cms(model, kind, test)
            C.save(f"{model}_ensemble", kind, cms[(model, kind)])
    for n in PUBLISHED:
        cms[(n, "gee")] = C.read(n, "gee")
    out = {"n_test_images": len(test), "rule_choice": json.loads((C.RUNS / "rule_choice.json").read_text())["scl_best"]}
    for (name, kind), cm in cms.items():
        r = C.summary(cm, test)
        if (name, kind) in singles:
            r["single_seeds"] = singles[(name, kind)]
        if name != "scl":
            d, lo, hi, n_places = C.bootstrap_difference(cms[("scl", kind)], cm, test)
            r.update(minus_scl=d, lo=lo, hi=hi, n_places=n_places)
        out[f"{name}|{kind}"] = r
    # decision on es
    winners = [c for c in CANDIDATES if out[f"{c}|es"]["lo"] > 0]
    decision = {"passed": winners}
    if winners:
        best = max(winners, key=lambda c: out[f"{c}|es"]["iou"])
        chosen = best
        for other in winners:
            if other == best:
                continue
            d, lo, hi, _ = C.bootstrap_difference(cms[(other, "es")], cms[(best, "es")], test)
            decision[f"{best}_minus_{other}"] = {"diff": d, "lo": lo, "hi": hi}
            if lo <= 0 and COST[other] < 0.95 * COST[chosen]:
                chosen = other
        decision["highest"] = best
        decision["adopted"] = chosen
        d, lo, hi, _ = C.bootstrap_difference(cms[("scl_best", "es")], cms[(chosen, "es")], test)
        decision["adopted_minus_scl_best"] = {"diff": d, "lo": lo, "hi": hi}
    else:
        decision["adopted"] = "scl (rule stays)"
    same_day = C.same_day_ids()
    rest = [i for i in test if i not in same_day]
    decision["without_same_day_images"] = {
        c: dict(zip(("diff", "lo", "hi", "n_places"), C.bootstrap_difference(cms[("scl", "es")], cms[(c, "es")], rest)))
        for c in CANDIDATES}
    decision["n_same_day_images_in_test"] = len(set(test) & same_day)
    out["decision"] = decision
    (C.RUNS / "report_A.json").write_text(json.dumps(out, indent=1))
    return out


def print_a(out: dict) -> None:
    f = lambda v: "" if v is None else f"{v:.3f}"
    print(f"Part A: {out['n_test_images']} test images. Stronger SCL rule chosen on validation: {out['rule_choice']}")
    for kind in ("es", "gee"):
        print(f"\n[{kind}] {'detector':16s} {'IoU':>6s} {'minus SCL [95 % interval]':>28s} {'let thr.':>8s} {'thrown':>7s} "
              f"{'thick':>6s} {'thin':>6s} {'shadow':>6s}  single seeds")
        for key, r in out.items():
            if not key.endswith(f"|{kind}"):
                continue
            name = key.split("|")[0]
            diff = "" if name == "scl" else f"{r['minus_scl']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
            seeds = ", ".join(f"{v:.3f}" for v in r.get("single_seeds", []))
            print(f"     {name:16s} {r['iou']:6.3f} {diff:>28s} {r['clouds_let_through']:8.3f} "
                  f"{r['good_pixels_thrown_away']:7.3f} {f(r['iou_thick']):>6s} {f(r['iou_thin']):>6s} "
                  f"{f(r['iou_shadow']):>6s}  {seeds}")
    print("\ndecision:", json.dumps(out["decision"], indent=1))


if __name__ == "__main__":
    if sys.argv[1:] == ["A"]:
        print_a(part_a())
