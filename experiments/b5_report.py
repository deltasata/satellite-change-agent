"""B5 Part A (the table and the decision) and Part C (false alarms on unburnt land), protocol:
docs/METHODS.md section 8.

Part A. Networks: per held-out patch, the mean of the 3 seeds' probabilities; "burnt" where it is above 0.5.
Prithvi-BurnScars: its own probability (one model). Every detector is scored on the same pixels (labelled,
and at least 2 clear observations before and 1 after). A network replaces the rule if the 95 % interval of
(its pooled IoU - the rule's), from resampling the 42 fires, lies wholly above zero; among several, the
highest pooled IoU; if two lie within each other's interval, the cheaper one (parameters; within 5 % counts
as equal). The fitted rules are reported; a fitted rule is only proposed, never adopted, if no network
passes and it does.

Part C. Questions from the held-out patches, answered with the pipeline's decision (fraction of the scored
pixels marked >= 3 % -> "yes"): "no" questions = patches with no burnt pixel, >= 90 % labelled and >= 50 %
scored; "yes" questions = patches with >= 15 % of the scored pixels burnt. Measured: false "yes" on the
"no" questions, and "yes" on the "yes" questions; paired differences (winner minus rule), intervals from
resampling the fires. The winner is not adopted if the whole interval shows more false alarms.

    python -m experiments.b5_report
Writes runs/b5/report_A.json, report_C.json and cm_<detector>.json.
"""

from __future__ import annotations

import json

import numpy as np

from experiments import b5_common as C

NETWORKS = {"unet": "U-Net, from nothing (before + after)", "terramind": "TerraMind-1.0-base, Siamese, fine-tuned",
            "prithvi": "Prithvi-EO-2.0-300M-BurnScars, as published (after)"}
RULES = {"rule": "the rule (change ratio > 4)", "ratio_fitted": "change ratio, threshold fitted",
         "dnbr_fitted": "dNBR, threshold fitted"}
MIN_FRACTION = 0.03


def probs(model: str) -> tuple[dict, list, int]:
    """Mean probability per test patch, the single-run probabilities, and the parameter count."""
    runs = sorted((C.RUNS / "runs").glob(f"{model}_s*")) if model != "prithvi" else [C.RUNS / "runs" / "prithvi"]
    loaded = [np.load(r / "prob.npz") for r in runs]
    params = json.loads((runs[0] / "result.json").read_text())["parameters"]
    names = loaded[0].files
    mean = {n: np.mean([z[n].astype(np.float32) / 255 for z in loaded], axis=0) for n in names}
    single = [{n: z[n].astype(np.float32) / 255 for n in names} for z in loaded] if len(loaded) > 1 else []
    return mean, single, params


def masks() -> dict:
    """Each detector's map of "burnt" on every test patch (booleans), for Part C."""
    out = {}
    for n in C.built("test"):
        z = C.load(n)
        out[n] = z
    return out


def part_a() -> tuple[dict, dict]:
    rows, maps = {}, {}
    for k, name in RULES.items():
        rows[k] = {"name": name, "cm": C.read(k), "params": 0}
    for m, name in NETWORKS.items():
        mean, single, params = probs(m)
        cms = {}
        maps[m] = {}
        for n, p in mean.items():
            z = C.load(n)
            cms[n] = C.confusion(p > 0.5, z)
            maps[m][n] = p > 0.5
        C.save(m, cms)
        extra = {}
        if single:
            extra["single_seeds"] = [C.iou(np.sum([C.confusion(s[n] > 0.5, C.load(n)) for n in s], axis=0)) for s in single]
        rows[m] = {"name": name, "cm": cms, "params": params, **extra}
    rule = rows["rule"]["cm"]
    n_fires = len({C.fire(n) for n in rule})
    print(f"{len(rule)} held-out patches from {n_fires} fires\n")
    print(f"{'detector':52s} {'IoU':>6s} {'F1':>6s} {'prec.':>6s} {'recall':>6s} {'fire mean':>9s}   minus the rule [95 %]")
    out = {}
    for k, r in rows.items():
        s = C.summary(r["cm"])
        d = C.bootstrap_difference(rule, r["cm"]) if k != "rule" else None
        out[k] = {"name": r["name"], **{q: s[q] for q in ("iou", "f1", "precision", "recall", "mean_fire_iou")},
                  "per_fire": s["per_fire"], "minus_rule": d, "params": r["params"],
                  **({"single_seeds": r["single_seeds"]} if "single_seeds" in r else {})}
        txt = f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else ""
        print(f"{r['name']:52s} {s['iou']:6.3f} {s['f1']:6.3f} {s['precision']:6.3f} {s['recall']:6.3f} {s['mean_fire_iou']:9.3f}   {txt}")
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
        for f in ("ratio_fitted", "dnbr_fitted"):
            decision[f"winner_minus_{f}"] = C.bootstrap_difference(rows[f]["cm"], rows[best]["cm"])
    else:
        decision["winner"] = None
        decision["fitted_rules_passing"] = [f for f in ("ratio_fitted", "dnbr_fitted") if out[f]["minus_rule"][1] > 0]
    out["decision"] = decision
    out["n_patches"], out["n_fires"] = len(rule), n_fires
    print("\ndecision:", json.dumps(decision))
    (C.RUNS / "report_A.json").write_text(json.dumps(out, indent=1))
    return out, maps


def _boot(diff: dict, n: int = 10000, seed: int = 0):
    fires = sorted({C.fire(k) for k in diff})
    per = [[diff[k] for k in diff if C.fire(k) == f] for f in fires]
    rng = np.random.default_rng(seed)
    vals = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(fires), len(fires))
        v = [x for j in pick for x in per[j]]
        vals[i] = np.mean(v)
    return float(np.mean(list(diff.values()))), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def part_c(out_a: dict, maps: dict) -> dict:
    winner = out_a["decision"]["winner"]
    names = C.built("test")
    no_q, yes_q, ans = [], [], {k: {} for k in ("rule", "ratio_fitted", "dnbr_fitted", *NETWORKS)}
    choice = json.loads((C.RUNS / "rule_choice.json").read_text())
    for n in names:
        z = C.load(n)
        m = C.scored(z)
        if m.sum() == 0:
            continue
        lab = z["label"]
        burnt = (lab == 1) & m
        if (lab == 1).sum() == 0 and (lab >= 0).mean() >= 0.9 and m.mean() >= 0.5:
            no_q.append(n)
        elif burnt.sum() / m.sum() >= 0.15:
            yes_q.append(n)
        else:
            continue
        s, d = np.nan_to_num(z["score"], nan=0.0), C.dnbr(z)
        pred = {"rule": s > 4, "ratio_fitted": s > choice["ratio"]["threshold"], "dnbr_fitted": d > choice["dnbr"]["threshold"],
                **{k: maps[k][n] for k in NETWORKS}}
        for k, p in pred.items():
            ans[k][n] = bool((p & m).sum() / m.sum() >= MIN_FRACTION)
    res = {"n_no": len(no_q), "n_yes": len(yes_q), "n_fires_no": len({C.fire(n) for n in no_q}),
           "n_fires_yes": len({C.fire(n) for n in yes_q}), "by_detector": {}}
    for k, a in ans.items():
        res["by_detector"][k] = {"false_yes": float(np.mean([a[n] for n in no_q])), "hit": float(np.mean([a[n] for n in yes_q]))}
    if winner:
        res["winner"] = winner
        res["false_yes_diff"] = _boot({n: float(ans[winner][n]) - float(ans["rule"][n]) for n in no_q})
        res["hit_diff"] = _boot({n: float(ans[winner][n]) - float(ans["rule"][n]) for n in yes_q})
        res["adopt"] = not (res["false_yes_diff"][1] > 0)
    print(json.dumps(res, indent=1))
    (C.RUNS / "report_C.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    out_a, maps = part_a()
    part_c(out_a, maps)


if __name__ == "__main__":
    main()
