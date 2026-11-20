"""B6: today's yes/no thresholds against fitted thresholds and a logistic regression (protocol:
docs/METHODS.md section 9).

Input: runs/b6/exam.json (experiments/b6_exam.py), today's path on all 150 questions with every measured
number. Only questions whose measurement reached the yes/no step are refitted; every other question keeps
its answer ("cannot tell") in every arm.

Kinds of measurement: "flood_optical" and "flood_radar" (new water = water fraction after - before),
"construction" and "vegetation_loss" (changed or burnt fraction).

Candidates, each scored by leaving one of the exam's 83 groups out (fitted on the other 82, answering the
left-out group):
  thresholds   per kind, from THRESHOLDS, the most right answers on the training questions; ties: the
               value closest to today's (ratio nearest 1), then the larger
  logistic     per kind its own intercept and weights; floods: new water, water before, water after;
               construction and vegetation loss: log10(fraction + 0.001); columns scaled on the training
               questions; L2 penalty 1; "yes" where the probability is above 0.5

    python -m experiments.b6_fit
Writes runs/b6/report.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs" / "b6"
TODAY = {"flood_optical": 0.05, "flood_radar": 0.05, "construction": 0.03, "vegetation_loss": 0.03}
THRESHOLDS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20)
KINDS = tuple(TODAY)


# ---------- the measured questions ----------

def measured(r: dict) -> dict | None:
    """The numbers the yes/no step used, or None if the question never reached it."""
    m = r.get("measurement") or {}
    if r["answer"] not in ("yes", "no") or not r.get("sensor"):
        return None
    if r["kind"] == "flood":
        wb, wa = m["water_fraction_before"], m["water_fraction_after"]
        return dict(kind=f"flood_{r['sensor']}", value=wa - wb, water_before=wb, water_after=wa)
    return dict(kind=r["kind"], value=m["changed_fraction"])


def load() -> list[dict]:
    rows = json.loads((RUNS / "exam.json").read_text())
    for r in rows:
        r["m"] = measured(r)
        if r["m"]:      # today's rule reproduces the stored answer
            assert (r["m"]["value"] >= TODAY[r["m"]["kind"]]) == (r["answer"] == "yes"), r["id"]
    return rows


# ---------- candidate 1: thresholds ----------

def fit_thresholds(train: list[dict]) -> dict:
    out = {}
    for k in KINDS:
        qs = [r for r in train if r["m"] and r["m"]["kind"] == k]
        if not qs:
            out[k] = TODAY[k]
            continue

        def right(t):
            return sum((r["m"]["value"] >= t) == (r["expected"] == "yes") for r in qs)
        best = max(right(t) for t in THRESHOLDS)
        tied = [t for t in THRESHOLDS if right(t) == best]
        out[k] = min(tied, key=lambda t: (abs(np.log(t / TODAY[k])), -t))
    return out


def answer_thresholds(r: dict, th: dict) -> str:
    return ("yes" if r["m"]["value"] >= th[r["m"]["kind"]] else "no") if r["m"] else r["answer"]


# ---------- candidate 2: logistic regression ----------

def features(m: dict) -> np.ndarray:
    if m["kind"].startswith("flood"):
        return np.array([m["value"], m["water_before"], m["water_after"]], float)
    return np.array([np.log10(max(m["value"], 0.0) + 0.001)], float)


def logistic(X: np.ndarray, y: np.ndarray, l2: float = 1.0, steps: int = 100) -> np.ndarray:
    """Weights (intercept first) maximising the log-likelihood minus l2/2 |w|^2 (intercept not penalised),
    by Newton's method. X: (n, d) scaled columns; y: 0 or 1."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    pen = np.full(A.shape[1], l2)
    pen[0] = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (y - p) - pen * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(pen) + 1e-9 * np.eye(len(w))
        dw = np.linalg.solve(H, g)
        w += dw
        if np.abs(dw).max() < 1e-10:
            break
    return w


def fit_logistic(train: list[dict]) -> dict:
    out = {}
    for k in KINDS:
        qs = [r for r in train if r["m"] and r["m"]["kind"] == k]
        if not qs:
            out[k] = None
            continue
        X = np.stack([features(r["m"]) for r in qs])
        y = np.array([r["expected"] == "yes" for r in qs], float)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        out[k] = dict(mu=mu, sd=sd, w=logistic((X - mu) / sd, y))
    return out


def answer_logistic(r: dict, model: dict) -> str:
    if not r["m"]:
        return r["answer"]
    p = model[r["m"]["kind"]]
    if p is None:       # no training question of this kind: today's threshold
        return answer_thresholds(r, TODAY)
    z = p["w"][0] + ((features(r["m"]) - p["mu"]) / p["sd"]) @ p["w"][1:]
    return "yes" if z > 0 else "no"


# ---------- leave one group out ----------

def cross_validated(rows: list[dict], fit, answer) -> dict:
    out = {}
    for g in sorted({r["cluster"] for r in rows}):
        model = fit([r for r in rows if r["cluster"] != g])
        for r in rows:
            if r["cluster"] == g:
                out[r["id"]] = answer(r, model)
    return out


def compare(rows, a: dict, b: dict) -> dict:
    """b minus a: accuracy difference, 95 % interval over the exam's groups."""
    from eval.metrics import cluster_bootstrap
    cl = {r["id"]: r["cluster"] for r in rows}
    ra = {r["id"]: float(a[r["id"]] == r["expected"]) for r in rows}
    rb = {r["id"]: float(b[r["id"]] == r["expected"]) for r in rows}
    boot = cluster_bootstrap({i: rb[i] - ra[i] for i in ra}, cl, n_boot=10000)
    return dict(diff=float(np.mean(list(rb.values())) - np.mean(list(ra.values()))),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)))


def per_kind(rows, ans: dict) -> dict:
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["m"]["kind"] if r["m"] else "not measured"
        out[k][0] += ans[r["id"]] == r["expected"]
        out[k][1] += 1
    return {k: f"{v[0]}/{v[1]}" for k, v in sorted(out.items())}


# ---------- false-alarm checks (B4 and B5 Part C) ----------

def b4_part_c() -> list[dict]:
    """The 78 flood questions asked as construction questions (expected "no"), TerraMind's fractions."""
    rows = json.loads((ROOT / "runs" / "b4" / "exam_terramind_C.json").read_text())
    return [dict(id=r["id"], cluster=r["cluster"], expected="no", answer=r["answer"],
                 m=dict(kind="construction", value=r["changed_fraction"]) if r["answer"] in ("yes", "no") else None)
            for r in rows]


def b5_part_c() -> list[dict]:
    """The unburnt held-out ImpactMesh patches (B5 Part C "no" questions), the U-Net's burnt fractions."""
    cache = RUNS / "b5_part_c.json"
    if not cache.exists():
        from experiments import b5_common as C
        probs = [np.load(C.RUNS / "runs" / f"unet_s{s}" / "prob.npz") for s in range(3)]
        out = []
        for n in C.built("test"):
            z = C.load(n)
            m = C.scored(z)
            lab = z["label"]
            if m.sum() == 0 or not ((lab == 1).sum() == 0 and (lab >= 0).mean() >= 0.9 and m.mean() >= 0.5):
                continue
            p = np.mean([q[n].astype(np.float32) / 255 for q in probs], axis=0) > 0.5
            out.append(dict(id=n, cluster=C.fire(n), value=float((p & m).sum() / m.sum())))
        cache.write_text(json.dumps(out))
    rows = json.loads(cache.read_text())
    return [dict(id=r["id"], cluster=r["cluster"], expected="no", answer="yes" if r["value"] >= 0.03 else "no",
                 m=dict(kind="vegetation_loss", value=r["value"])) for r in rows]


def false_alarms(rows, answer, model) -> dict:
    today = {r["id"]: answer_thresholds(r, TODAY) for r in rows}
    new = {r["id"]: answer(r, model) for r in rows}
    d = compare(rows, new, today)       # (today right) - (new right) = new false "yes" - today's
    return dict(n=len(rows), false_yes_today=sum(v == "yes" for v in today.values()),
                false_yes_new=sum(v == "yes" for v in new.values()), more_false_yes=d,
                fail=d["lo"] > 0)


# ---------- the report ----------

def main() -> dict:
    rows = load()
    today = {r["id"]: r["answer"] for r in rows}
    cands = {"thresholds": (fit_thresholds, answer_thresholds), "logistic": (fit_logistic, answer_logistic)}
    acc = lambda ans: float(np.mean([ans[r["id"]] == r["expected"] for r in rows]))
    out = {"n": len(rows), "n_groups": len({r["cluster"] for r in rows}),
           "n_measured": {k: sum(1 for r in rows if r["m"] and r["m"]["kind"] == k) for k in KINDS},
           "accuracy_today": acc(today), "per_kind_today": per_kind(rows, today), "candidates": {}}
    print(f"{out['n']} questions in {out['n_groups']} groups; measured: {out['n_measured']}")
    print(f"today's thresholds: accuracy {out['accuracy_today']:.3f}  {out['per_kind_today']}")
    for name, (fit, answer) in cands.items():
        cv = cross_validated(rows, fit, answer)
        c = dict(accuracy=acc(cv), minus_today=compare(rows, today, cv), per_kind=per_kind(rows, cv),
                 changed=[dict(id=r["id"], expected=r["expected"], today=today[r["id"]], new=cv[r["id"]],
                               value=r["m"]["value"]) for r in rows if cv[r["id"]] != today[r["id"]]])
        c["passes"] = c["minus_today"]["lo"] > 0
        full = fit(rows)
        c["fitted_on_all"] = ({k: v for k, v in full.items()} if name == "thresholds" else
                              {k: (None if v is None else {q: np.round(v[q], 4).tolist() for q in v}) for k, v in full.items()})
        if name == "thresholds":
            c["fold_values"] = {k: sorted({fit([r for r in rows if r["cluster"] != g])[k]
                                           for g in {r["cluster"] for r in rows}}) for k in KINDS}
        if c["passes"]:
            c["false_alarms"] = {"construction (B4 Part C)": false_alarms(b4_part_c(), answer, full),
                                 "vegetation loss (B5 Part C)": false_alarms(b5_part_c(), answer, full)}
        d = c["minus_today"]
        print(f"{name:10s}: accuracy {c['accuracy']:.3f}, minus today {d['diff']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
              f"  {c['per_kind']}  passes: {c['passes']}")
        out["candidates"][name] = c
    passing = [k for k, c in out["candidates"].items() if c["passes"]
               and not any(f["fail"] for f in c.get("false_alarms", {}).values())]
    if len(passing) == 2:
        d = compare(rows, *[cross_validated(rows, *cands[k]) for k in ("thresholds", "logistic")])
        out["logistic_minus_thresholds"] = d
        passing = ["logistic"] if d["lo"] > 0 else ["thresholds"]
    out["adopted"] = passing[0] if passing else None
    print("adopted:", out["adopted"])
    (RUNS / "report.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
