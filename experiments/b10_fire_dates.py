"""B10: which images a fire question uses (protocol: docs/METHODS.md section 13).

Questions with known answers:
- new fires of 2026 (B8's CEMS patches): at most 3 "yes" (burnt fraction >= 0.10) and 3 "no" (burnt 0) patches
  per fire, drawn with seed 0; box 2.56 km on the patch centre; asked about the fire's day;
- the exam's 13 fires (answer "yes"), asked about the month (or summer) the exam names.
Candidates: A (today's live route), T (the burn network's training periods), S (the 8 nearest useful images).

    .venv/bin/python -m experiments.b10_fire_dates questions
    .venv/bin/python -m experiments.b10_fire_dates run A        # then T, S
    .venv/bin/python -m experiments.b10_fire_dates report
Writes runs/b10/questions.json, runs/b10/answers_<candidate>.json, runs/b10/report.json.
"""

from __future__ import annotations

import calendar
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs" / "b10"
PATCHES = ROOT / "data_cache" / "burn" / "cems" / "patches.json"
SELECTION = {"A": "fixed", "T": "training", "S": "nearest8"}
PATCH_KM = 2.56
# The exam's fires, asked as the exam words them: a month, or summer (June to August).
EXAM_SPANS = {"fire_lahaina": (2023, 8), "fire_lytton": (2021, "summer"), "fire_paradise": (2018, 11),
              "fire_greenville_dixie": (2021, 7), "fire_jasper": (2024, 7), "fire_evia": (2021, 8),
              "fire_pedrogao": (2017, 6), "fire_kangaroo_island": (2020, 1), "fire_palisades": (2025, 1),
              "fire_eaton_altadena": (2025, 1), "fire_mati": (2018, 7), "fire_coffey_park": (2017, 10),
              "fire_valparaiso": (2024, 2)}


def make_questions() -> list[dict]:
    rng = np.random.default_rng(0)
    patches = json.loads(PATCHES.read_text())
    qs = []
    for fire in sorted({p["fire"] for p in patches}):
        ps = [p for p in patches if p["fire"] == fire]
        for label, pick in (("yes", [p for p in ps if p["burnt"] >= 0.10]), ("no", [p for p in ps if p["burnt"] == 0])):
            pick = sorted(pick, key=lambda p: p["name"])
            for k in sorted(rng.permutation(len(pick))[:3]):
                p = pick[k]
                qs.append(dict(id=p["name"], cluster=fire, expected=label, form="day", lat=p["lat"], lon=p["lon"],
                               size_km=PATCH_KM, d1=p["date"], d2=p["date"], d0=p["date"], burnt=p["burnt"]))
    for line in (ROOT / "eval" / "benchmark_v2.jsonl").open():
        q = json.loads(line)
        if q["id"] not in EXAM_SPANS:
            continue
        y, m = EXAM_SPANS[q["id"]]
        if m == "summer":
            d1, d2, d0 = date(y, 6, 1), date(y, 8, 31), date(y, 7, 16)
        else:
            d1, d2, d0 = date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1]), date(y, m, 15)
        r = q["reference"]
        qs.append(dict(id=q["id"], cluster=q["id"], expected="yes", form="month", lat=r["lat"], lon=r["lon"],
                       size_km=r["size_km"], d1=d1.isoformat(), d2=d2.isoformat(), d0=d0.isoformat(), burnt=None))
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "questions.json").write_text(json.dumps(qs, indent=1))
    n = {k: sum(q["expected"] == k for q in qs) for k in ("yes", "no")}
    print(f"{len(qs)} questions ({n['yes']} yes, {n['no']} no) from {len({q['cluster'] for q in qs})} fires")
    return qs


def answer(q: dict, cand: str) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    from core import tools
    from core.measure import measure_change
    tools.render_evidence = lambda *a, **k: type("E", (), {"path": None})()
    t0 = time.time()
    when = (dict(event_date=q["d0"]) if cand == "A" else
            dict(event_date=q["d1"], event_end=q["d2"]))
    try:
        r = measure_change(lat=q["lat"], lon=q["lon"], kind="vegetation_loss", size_km=q["size_km"],
                           selection=SELECTION[cand], **when)
        out = dict(answer=r.answer, reason=r.reason, changed_fraction=(r.measurement or {}).get("changed_fraction"),
                   usable=(r.measurement or {}).get("fraction_pixels_usable"), n_before=r.n_images_before,
                   n_after=r.n_images_after, before=r.before_period, after=r.after_period, retried=r.retried,
                   caution=r.caution is not None, steps=r.steps)
    except Exception as e:                      # recorded; counts as "cannot tell"
        out = dict(answer="cannot_tell", error=f"{type(e).__name__}: {str(e)[:300]}")
    return dict(id=q["id"], cluster=q["cluster"], expected=q["expected"], form=q["form"],
                seconds=round(time.time() - t0, 1), **out)


def run(cand: str, workers: int = 6) -> None:
    qs = json.loads((RUNS / "questions.json").read_text())
    dest = RUNS / f"answers_{cand}.json"
    done = {r["id"]: r for r in json.loads(dest.read_text())} if dest.exists() else {}
    todo = [q for q in qs if q["id"] not in done or "error" in done[q["id"]]]
    print(f"{cand}: {len(todo)} questions to answer", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(answer, todo, [cand] * len(todo))):
            done[r["id"]] = r
            if k % 10 == 9 or k == len(todo) - 1:      # saved as it goes: a stopped run continues where it was
                dest.write_text(json.dumps([done[q["id"]] for q in qs if q["id"] in done], indent=1))
                print(f"  {k + 1}/{len(todo)}", flush=True)
    res = [done[q["id"]] for q in qs]
    print(f"{cand}: {sum(r['answer'] == r['expected'] for r in res)} of {len(res)} right; "
          f"{sum(r['answer'] == 'cannot_tell' for r in res)} cannot tell; {sum('error' in r for r in res)} errors")


def report() -> dict:
    from eval.metrics import cluster_bootstrap
    qs = json.loads((RUNS / "questions.json").read_text())
    ans = {c: {r["id"]: r for r in json.loads((RUNS / f"answers_{c}.json").read_text())} for c in SELECTION}
    ids = [q["id"] for q in qs]
    cl = {q["id"]: q["cluster"] for q in qs}
    exp = {q["id"]: q["expected"] for q in qs}
    no_ids = [i for i in ids if exp[i] == "no"]
    right = {c: {i: float(ans[c][i]["answer"] == exp[i]) for i in ids} for c in SELECTION}
    fyes = {c: {i: float(ans[c][i]["answer"] == "yes") for i in no_ids} for c in SELECTION}

    def diff(a: dict, b: dict, keys: list[str]) -> list[float]:
        boot = cluster_bootstrap({i: b[i] - a[i] for i in keys}, {i: cl[i] for i in keys}, n_boot=10000)
        return [float(np.mean([b[i] - a[i] for i in keys])), float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))]

    out = {"n": len(ids), "n_yes": len(ids) - len(no_ids), "n_no": len(no_ids), "n_groups": len(set(cl.values()))}
    for c in SELECTION:
        rs = [ans[c][i] for i in ids]
        out[c] = {"right": float(np.mean(list(right[c].values()))),
                  "cannot_tell": float(np.mean([r["answer"] == "cannot_tell" for r in rs])),
                  "wrong": float(np.mean([r["answer"] not in ("cannot_tell", exp[r["id"]]) for r in rs])),
                  "false_yes": float(np.mean(list(fyes[c].values()))),
                  "seconds_median": float(np.median([r["seconds"] for r in rs])),
                  "by_form": {f: float(np.mean([right[c][i] for i in ids if ans[c][i]["form"] == f]))
                              for f in ("day", "month")}}
        if c != "A":
            out[c]["right_minus_A"] = diff(right["A"], right[c], ids)
            out[c]["false_yes_minus_A"] = diff(fyes["A"], fyes[c], no_ids)
            out[c]["passes"] = out[c]["right_minus_A"][1] > 0 and out[c]["false_yes_minus_A"][2] <= 0.03
    out["T_minus_S"] = diff(right["S"], right["T"], ids)
    passing = [c for c in ("T", "S") if out[c]["passes"]]
    out["adopted"] = ("S" if "S" in passing and not ("T" in passing and out["T_minus_S"][1] > 0) else
                      "T" if "T" in passing else "A")
    (RUNS / "report.json").write_text(json.dumps(out, indent=1))
    print(f"{out['n']} questions ({out['n_yes']} yes, {out['n_no']} no), {out['n_groups']} fires")
    for c in SELECTION:
        o = out[c]
        extra = (f"  right - A {o['right_minus_A'][0]:+.3f} [{o['right_minus_A'][1]:+.3f}, {o['right_minus_A'][2]:+.3f}]"
                 f"  false yes - A {o['false_yes_minus_A'][0]:+.3f} [{o['false_yes_minus_A'][1]:+.3f}, "
                 f"{o['false_yes_minus_A'][2]:+.3f}]  passes {o['passes']}" if c != "A" else "")
        print(f"{c} ({SELECTION[c]:8s}) right {o['right']:.3f}  cannot tell {o['cannot_tell']:.3f}  wrong {o['wrong']:.3f}"
              f"  false yes {o['false_yes']:.3f}  day {o['by_form']['day']:.3f} month {o['by_form']['month']:.3f}"
              f"  {o['seconds_median']:.0f} s{extra}")
    print("T - S:", [round(v, 3) for v in out["T_minus_S"]], "-> adopted:", out["adopted"])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "questions":
        make_questions()
    elif cmd == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
    elif cmd == "report":
        report()
