"""B2 tables (protocol: docs/METHODS.md section 4).

    python -m experiments.b2_report A          # Part A: radar water detectors, leave one event out (decision)
    python -m experiments.b2_report B-select   # Part B: validation scores of the candidates; writes the choice
    python -m experiments.b2_report B-final    # Part B: validation checks of ensemble and flips, then the test part
    python -m experiments.b2_report C          # Part C: the exam
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import sen1floods11 as s1f  # noqa: E402
from experiments import b2_common as B  # noqa: E402

RUNS = B.RUNS / "runs"


def run_dir(model, inp, kind, fold, seed, loss="ce", lr=None) -> Path:
    from experiments.b2_train import run_name
    return RUNS / run_name(model, inp, kind, fold, seed, loss, lr)


def counts_from_prob(probs: list[dict], chips, inp: str, kind: str) -> dict:
    """Counts of the mean probability of several runs (an ensemble), water where > 0.5."""
    out = {}
    for c in chips:
        p = np.mean([pr[c].astype(np.float32) / 255 for pr in probs], axis=0)
        _, y, m = B.scored(c, inp, kind)
        out[c] = B.counts(p > 0.5, y, m)
    return out


def part_a() -> None:
    rules = {name: json.loads((B.RUNS / f"counts_rtc_{name}.json").read_text())
             for name in ("vh_fit", "vv_fit", "otsu_vh")}
    chips = sorted(rules["vh_fit"])
    events = sorted({s1f.event(c) for c in chips})
    rows = dict(rules)
    seeds_iou = {}
    for model in ("unet18", "terramind"):
        probs = {s: {} for s in range(3)}
        for e in events:
            for s in range(3):
                d = run_dir(model, "s1", "rtc", e, s)
                if (d / "result.json").exists():
                    probs[s].update(dict(np.load(d / "prob.npz")))
        if not all(len(probs[s]) == len(chips) for s in range(3)):
            print(f"{model}: runs not finished ({[len(probs[s]) for s in range(3)]} of {len(chips)} tiles)")
            continue
        rows[model] = counts_from_prob([probs[s] for s in range(3)], chips, "s1", "rtc")
        seeds_iou[model] = [B.pooled_iou(counts_from_prob([probs[s]], chips, "s1", "rtc"), chips) for s in range(3)]
    ref = rows["vh_fit"]
    print(f"Part A: radar copy (Planetary Computer RTC), leave one event out, {len(chips)} tiles, {len(events)} events")
    print(f"{'detector':12s} {'water IoU':>9s}   minus VH fitted [95 % interval]   single seeds")
    report = {}
    for name, cnt in rows.items():
        v = B.pooled_iou(cnt, chips)
        line = f"{name:12s} {v:9.3f}"
        if name != "vh_fit":
            d, lo, hi, n = B.bootstrap_difference(ref, cnt, chips)
            line += f"   {d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            report[name] = dict(iou=v, diff=d, lo=lo, hi=hi)
        else:
            report[name] = dict(iou=v)
        if name in seeds_iou:
            line += "   " + ", ".join(f"{x:.3f}" for x in seeds_iou[name])
            report[name]["seeds"] = seeds_iou[name]
        print(line)
    if "unet18" in rows and "terramind" in rows:
        d, lo, hi, _ = B.bootstrap_difference(rows["unet18"], rows["terramind"], chips)
        print(f"terramind minus unet18: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        report["terramind_minus_unet18"] = dict(diff=d, lo=lo, hi=hi)
    print("\nper event (water IoU):")
    print(f"{'event':10s} " + " ".join(f"{n:>10s}" for n in rows))
    for e in events:
        ce = [c for c in chips if s1f.event(c) == e]
        print(f"{e:10s} " + " ".join(f"{B.pooled_iou(rows[n], ce):10.3f}" for n in rows))
        report.setdefault("per_event", {})[e] = {n: B.pooled_iou(rows[n], ce) for n in rows}
    (B.RUNS / "report_A.json").write_text(json.dumps(report, indent=1))


def b_candidates() -> list[dict]:
    from experiments.b2_train import plan_b_select
    return plan_b_select()


def part_b_select() -> None:
    """Validation water IoU of every candidate (seed 0, no flips); the best per family and input set."""
    val = B.official_parts()["valid"]
    print("Part B, validation part (89 tiles), water IoU / mIoU, seed 0:")
    best = {}
    for j in b_candidates():
        d = run_dir(j["model"], j["inp"], "grd", "official", 0, j.get("loss", "ce"), j.get("lr"))
        if not (d / "result.json").exists():
            print(f"  {d.name}: not finished")
            continue
        cnt = json.loads((d / "result.json").read_text())["counts"]
        v = B.pooled_iou(cnt, val)
        mi = B.miou(np.array([cnt[c] for c in val if c in cnt]))
        print(f"  {d.name:45s} {v:.3f} / {mi:.3f}")
        fam = ("unet" if j["model"].startswith("unet") else "terramind", j["inp"])
        if fam not in best or v > best[fam][0]:
            best[fam] = (v, j)
    for inp in ("s1", "s2", "s1s2"):
        tiles = [c for c in val]
        for name in ("vh_fit", "otsu_vh"):
            if inp == "s1":
                cnt = json.loads((B.RUNS / f"counts_grd_{name}.json").read_text())
                print(f"  rule {name:40s} {B.pooled_iou(cnt, tiles):.3f} / {B.miou(np.array([cnt[c] for c in tiles])):.3f}")
    choice = {f"{f[0]}_{f[1]}": v[1] for f, v in best.items()}
    (B.RUNS / "partB_choice.json").write_text(json.dumps(choice, indent=1))
    print("chosen:", json.dumps(choice))


PUBLISHED = {  # official test part; water IoU, mIoU (docs/METHODS.md section 4)
    "s1": ("TerraMind-1.0-base, TerraMind paper Table 8", 0.680, 0.811),
    "s2": ("TerraMind-1.0-base, TerraMind paper Table 8", 0.823, 0.897),
    "s1s2": ("U-Net, PANGAEA Table 5 (mIoU only)", None, 0.914),
}


def part_b_final() -> None:
    """On validation: one seed, 5 seeds, 5 seeds with 8 views; the best is used on the test part."""
    choice = json.loads((B.RUNS / "partB_choice.json").read_text())
    parts = B.official_parts()
    result = {}
    for key, j in choice.items():
        dirs = [run_dir(j["model"], j["inp"], "grd", "official", s, j.get("loss", "ce"), j.get("lr")) for s in range(5)]
        p1 = [dict(np.load(d / "prob.npz")) for d in dirs]
        p8 = [dict(np.load(d / "prob_views8.npz")) for d in dirs]
        ways = {"seed 0": [p1[0]], "5 seeds": p1, "5 seeds, 8 views": p8}
        val = {w: counts_from_prob(p, parts["valid"], j["inp"], "grd") for w, p in ways.items()}
        vi = {w: B.pooled_iou(c, parts["valid"]) for w, c in val.items()}
        best = max(vi, key=vi.get)
        test = counts_from_prob(ways[best], parts["test"], j["inp"], "grd")
        bol = counts_from_prob(ways[best], parts["bolivia"], j["inp"], "grd")
        single = [B.pooled_iou(counts_from_prob([p], parts["test"], j["inp"], "grd"), parts["test"]) for p in p1]
        tarr = np.array([test[c] for c in parts["test"]])
        result[key] = dict(config=j, valid=vi, used=best, test_iou=B.iou(tarr), test_miou=B.miou(tarr),
                           test_single_seeds=single, bolivia_iou=B.pooled_iou(bol, parts["bolivia"]),
                           bolivia_miou=B.miou(np.array([bol[c] for c in parts["bolivia"]])))
    for name in ("vh_fit", "otsu_vh"):
        cnt = json.loads((B.RUNS / f"counts_grd_{name}.json").read_text())
        t = np.array([cnt[c] for c in parts["test"]])
        per_tile = np.nanmean([B.iou(cnt[c]) for c in parts["test"]])
        result[f"rule_{name}"] = dict(test_iou=B.iou(t), test_miou=B.miou(t), test_mean_over_tiles=float(per_tile))
    print("Part B: the data set's own images, official split. Validation choice, then the test part (90 tiles).")
    for k, r in result.items():
        if "config" in r:
            print(f"  {k:16s} valid {', '.join(f'{w} {v:.3f}' for w, v in r['valid'].items())} -> {r['used']}")
    print(f"\n{'':16s} {'test water IoU':>15s} {'test mIoU':>10s} {'Bolivia IoU':>12s}   single seeds (test)")
    for k, r in result.items():
        extra = f"{r['bolivia_iou']:12.3f}   " + ", ".join(f"{x:.3f}" for x in r["test_single_seeds"]) if "config" in r else ""
        print(f"  {k:16s} {r['test_iou']:13.3f} {r['test_miou']:10.3f} {extra}")
    print("\npublished, same test part:")
    for inp, (who, w, m) in PUBLISHED.items():
        print(f"  {inp:5s} {who:46s} water IoU {w if w is None else f'{w:.3f}'}  mIoU {m:.3f}")
    (B.RUNS / "report_B.json").write_text(json.dumps(result, indent=1))


def accuracy_difference(rows, a: str, b: str, n: int = 10000, seed: int = 0):
    """Accuracy of b minus accuracy of a, 95 % interval from resampling whole events."""
    events = sorted({r["event"] for r in rows})
    per = {e: np.array([[r[a] == r["key"], r[b] == r["key"]] for r in rows if r["event"] == e], float) for e in events}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = np.concatenate([per[e] for e in rng.choice(events, len(events))])
        diffs.append(s[:, 1].mean() - s[:, 0].mean())
    allr = np.concatenate(list(per.values()))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(allr[:, 1].mean() - allr[:, 0].mean()), float(lo), float(hi)


def part_c(model: str = "terramind") -> None:
    optical = {r["id"]: r for r in json.loads((ROOT / "runs" / "b1" / "exam_unet.json").read_text())}
    radar = {r["id"]: r for r in json.loads((B.RUNS / f"exam_radar_{model}.json").read_text())}
    rows = []
    for qid, o in optical.items():
        r = radar[qid]
        row = dict(id=qid, event=o["event"], expected=o["expected"], optical=o["model"],
                   radar_rule=r["rule"], radar_net=r["net"])
        row["combined"] = row["optical"] if row["optical"] != "cannot_tell" else row["radar_net"]
        row["combined_rule"] = row["optical"] if row["optical"] != "cannot_tell" else row["radar_rule"]
        # India: written as "cannot tell" (no optical image); scored against the hand label here
        row["key"] = ("yes" if "_yes" in qid else "no") if row["event"] == "India" else row["expected"]
        rows.append(row)
    main = [r for r in rows if r["event"] != "India"]
    india = [r for r in rows if r["event"] == "India"]
    cols = ("optical", "combined", "combined_rule", "radar_net", "radar_rule")
    print(f"Part C: exam v2 flood questions. Optical = B1's U-Net; radar network = {model} (Part A winner).")
    print(f"{'answer from':16s} {'72 outside India':>17s} {'India (6, hand label)':>22s} {'cannot tell (72)':>17s}")
    out = {}
    for c in cols:
        acc = np.mean([r[c] == r["key"] for r in main])
        ai = np.mean([r[c] == r["key"] for r in india])
        ct = sum(r[c] == "cannot_tell" for r in main)
        print(f"{c:16s} {acc:17.3f} {ai:22.3f} {ct:17d}")
        out[c] = dict(acc72=acc, india=ai, cannot_tell=ct)
    d, lo, hi = accuracy_difference(main, "optical", "combined")
    print(f"\ncombined minus optical (72 questions): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out["combined_minus_optical"] = dict(diff=d, lo=lo, hi=hi)
    print("\nchanged answers (combined vs optical):")
    for r in main:
        if r["combined"] != r["optical"]:
            print(f"  {r['id']:24s} expected {r['key']:11s} optical {r['optical']:11s} -> combined {r['combined']}")
    out["rows"] = rows
    (B.RUNS / "report_C.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"A": part_a, "B-select": part_b_select, "B-final": part_b_final, "C": part_c}[sys.argv[1]]()
