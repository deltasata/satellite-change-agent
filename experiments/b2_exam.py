"""B2, Part C: the 78 flood questions of exam v2, answered with radar (protocol: docs/METHODS.md section 4).

Held fixed as in B1: the true periods of each question, the decision rule (new water >= 0.05 -> yes),
and the optical answer of B1's U-Net (runs/b1/exam_unet.json).

Radar per question: every sentinel-1-rtc scene of the before and after periods that covers the 5 km
area, VV and VH in dB on a 10 m grid. A detector gives a water probability per scene (a rule gives 0 or
1). Per period: the per-pixel median over the scenes, water where > 0.5. Usable pixels have a radar value
in both periods; at least 50 % of the area must be usable, otherwise the radar answer is "cannot tell".

Combined answer: the optical answer when the optical images can tell, otherwise the radar answer.

    source activate.sh
    python -m experiments.b2_exam radar                                  # downloads; runs/b2/exam/<id>.npz
    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b2_exam score --model unet18
    python -m experiments.b2_exam report
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.b1_exam import EVENT, FLOOD_MIN_NEW_WATER, questions  # noqa: E402

OUT = ROOT / "runs" / "b2" / "exam"
MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
INDIA_LABEL = {"yes": "yes", "no": "no"}   # India's questions are scored against the hand label (section 8)


def load_period(bbox, d0: str, d1: str):
    """(x, days, scenes): x of shape (n, 2, H, W), VV and VH in dB, one entry per day with a scene."""
    import odc.stac
    import planetary_computer
    from pystac_client import Client

    from core.data.sen1floods11_s1rtc import to_db
    cat = Client.open(MPC_STAC, modifier=planetary_computer.sign_inplace)
    end = (date.fromisoformat(d1) + timedelta(days=1)).isoformat()
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{d0}/{end}").items())
    items = [it for it in items if "vv" in it.assets and "vh" in it.assets]   # some scenes are HH/HV or VV only
    if not items:
        return None, [], []
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=10,
                           groupby="solar_day", resampling="bilinear", nodata=-32768, dtype="float32",
                           fail_on_error=True)
    x = np.stack([to_db(np.where(ds[b].values == -32768, np.nan, ds[b].values)) for b in ("vv", "vh")], axis=1)
    keep = [k for k in range(x.shape[0]) if np.isfinite(x[k]).all(axis=0).mean() > 0.01]
    days = [str(t)[:10] for t in ds.time.values]
    return x[keep], [days[k] for k in keep], [it.id for it in items]


def radar() -> None:
    from core.tools import Area
    OUT.mkdir(parents=True, exist_ok=True)
    qs = questions()
    for i, q in enumerate(qs, 1):
        f = OUT / f"{q['id']}.npz"
        if f.exists():
            continue
        ref = q["reference"]
        a = Area(lat=ref["lat"], lon=ref["lon"], size_km=ref["size_km"])
        b = a.bbox()
        bbox = [b.west, b.south, b.east, b.north]
        got = {}
        for name in ("before", "after"):
            got[name] = load_period(bbox, *ref[name])
        nb, na = (0 if got[n][0] is None else got[n][0].shape[0] for n in ("before", "after"))
        if nb and na:
            np.savez_compressed(f, before=got["before"][0], after=got["after"][0],
                                before_days=got["before"][1], after_days=got["after"][1])
        else:
            np.savez_compressed(f, before_days=got["before"][1], after_days=got["after"][1])
        print(f"  {i}/{len(qs)} {q['id']}: {nb} radar days before, {na} after", flush=True)


def answer(prob_before: np.ndarray, prob_after: np.ndarray) -> tuple[str, list]:
    """Per-pixel median over the scenes of a period, water where > 0.5, then the frozen rule."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wb, wa = np.nanmedian(prob_before, axis=0), np.nanmedian(prob_after, axis=0)
    usable = np.isfinite(wb) & np.isfinite(wa)
    if usable.mean() < 0.5:
        return "cannot_tell", [f"only {usable.mean():.0%} usable"]
    fb, fa = float((wb[usable] > 0.5).mean()), float((wa[usable] > 0.5).mean())
    return ("yes" if fa - fb >= FLOOD_MIN_NEW_WATER else "no"), [round(fb, 4), round(fa, 4)]


def score(model: str) -> None:
    """Radar answers with the rule (VH < t, t fitted without the question's event) and with the
    network (mean of the 3 seeds of the fold that left the question's event out)."""
    import torch

    from experiments import b2_common as B
    from experiments.b2_train import Net, predict, run_name

    dev = torch.device("cuda")
    th = json.loads((B.RUNS / "thresholds_rtc.json").read_text())
    mean, std = (np.array(v, np.float32)[:, None] for v in B.STATS["rtc"])
    nets = {}
    rows = []
    for q in questions():
        z = np.load(OUT / f"{q['id']}.npz")
        ev = EVENT[q["cluster"]]
        fold = ev if f"vh_without_{ev}" in th else "ALL"          # Paraguay is in no training set
        row = dict(id=q["id"], event=ev, fold=fold, expected=q["expected"],
                   radar_days=[len(z["before_days"]), len(z["after_days"])])
        if "before" not in z:
            row.update(rule="cannot_tell", net="cannot_tell", rule_water=["no radar image"], net_water=["no radar image"])
            rows.append(row)
            continue
        if fold not in nets:
            nets.clear()                       # keep one event's networks in GPU memory at a time
            torch.cuda.empty_cache()
            nets[fold] = []
            for s in range(3):
                net = Net(model, "s1", "rtc")
                w = B.RUNS / "runs" / run_name(model, "s1", "rtc", fold, s) / "weights.pt"
                net.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
                nets[fold].append(net.to(dev).eval())
        t = th[f"vh_without_{fold}"] if fold != "ALL" else th["vh_ALL"]
        probs = {"rule": {}, "net": {}}
        for period in ("before", "after"):
            x = z[period]                                            # (n, 2, H, W) dB
            present = np.isfinite(x).all(axis=1)
            probs["rule"][period] = np.where(present, (x[:, 1] < t).astype(np.float32), np.nan)
            out = []
            for k in range(x.shape[0]):
                xs = np.nan_to_num((x[k].reshape(2, -1) - mean) / std).reshape(x[k].shape).astype(np.float32)
                xs = pad(xs)
                p = np.mean([predict(n, xs, dev) for n in nets[fold]], axis=0)[:x.shape[2], :x.shape[3]]
                out.append(np.where(present[k], p, np.nan))
            probs["net"][period] = np.stack(out)
        for name in ("rule", "net"):
            row[name], row[f"{name}_water"] = answer(probs[name]["before"], probs[name]["after"])
        rows.append(row)
        print(f"  {q['id']:26s} expected {q['expected']:11s} rule {row['rule']:11s} {row['rule_water']}  "
              f"net {row['net']:11s} {row['net_water']}", flush=True)
    (OUT.parent / f"exam_radar_{model}.json").write_text(json.dumps(rows, indent=1))


def pad(x: np.ndarray, size: int = 224) -> np.ndarray:
    """Pad to at least one window in each direction (the 5 km areas are about 500 x 500 pixels)."""
    h, w = x.shape[1:]
    return np.pad(x, ((0, 0), (0, max(0, size - h)), (0, max(0, size - w))))


def pipeline() -> None:
    """The 78 questions through measure_change itself (true periods given explicitly, the U-Net and the
    radar model that left the question's event out). Must give the 'combined' answers of the report."""
    from core.measure import measure_change
    want = {r["id"]: r["combined"] for r in json.loads((OUT.parent / "report_C.json").read_text())["rows"]}
    folds = {p.name.split("_")[2] for p in (ROOT / "data_cache" / "models").glob("radar_terramind_*_s0.onnx")}
    same = 0
    for q in questions():
        ref, ev = q["reference"], EVENT[q["cluster"]]
        r = measure_change(lat=ref["lat"], lon=ref["lon"], kind="flood", size_km=ref["size_km"],
                           before_start=ref["before"][0], before_end=ref["before"][1],
                           after_start=ref["after"][0], after_end=ref["after"][1],
                           water=f"unet-without-{ev}", radar=ev if ev in folds else "ALL")
        same += r.answer == want[q["id"]]
        if r.answer != want[q["id"]]:
            print(f"  DIFFERENT {q['id']}: pipeline {r.answer} ({r.sensor}), experiment {want[q['id']]}")
    print(f"pipeline code gives the experiment's combined answer on {same} of {len(want)} questions")


if __name__ == "__main__":
    if sys.argv[1] == "pipeline":
        pipeline()
        sys.exit()
    if sys.argv[1] == "radar":
        radar()
    elif sys.argv[1] == "score":
        score(sys.argv[sys.argv.index("--model") + 1])
