"""B1, test 3: the whole answer path on the 78 flood questions of exam v2, NDWI > 0 against a trained model.

Everything except the water detector is held fixed: the true periods of each question (as in the
no-model reference), the same images, the same cloud mask, the same composites (per-pixel median of the
clear images of each period), the same usable pixels, the same decision rule (new water >= 0.05 -> yes).
Only "which pixels of a composite are water" changes. For a question about flood event E, the model is
the one fine-tuned with E left out.

Two steps, because the image code and the model code live in different Python environments:
    source activate.sh
    python -m experiments.b1_exam composites                       # downloads; writes runs/b1/exam/<id>.npz
    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b1_exam score --model prithvi
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "b1" / "exam"
BANDS7 = ("blue", "green", "red", "nir", "nir08", "swir16", "swir22")
FLOOD_MIN_NEW_WATER = 0.05            # the frozen rule (eval/reference_pipeline.py)
EVENT = {"flood_" + e: e for e in ("Bolivia", "Ghana", "India", "Mekong", "Nigeria", "Pakistan", "Paraguay",
                                   "Somalia", "Spain", "Sri-Lanka", "USA")}


def questions() -> list[dict]:
    qs = [json.loads(line) for line in (ROOT / "eval" / "benchmark_v2.jsonl").open()]
    return [q for q in qs if q["category"] == "flood"]


def composites() -> None:
    """Per question: median composites of both periods (7 bands) and the usable mask, as in
    detect_change_timeseries (>= 2 clear images before and >= 1 after; at least 50 % of pixels usable)."""
    from core.data.timeseries import load_timeseries
    from core.schemas import SearchRequest
    from core.tools import Area, _search

    OUT.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(questions(), 1):
        f = OUT / f"{q['id']}.npz"
        if f.exists():
            continue
        ref = q["reference"]
        bbox = Area(lat=ref["lat"], lon=ref["lon"], size_km=ref["size_km"]).bbox()
        comp, counts, status = {}, {}, "ok"
        for name in ("before", "after"):
            d0, d1 = (date.fromisoformat(s) for s in ref[name])
            scenes = _search(SearchRequest(bbox=bbox, start=d0, end=d1)).scenes
            if not scenes:
                status = f"no images in the {name} period"
                break
            ds = load_timeseries(scenes, bbox, bands=BANDS7)
            x = np.stack([ds[b].values for b in BANDS7])          # (7, time, y, x), NaN where not clear
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore")
                comp[name] = np.nanmedian(x, axis=1).astype(np.float32)
            counts[name] = np.isfinite(x[2]).sum(axis=0)           # clear images per pixel (red band)
        if status == "ok":
            usable = (counts["before"] >= 2) & (counts["after"] >= 1)
            if usable.mean() < 0.5:
                status = f"only {usable.mean():.0%} of pixels usable"
            np.savez_compressed(f, status=status, before=comp["before"], after=comp["after"], usable=usable)
        else:
            np.savez_compressed(f, status=status)
        print(f"  {i}/{len(questions())} {q['id']}: {status}", flush=True)


def ndwi_water(c: np.ndarray) -> np.ndarray:
    green, nir = c[1], c[3]
    with np.errstate(all="ignore"):
        return (green - nir) / (green + nir) > 0


def score(model: str) -> None:
    import torch
    from experiments import b1_common as C
    from experiments.b1_train import STATS, build, predict

    dev = torch.device("cuda")
    mean, std = (np.array(v, np.float32)[:, None, None] * 1e-4 for v in STATS[model])
    nets = {}
    rows = []
    for q in questions():
        z = np.load(OUT / f"{q['id']}.npz")
        status = str(z["status"])
        row = dict(id=q["id"], event=EVENT[q["cluster"]], expected=q["expected"], status=status)
        if status != "ok":
            row.update(rule="cannot_tell", model="cannot_tell")
            rows.append(row)
            continue
        u = z["usable"]
        ev = row["event"]
        if ev not in nets:
            w = C.RUNS / "weights" / model / f"{ev}.pt"
            if not w.exists():   # India: no L2A tiles, so no fold; its questions have no usable images anyway
                raise FileNotFoundError(w)
            net = build(model)
            net.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
            nets[ev] = net.to(dev).eval()
        for name, water in (("rule", ndwi_water), ("model", None)):
            frac = {}
            for period in ("before", "after"):
                c = z[period]
                if water is None:
                    x = np.nan_to_num((c[C.MODEL_BANDS] - mean) / std, nan=0.0).astype(np.float32)
                    w_ = predict(nets[ev], x, dev) > 0.5
                else:
                    w_ = water(c)
                frac[period] = float(w_[u].mean())
            new = frac["after"] - frac["before"]
            row[name] = "yes" if new >= FLOOD_MIN_NEW_WATER else "no"
            row[f"{name}_water"] = [round(frac["before"], 4), round(frac["after"], 4)]
        rows.append(row)
        print(f"  {q['id']:28s} expected {q['expected']:11s} rule {row['rule']:4s} {row['rule_water']}   "
              f"{model} {row['model']:4s} {row['model_water']}", flush=True)
    (OUT.parent / f"exam_{model}.json").write_text(json.dumps(rows, indent=1))
    acc = lambda k: np.mean([r[k] == r["expected"] for r in rows])
    print(f"\n{len(rows)} flood questions: accuracy NDWI>0 {acc('rule'):.3f}, {model} {acc('model'):.3f}")


def pipeline() -> None:
    """The same 78 questions through the pipeline's own code (core.tools with the U-Net in
    detect_change), with each question's event left out. Must give the answers of `score --model unet`."""
    from core import tools
    from core.measure import decide
    from core.models.water import BANDS7 as B7

    want = {r["id"]: r for r in json.loads((OUT.parent / "exam_unet.json").read_text())}
    same = 0
    for q in questions():
        ref, ev = q["reference"], EVENT[q["cluster"]]
        area = dict(lat=ref["lat"], lon=ref["lon"], size_km=ref["size_km"])
        b = tools.cloud_mask_and_stack(start=date.fromisoformat(ref["before"][0]),
                                       end=date.fromisoformat(ref["before"][1]), bands=B7, **area)
        a = tools.cloud_mask_and_stack(start=date.fromisoformat(ref["after"][0]),
                                       end=date.fromisoformat(ref["after"][1]), bands=B7, **area)
        if b.cube_id is None or a.cube_id is None:
            ans = "cannot_tell"
        else:
            ans = decide("flood", tools.detect_change(b.cube_id, a.cube_id, water=f"unet-without-{ev}"))[0]
        same += ans == want[q["id"]]["model"]
        if ans != want[q["id"]]["model"]:
            print(f"  DIFFERENT {q['id']}: pipeline {ans}, experiment {want[q['id']]['model']}")
    print(f"pipeline code gives the experiment's answer on {same} of {len(want)} questions")


if __name__ == "__main__":
    if sys.argv[1] == "composites":
        composites()
    elif sys.argv[1] == "pipeline":
        pipeline()
    else:
        score(sys.argv[sys.argv.index("--model") + 1])
