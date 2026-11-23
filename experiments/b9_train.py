"""B9, avenue 1, candidate N: B4's TerraMind trained with flooded places labelled "no change"
(protocol: docs/METHODS.md section 12).

Training exactly as B4 (experiments/b4_train.py: 7 bands, TerraMind statistics, 224 crops, flips and
rotations, AdamW 5e-5 cosine, weight decay 0.05, 3 000 steps, cross-entropy + Dice, bf16), with 2 crops of
flooded places added to each step's 8 OSCD crops (batch 10). Flooded places: the recorded Sen1Floods11
chips (experiments/b9_flood_data.py), label "no change" on the usable pixels, unlabelled elsewhere, every
chip with equal weight.

Runs ("groups"):
    ev_<event>     every OSCD city; chips of every other flood event; predicts that event's Part C questions
    <oscd group>   B4's leave-one-group-out; chips of every event except those within 20 km of the group's
                   cities; predicts the group's cities (Part A)
    ALL            every city and every chip; predicts Conselice; weights kept (the pipeline)
    paris_bercy    every city except Paris and Bercy, every chip; weights kept (the exam's Paris question)

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b9_train --group ev_Spain --seed 0
    .venv-ml/bin/python -m experiments.b9_train --all --gpus 1,2,4,5 [--model s,sn]   (avenue 2: s, sn)
Writes runs/b9/runs/n_<group>_s<seed>/: result.json, prob.npz (and weights.pt for ALL and paris_bercy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import oscd_pc as D  # noqa: E402
from experiments import b4_common as C  # noqa: E402
from experiments.b4_train import CROP, LR, STEPS, Net, crops, load, loss_fn, predict  # noqa: E402

RUNS = ROOT / "runs" / "b9"
FLOOD = ROOT / "data_cache" / "b9" / "flood"
PARTC = ROOT / "data_cache" / "b9" / "partc"
N_FLOOD = 2
SN7 = ROOT / "data_cache" / "spacenet7" / "inputs"
MODELS = {"n": (2, 0), "s": (0, 2), "sn": (2, 2)}      # (flood crops, SpaceNet 7 crops) added to B4's 8 OSCD crops
NEAR_KM = 20.0
KEEP_WEIGHTS = ("ALL", "paris_bercy")
EVENTS = ("Bolivia", "Ghana", "Mekong", "Nigeria", "Pakistan", "Paraguay", "Somalia", "Spain", "Sri-Lanka", "USA")
# (India: its 2016 chips have no Level-2A images, so it gives no examples and all its questions are "cannot tell")


def chip_list() -> list[dict]:
    from experiments.b9_flood_data import chips
    meta = {c["name"]: c for c in chips()}
    return [meta[f.stem] for f in sorted(FLOOD.glob("*.npz")) if f.stem in meta]


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def city_centre(city: str) -> tuple[float, float]:
    w, s, e, n = D.grid(city)[1]
    return (s + n) / 2, (w + e) / 2


def sn7_list() -> list[dict]:
    """SpaceNet 7 areas with inputs: name and centre (lat, lon)."""
    from pyproj import Transformer
    out = []
    for f in sorted(SN7.glob("*.npz")):
        meta = json.loads((ROOT / "data_cache" / "spacenet7" / "labels" / f.stem / "meta.json").read_text())
        w, s, e, n = meta["bounds"]
        lon, lat = Transformer.from_crs(meta["crs"], "EPSG:4326", always_xy=True).transform((w + e) / 2, (s + n) / 2)
        out.append({"name": f.stem, "lat": lat, "lon": lon})
    return out


def plan(group: str, model: str = "n") -> tuple[list[str], list[dict], list[str], list[dict]]:
    """OSCD training cities, training chips, what is predicted, and training SpaceNet 7 areas."""
    kept = C.kept()
    chips = chip_list() if MODELS[model][0] else []
    areas = sn7_list() if MODELS[model][1] else []
    if group.startswith("ev_"):
        ev = group[3:]
        return kept, [c for c in chips if c["event"] != ev], [f.stem for f in PARTC.glob("*.npz") if
                                                              f.stem.startswith(f"flood_{ev.lower()}_")], areas
    if group == "ALL":
        return kept, chips, [f.stem for f in PARTC.glob("*.npz")], areas
    test = [c for c in kept if D.group(c) == group]
    centres = [city_centre(c) for c in test]
    near = lambda ch: any(_km(ch["lat"], ch["lon"], *x) < NEAR_KM for x in centres)   # noqa: E731
    return ([c for c in kept if D.group(c) != group], [c for c in chips if not near(c)], test,
            [a for a in areas if not near(a)])


def load_chips(chips: list[dict]) -> list[tuple]:
    out = []
    for c in chips:
        z = np.load(FLOOD / f"{c['name']}.npz")
        b, a = z["before"], z["after"]
        ok = z["usable"] & np.isfinite(b).all(0) & np.isfinite(a).all(0)
        if b.shape[1] < CROP or b.shape[2] < CROP or ok.sum() == 0:
            continue
        out.append((C.scale(b), C.scale(a), np.where(ok, 0, -1).astype(np.int8)))
    return out


def load_sn7(areas: list[dict]) -> list[tuple]:
    out = []
    for a in areas:
        z = np.load(SN7 / f"{a['name']}.npz")
        if z["label"].shape[0] < CROP or z["label"].shape[1] < CROP:
            continue
        ok = np.isfinite(z["before"]).all(0) & np.isfinite(z["after"]).all(0)
        out.append((C.scale(z["before"]), C.scale(z["after"]), np.where(ok, z["label"], -1).astype(np.int8)))
    return out


def flood_crops(chips: list[tuple], rng, n: int = N_FLOOD) -> tuple:
    B, A, Y = [], [], []
    for _ in range(n):
        b, a, y = chips[rng.integers(len(chips))]
        i, j = rng.integers(0, y.shape[0] - CROP + 1), rng.integers(0, y.shape[1] - CROP + 1)
        b, a, y = b[:, i:i + CROP, j:j + CROP], a[:, i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP]
        k = rng.integers(4)
        b, a, y = np.rot90(b, k, (1, 2)), np.rot90(a, k, (1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            b, a, y = b[:, :, ::-1], a[:, :, ::-1], y[:, ::-1]
        B.append(np.ascontiguousarray(b)), A.append(np.ascontiguousarray(a)), Y.append(np.ascontiguousarray(y))
    return np.stack(B), np.stack(A), np.stack(Y).astype(np.int64)


def _inputs(name: str) -> dict:
    if name in C.kept():
        return C.inputs(name)
    z = np.load(PARTC / f"{name}.npz")
    return {"before": z["before"], "after": z["after"]}


def train_one(group: str, seed: int, model: str = "n") -> dict:
    name = f"{model}_{group}_s{seed}"
    out_dir = RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)            # the OSCD crops: the same stream of numbers as B4's run
    rng_f = np.random.default_rng([seed, 9])     # the flood crops: a separate stream
    rng_s = np.random.default_rng([seed, 7])     # the SpaceNet 7 crops: another one
    n_f, n_s = MODELS[model]
    cities, chips, test, areas = plan(group, model)
    data, fl, sn = load(cities), load_chips(chips), load_sn7(areas)
    net = Net("terramind").to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR["terramind"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    losses, run = [], []
    net.train()
    for step in range(STEPS):
        b, a, y = crops(data, rng)
        for src, rg, k in ((fl, rng_f, n_f), (sn, rng_s, n_s)):
            if k:
                fb, fa, fy = flood_crops(src, rg, k)
                b, a, y = np.concatenate([b, fb]), np.concatenate([a, fa]), np.concatenate([y, fy])
        yb = torch.from_numpy(y).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(torch.from_numpy(b).to(dev), torch.from_numpy(a).to(dev)).float()
        lo = loss_fn(logits, yb)
        opt.zero_grad(set_to_none=True)
        lo.backward()
        opt.step()
        sched.step()
        run.append(lo.item())
        if (step + 1) % 250 == 0:
            losses.append(float(np.mean(run)))
            run = []
            print(f"  [{name}] step {step + 1}/{STEPS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = {c: np.round(predict(net, _inputs(c), dev) * 255).astype(np.uint8) for c in test}
    np.savez_compressed(out_dir / "prob.npz", **probs)
    if group in KEEP_WEIGHTS:
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
                   out_dir / "weights.pt")
    res = {"name": name, "model": model, "group": group, "seed": seed, "lr": LR["terramind"], "steps": STEPS,
           "n_flood_per_step": n_f, "n_sn7_per_step": n_s, "n_train_sn7": len(sn),
           "train_cities": cities, "n_train_chips": len(fl), "train_events": sorted({c["event"] for c in chips}),
           "test": test, "seconds": round(time.time() - t0), "loss_per_250_steps": losses,
           "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def groups(model: str = "n") -> list[str]:
    oscd = sorted({D.group(c) for c in C.kept()}) + ["ALL"]          # "paris_bercy" is one of B4's groups
    return ([f"ev_{e}" for e in EVENTS] if MODELS[model][0] else []) + oscd


def run_all(gpus: list[str], models: list[str], per_gpu: int = 2) -> None:
    jobs = [(m, g, s) for m in models for g in groups(m) for s in range(3)
            if not (RUNS / "runs" / f"{m}_{g}_s{s}" / "result.json").exists()]
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = []
    print(f"{len(jobs)} runs to do", flush=True)
    while jobs or running:
        running = [(p, g) for p, g in running if p.poll() is None]
        free = list(slots)
        for _, g in running:
            free.remove(g)
        while jobs and free:
            (m, gr, s), g = jobs.pop(0), free.pop(0)
            p = subprocess.Popen([sys.executable, "-m", "experiments.b9_train", "--model", m, "--group", gr, "--seed", str(s)],
                                 cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"{m}_{gr}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(10)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="ALL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default="1,2,4,5")
    ap.add_argument("--model", default="n", help="n, s, sn, or with --all a comma list")
    a = ap.parse_args()
    if a.all:
        run_all(a.gpus.split(","), a.model.split(","))
    else:
        train_one(a.group, a.seed, a.model)
