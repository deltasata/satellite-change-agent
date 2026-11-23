"""B9, avenue 3, candidate M: B2's radar TerraMind trained with MMFlood's tiles added
(protocol: docs/METHODS.md section 12).

Training exactly as B2 (experiments/b2_train.py: TerraMind-1.0-base S1RTC + UperNet, batch 16 of the fold's
Sen1Floods11 tiles, crops of 224 with flips and rotations, AdamW 5e-5 cosine, 50 epochs, cross-entropy),
with 8 crops of MMFlood tiles added to every step. MMFlood tiles (core/data/mmflood_s1rtc.py) are stored once
as uint8-free arrays on disk (float16 input, int8 label) and read without loading them all into memory.

    .venv-ml/bin/python -m experiments.b9_radar_train prep            # MMFlood tiles -> arrays on disk
    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b9_radar_train --fold Spain --seed 0
    .venv-ml/bin/python -m experiments.b9_radar_train --all --gpus 1,2,4,5
Writes runs/b9/radar/m_<fold>_s<seed>/: result.json (counts per predicted tile, as B2), prob.npz, weights.pt.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b2_common as B  # noqa: E402

RUNS = ROOT / "runs" / "b9" / "radar"
MM = ROOT / "data_cache" / "mmflood"
ARR = MM / "arrays"
N_MM = 8
NEAR_KM = 300.0
NEVER = ("EMSR314", "EMSR388")


def prep() -> None:
    ARR.mkdir(parents=True, exist_ok=True)
    mean, std = (np.array(v, np.float32)[:, None, None] for v in B.STATS["rtc"])
    n = 0
    for f in sorted((MM / "s1rtc").glob("*.npz")):
        if (ARR / f"{f.stem}_x.npy").exists():
            continue
        z = np.load(f)
        x = np.nan_to_num((z["x"] - mean) / std, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)
        np.save(ARR / f"{f.stem}_x.npy", x)
        np.save(ARR / f"{f.stem}_y.npy", z["label"].astype(np.int8))
        n += 1
    print(f"{n} tiles written")


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(a))


def mm_tiles(fold: str) -> list[str]:
    """MMFlood tiles allowed in this fold."""
    acts = json.loads((MM / "extracted" / "activations.json").read_text())
    from experiments.b9_flood_data import chips
    if fold not in ("ALL", "official"):
        cs = [c for c in chips() if c["event"] == fold]
        lat, lon = np.mean([c["lat"] for c in cs]), np.mean([c["lon"] for c in cs])
    bad = set(NEVER)
    if fold != "ALL":
        bad |= {k for k, v in acts.items() if _km(v["lat"], v["lon"], lat, lon) < NEAR_KM}
    names = sorted(f.name[:-6] for f in ARR.glob("*_x.npy"))
    return [t for t in names if t.split("-")[0] not in bad and min(np.load(ARR / f"{t}_y.npy", mmap_mode="r").shape) >= 224]


def mm_crops(tiles: list[str], rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    from experiments.b2_train import CROP
    xs, ys = [], []
    for _ in range(n):
        t = tiles[rng.integers(len(tiles))]
        x = np.load(ARR / f"{t}_x.npy", mmap_mode="r")
        y = np.load(ARR / f"{t}_y.npy", mmap_mode="r")
        i, j = rng.integers(0, y.shape[0] - CROP + 1), rng.integers(0, y.shape[1] - CROP + 1)
        xc, yc = np.asarray(x[:, i:i + CROP, j:j + CROP], np.float32), np.asarray(y[i:i + CROP, j:j + CROP], np.int64)
        k = rng.integers(4)
        xc, yc = np.rot90(xc, k, (1, 2)), np.rot90(yc, k)
        if rng.random() < 0.5:
            xc, yc = xc[:, :, ::-1], yc[:, ::-1]
        xs.append(np.ascontiguousarray(xc)), ys.append(np.ascontiguousarray(yc))
    return np.stack(xs), np.stack(ys)


def train_one(fold: str, seed: int) -> dict:
    import torch
    from core.data import sen1floods11 as s1f
    from experiments.b2_train import BATCH, DEFAULTS, Net, augment, loss_fn, predict, split
    name = f"m_{fold}_s{seed}"
    out_dir = RUNS / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)               # the Sen1Floods11 batches: B2's stream of numbers
    rng_m = np.random.default_rng([seed, 3])        # the MMFlood crops: a separate stream
    data = {}
    for c in s1f.chips():
        s = B.scored(c, "s1", "rtc")
        if s is not None:
            x, y, m = s
            data[c] = (x, np.where(m, y, -1).astype(np.int64))
    train, test = split(fold, data)
    tiles = mm_tiles(fold)
    net = Net("terramind", "s1", "rtc").to(dev)
    lr, epochs = DEFAULTS["terramind"]["lr"], DEFAULTS["terramind"]["epochs"]
    steps_per_epoch = math.ceil(len(train) / BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    losses = []
    net.train()
    for epoch in range(epochs):
        order = rng.permutation(train)
        run = []
        for k in range(steps_per_epoch):
            batch = [augment(*data[c], rng) for c in order[k * BATCH:(k + 1) * BATCH]]
            if len(batch) < 2:
                continue
            mx, my = mm_crops(tiles, rng_m, N_MM)
            xb = torch.from_numpy(np.concatenate([np.stack([b[0] for b in batch]), mx])).to(dev)
            yb = torch.from_numpy(np.concatenate([np.stack([b[1] for b in batch]), my])).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xb).float()
            lo = loss_fn(logits, yb, "ce")
            opt.zero_grad(set_to_none=True)
            lo.backward()
            opt.step()
            sched.step()
            run.append(lo.item())
        losses.append(float(np.mean(run)) if run else float("nan"))
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  [{name}] epoch {epoch + 1}/{epochs} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    per_chip, probs = {}, {}
    for c in test:
        x, y = data[c]
        p = predict(net, x, dev, views=1)
        per_chip[c] = B.counts(p > 0.5, y, y >= 0)
        probs[c] = np.round(p * 255).astype(np.uint8)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "prob.npz", **probs)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    res = {"name": name, "fold": fold, "seed": seed, "lr": lr, "epochs": epochs, "n_train_tiles": len(train),
           "n_mmflood_tiles": len(tiles), "mmflood_events": sorted({t.split("-")[0] for t in tiles}),
           "n_mmflood_per_step": N_MM, "n_predicted_tiles": len(test), "seconds": round(time.time() - t0),
           "loss_per_epoch": losses, "counts": per_chip}
    (out_dir / "result.json").write_text(json.dumps(res))     # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def folds() -> list[str]:
    from core.data import sen1floods11 as s1f
    evs = sorted({s1f.event(c) for c in s1f.chips() if B.radar(c, "rtc") is not None})
    return evs + ["ALL"]


def run_all(gpus: list[str], per_gpu: int = 2) -> None:
    jobs = [(f, s) for f in folds() for s in range(3) if not (RUNS / f"m_{f}_s{s}" / "result.json").exists()]
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
            (f, s), g = jobs.pop(0), free.pop(0)
            p = subprocess.Popen([sys.executable, "-m", "experiments.b9_radar_train", "--fold", f, "--seed", str(s)],
                                 cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"m_{f}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(10)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["prep"]:
        prep()
        sys.exit()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="ALL")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default="1,2,4,5")
    a = ap.parse_args()
    if a.all:
        run_all(a.gpus.split(","))
    else:
        train_one(a.fold, a.seed)
