"""B8, the time-series burn detector with radar and elevation (protocol: docs/METHODS.md section 11, detector 3).

"utae_rd": B7's U-TAE (core/models/utae.py) with 12 input channels per date. The optical dates of the B7 copy
(core/data/impactmesh_ts.py) and the radar dates of core/data/burn_radar.py form one sequence in time order
(optical first on a day with both). Channels of every date:
    0-6   the 7 optical bands, scaled as in B5 (missing 0)        radar dates: 0
    7     clear flag of the optical pixel                          radar dates: 0
    8-9   VV, VH in dB, scaled with ImpactMesh's fire statistics   optical dates: 0
    10    radar flag: 1 where the radar pixel has data             optical dates: 0
    11    elevation, scaled with ImpactMesh's fire statistics      every date
Training as B7 (B5's training patches, random 224 x 224 crops with the same crop, rotation and flip for every
date, AdamW 1e-3 cosine, weight decay 0.05, 10 000 steps of 8 patches, cross-entropy + Dice, bf16, 3 seeds);
8 patches per update as batch 2 with 4 steps accumulated (up to 40 dates do not fit in batch 8 on one A10).

    .venv-ml/bin/python -m experiments.b8_train --all --slots 1,2,4
Writes runs/b8/runs/utae_rd_s<seed>/: result.json, weights.pt, prob_heldout.npz (the 42 held-out fires).
Predictions on the new fires of 2026: experiments/b8_predict.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models.utae import UTAE  # noqa: E402
from experiments import b5_common as C  # noqa: E402
from experiments.b4_train import loss_fn  # noqa: E402  (cross-entropy + Dice, label -1 ignored)
from experiments.b5_train import CORNERS, CROP, STEPS  # noqa: E402
from experiments.b7_train import LR  # noqa: E402

RUNS = ROOT / "runs" / "b8"
DATA = ROOT / "data_cache" / "burn"
SOURCES = {"impactmesh": DATA / "impactmesh", "cems2026": DATA / "cems"}
S1_MEAN, S1_STD = np.array([-10.132, -15.81], np.float32), np.array([3.677, 3.682], np.float32)
DEM_MEAN, DEM_STD = 456.555, 387.825
N_CH = 12
BATCH, ACCUM = 2, 4


def series(name: str, source: str = "impactmesh", mmap: bool = True):
    """Optical and radar dates of one patch, merged in time order: (T, 12, 256, 256) float32, days, period."""
    d = SOURCES[source]
    mo = json.loads((d / "ts" / f"{name}.json").read_text())
    xo = np.load(d / "ts" / f"{name}.npy", mmap_mode="r" if mmap else None)
    mr = json.loads((d / "s1" / f"{name}.json").read_text())
    xr = np.load(d / "s1" / f"{name}.npy", mmap_mode="r" if mmap else None)
    dem = np.load(d / "s1" / f"{name}_dem.npy").astype(np.float32)
    order = sorted([(day, 0, k) for k, day in enumerate(mo["days"])] + [(day, 1, k) for k, day in enumerate(mr["days"])])
    per = [mo["period"][k] if kind == 0 else mr["period"][k] for _, kind, k in order]
    return (xo, xr, dem, order), np.array([o[0] for o in order]), np.array(per)


def inputs(parts, i: int = 0, j: int = 0, size: int = 256) -> np.ndarray:
    """The (T, 12, size, size) network input of a window of the patch."""
    xo, xr, dem, order = parts
    h = np.nan_to_num((dem[i:i + size, j:j + size] - DEM_MEAN) / DEM_STD, nan=0.0)
    out = np.zeros((len(order), N_CH, size, size), np.float32)
    for t, (_, kind, k) in enumerate(order):
        if kind == 0:
            x = np.asarray(xo[k, :, i:i + size, j:j + size], np.float32)
            out[t, :7] = C.scale(x)
            out[t, 7] = np.isfinite(x).all(0)
        else:
            r = np.asarray(xr[k, :, i:i + size, j:j + size], np.float32)
            ok = np.isfinite(r).all(0)
            out[t, 8:10] = np.where(ok, (r - S1_MEAN[:, None, None]) / S1_STD[:, None, None], 0.0)
            out[t, 10] = ok
        out[t, 11] = h
    return out


def batch(items: list) -> tuple:
    T = max(len(d) for _, d, _, _ in items)
    B, H, W = len(items), *items[0][0].shape[-2:]
    X = np.zeros((B, T, N_CH, H, W), np.float32)
    D, P = np.zeros((B, T), np.int64), np.zeros((B, T), np.int64)
    pad = np.ones((B, T), bool)
    Y = np.stack([y for *_, y in items]).astype(np.int64)
    for k, (x, d, p, _) in enumerate(items):
        X[k, :len(d)], D[k, :len(d)], P[k, :len(d)], pad[k, :len(d)] = x, d, p, False
    return tuple(torch.from_numpy(a) for a in (X, D, P, pad, Y))


class Crops(torch.utils.data.IterableDataset):
    def __init__(self, seed: int, n: int, n_batch: int):
        self.seed, self.n, self.n_batch = seed, n, n_batch
        self.names = json.loads((C.PACKED / "train_names.json").read_text())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, wid])
        Y = np.load(C.PACKED / "train_y.npy", mmap_mode="r")
        for _ in range(wid, self.n, nw):
            items = []
            for k in rng.integers(0, len(self.names), self.n_batch):
                i, j = rng.integers(0, 256 - CROP + 1, 2)
                parts, d, p = series(self.names[k])
                x = inputs(parts, i, j, CROP)
                y = np.asarray(Y[k, i:i + CROP, j:j + CROP])
                r = rng.integers(4)
                x, y = np.rot90(x, r, (2, 3)), np.rot90(y, r)
                if rng.random() < 0.5:
                    x, y = x[..., ::-1], y[:, ::-1]
                items.append((np.ascontiguousarray(x), d, p, np.ascontiguousarray(y)))
            yield batch(items)


@torch.no_grad()
def predict(net, name: str, source: str, dev) -> np.ndarray:
    """Probability of "burnt" (256, 256): the four corner windows of 224, averaged where they overlap."""
    parts, d, p = series(name, source, mmap=False)
    X = torch.from_numpy(np.stack([inputs(parts, i, j, CROP) for i, j in CORNERS])).to(dev)
    D = torch.from_numpy(np.repeat(d[None], 4, 0)).to(dev)
    P = torch.from_numpy(np.repeat(p[None], 4, 0)).to(dev)
    pad = torch.zeros(D.shape, dtype=torch.bool, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pr = torch.softmax(net(X, D, P, pad).float(), dim=1)[:, 1].cpu().numpy()
    prob, hits = np.zeros((256, 256), np.float32), np.zeros((256, 256), np.float32)
    for (i, j), pi in zip(CORNERS, pr):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return prob / hits


def network() -> UTAE:
    return UTAE(c_in=N_CH)


def train_one(seed: int) -> dict:
    name = f"utae_rd_s{seed}"
    out_dir = RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    ds = Crops(seed, STEPS * ACCUM, BATCH)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=6, prefetch_factor=4, persistent_workers=True)
    net = network().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    losses, run = [], []
    net.train()
    it = iter(loader)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            X, D, P, pad, Y = (a.to(dev, non_blocking=True) for a in next(it))
            if (Y >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(X, D, P, pad).float()
            lo = loss_fn(logits, Y) / ACCUM
            lo.backward()
            run.append(lo.item() * ACCUM)
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            losses.append(float(np.mean(run)))
            run = []
            print(f"  [{name}] step {step + 1}/{STEPS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    test = C.built("test")
    probs = {n: np.round(predict(net, n, "impactmesh", dev) * 255).astype(np.uint8) for n in test}
    np.savez_compressed(out_dir / "prob_heldout.npz", **probs)
    res = {"name": name, "model": "utae_rd", "seed": seed, "lr": LR, "steps": STEPS, "batch": BATCH, "accumulate": ACCUM,
           "n_train_patches": len(ds.names), "n_test_patches": len(test), "seconds": round(time.time() - t0),
           "loss_per_500_steps": losses, "parameters": sum(p.numel() for p in net.parameters()),
           "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def run_all(slots: list[str]) -> None:
    jobs = [s for s in range(3) if not (RUNS / "runs" / f"utae_rd_s{s}" / "result.json").exists()]
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for s, g in zip(jobs, slots):
        procs.append(subprocess.Popen([sys.executable, "-m", "experiments.b8_train", "--seed", str(s)], cwd=ROOT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                      stdout=open(logs / f"train_utae_rd_s{s}.log", "w"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slots", default="1,2,4")
    a = ap.parse_args()
    if a.all:
        run_all(a.slots.split(","))
    else:
        train_one(a.seed)
