"""B7, the time-series burn detector (protocol: docs/METHODS.md section 10).

One run = one network, one seed, trained on B5's training patches (the fires near exam fires left out),
tested on the 42 held-out fires.
  - "utae":   U-TAE (core/models/utae.py) on every kept clear date of the per-date copy
              (core/data/impactmesh_ts.py): at most 12 dates per window, days since the after window began.
  - "utae2":  the control: the same U-TAE on B5's two median images, each placed at the middle day of its
              window.
Each date: the 7 bands scaled as in B5 (missing values 0) and a "clear" channel. Settings fixed in the
protocol, as in B5: random 224 x 224 crops (the same crop, rotation and flip for every date), batch 8 (or
4 with 2 steps accumulated, if 8 does not fit on one A10), AdamW (weight decay 0.05), learning rate 1e-3,
cosine decrease, 10 000 steps, cross-entropy + Dice of "burnt" (B5's labels, -1 where unlabelled or a
median is missing), bf16, no early stopping. Prediction: the four corner windows of 224.

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b7_train --model utae --seed 0
    .venv-ml/bin/python -m experiments.b7_train --all --slots 1,2,4,5,5,5 --batch 4 --accum 2
(U-TAE with 24 dates needs about 13 GB at batch 4, so batch 8 is reached by accumulating 2 steps; the
control fits batch 8 directly.)
Writes runs/b7/runs/<model>_s<seed>/ : result.json, prob.npz (held-out patches, x 255) and weights.pt.
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
from core.data import impactmesh_es as E  # noqa: E402
from core.data.impactmesh_ts import TS  # noqa: E402
from core.models.utae import UTAE  # noqa: E402
from experiments import b5_common as C  # noqa: E402
from experiments.b4_train import loss_fn  # noqa: E402  (cross-entropy + Dice, label -1 ignored)
from experiments.b5_train import CORNERS, CROP, STEPS  # noqa: E402

RUNS = ROOT / "runs" / "b7"
LR = 1e-3
MODELS = ("utae", "utae2")
MID_DAYS = {"normal": (-38, 26), "wide": (-68, 41)}   # middle of the before and after windows, days since after start


def series(name: str, model: str, mmap: bool = True, z: dict | None = None):
    """(T, 7, 256, 256) reflectance (NaN where not clear; float16 memory map for "utae"), days and period."""
    if model == "utae":
        meta = json.loads((TS / f"{name}.json").read_text())
        x = np.load(TS / f"{name}.npy", mmap_mode="r" if mmap else None)
        return x, np.array(meta["days"]), np.array(meta["period"])
    z = z if z is not None else C.load(name)
    stage = str(z["stage"])
    return (np.stack([z["before"], z["after"]]), np.array(MID_DAYS[stage]), np.array([0, 1]))


def inputs(x: np.ndarray) -> np.ndarray:
    """(T, 7, h, w) reflectance -> (T, 8, h, w): scaled bands (missing 0) and the clear flag."""
    x = np.asarray(x, np.float32)
    clear = np.isfinite(x).all(1, keepdims=True).astype(np.float32)
    return np.concatenate([np.stack([C.scale(d) for d in x]), clear], axis=1)


def batch(items: list) -> tuple:
    """Pads the dates to the longest series in the batch."""
    T = max(len(d) for _, d, _, _ in items)
    B, H, W = len(items), *items[0][0].shape[-2:]
    X = np.zeros((B, T, 8, H, W), np.float32)
    D, P = np.zeros((B, T), np.int64), np.zeros((B, T), np.int64)
    pad = np.ones((B, T), bool)
    Y = np.stack([y for *_, y in items]).astype(np.int64)
    for k, (x, d, p, _) in enumerate(items):
        X[k, :len(d)], D[k, :len(d)], P[k, :len(d)], pad[k, :len(d)] = x, d, p, False
    return tuple(torch.from_numpy(a) for a in (X, D, P, pad, Y))


class Crops(torch.utils.data.IterableDataset):
    def __init__(self, model: str, seed: int, n: int, n_batch: int):
        self.model, self.seed, self.n, self.n_batch = model, seed, n, n_batch
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
                if self.model == "utae":
                    x, d, p = series(self.names[k], "utae")
                else:
                    x, d, p = np.stack([np.load(C.PACKED / "train_before.npy", mmap_mode="r")[k],
                                        np.load(C.PACKED / "train_after.npy", mmap_mode="r")[k]]), *series_mid(self.names[k])
                x = inputs(x[:, :, i:i + CROP, j:j + CROP])
                y = np.asarray(Y[k, i:i + CROP, j:j + CROP])
                r = rng.integers(4)
                x, y = np.rot90(x, r, (2, 3)), np.rot90(y, r)
                if rng.random() < 0.5:
                    x, y = x[..., ::-1], y[:, ::-1]
                items.append((np.ascontiguousarray(x), d, p, np.ascontiguousarray(y)))
            yield batch(items)


def series_mid(name: str):
    stage = json.loads((TS / f"{name}.json").read_text())["stage"] if (TS / f"{name}.json").exists() else "normal"
    return np.array(MID_DAYS[stage]), np.array([0, 1])


@torch.no_grad()
def predict(net, name: str, model: str, dev) -> np.ndarray:
    """Probability of "burnt" (256, 256): the four corner windows of 224, averaged where they overlap."""
    x, d, p = series(name, model, mmap=False)
    x = inputs(x)
    X = torch.from_numpy(np.stack([x[:, :, i:i + CROP, j:j + CROP] for i, j in CORNERS])).to(dev)
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


def train_one(model: str, seed: int, batch_size: int = 8, accum: int = 1) -> dict:
    name = f"{model}_s{seed}"
    out_dir = RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    ds = Crops(model, seed, STEPS * accum, batch_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=6, prefetch_factor=4, persistent_workers=True)
    net = UTAE().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    losses, run = [], []
    net.train()
    it = iter(loader)
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            X, D, P, pad, Y = (a.to(dev, non_blocking=True) for a in next(it))
            if (Y >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(X, D, P, pad).float()
            lo = loss_fn(logits, Y) / accum
            lo.backward()
            run.append(lo.item() * accum)
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0:
            losses.append(float(np.mean(run)))
            run = []
            print(f"  [{name}] step {step + 1}/{STEPS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    test = C.built("test")
    probs = {n: np.round(predict(net, n, model, dev) * 255).astype(np.uint8) for n in test}
    np.savez_compressed(out_dir / "prob.npz", **probs)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    res = {"name": name, "model": model, "seed": seed, "lr": LR, "steps": STEPS, "batch": batch_size, "accumulate": accum,
           "n_train_patches": len(ds.names), "n_test_patches": len(test), "seconds": round(time.time() - t0),
           "loss_per_500_steps": losses, "parameters": sum(p.numel() for p in net.parameters()),
           "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def run_all(slots: list[str], batch_size: int, accum: int) -> None:
    jobs = [(m, s) for m in MODELS for s in range(3) if not (RUNS / "runs" / f"{m}_s{s}" / "result.json").exists()]
    logs = RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = []
    print(f"{len(jobs)} runs to do", flush=True)
    while jobs or running:
        running = [(p, g) for p, g in running if p.poll() is None]
        free = list(slots)
        for _, g in running:
            free.remove(g)
        while jobs and free:
            (m, s), g = jobs.pop(0), free.pop(0)
            b, acc = (batch_size, accum) if m == "utae" else (8, 1)   # the control (2 dates) fits batch 8
            p = subprocess.Popen([sys.executable, "-m", "experiments.b7_train", "--model", m, "--seed", str(s),
                                  "--batch", str(b), "--accum", str(acc)],
                                 cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"train_{m}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(10)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--slots", default="1,2,4,5,5,5")
    a = ap.parse_args()
    if a.all:
        run_all(a.slots.split(","), a.batch, a.accum)
    else:
        train_one(a.model, a.seed, a.batch, a.accum)
