"""B5, the trained burn detectors (protocol: docs/METHODS.md section 8, detectors 3 and 4).

One run = one network, one seed, trained on the training and validation fires (without the ten fires at
exam places), tested on the 42 held-out fires. The same runs are the networks the pipeline uses, since no
exam fire is in their training.
  - "unet":      U-Net from nothing (segmentation_models_pytorch, ResNet-18 encoder as in B1), the before
                 and after median images stacked as 14 input bands.
  - "terramind": B4's Siamese TerraMind-1.0-base (feature difference, UperNet decoder), fine-tuned.
Settings fixed in the protocol: 7 bands with TerraMind's Level-2A scaling, missing values 0 and left out
of the loss (and label -1); random 224 x 224 crops of the 256 x 256 patches, flips and 90 degree
rotations; batch 8; AdamW, weight decay 0.05; learning rate 1e-3 (U-Net) or 5e-5 (TerraMind), cosine
decrease; 10 000 steps; cross-entropy + Dice loss of "burnt"; bf16; no early stopping.
Prediction on a patch: the four corner windows of 224, averaged where they overlap.

    python -m experiments.b5_common pack                                     # once, before training
    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b5_train --model unet --seed 0
    .venv-ml/bin/python -m experiments.b5_train --all --gpus 1,2,4           # the 6 runs, 2 per GPU
Writes runs/b5/runs/<model>_s<seed>/ : result.json, prob.npz (held-out patches, x 255) and weights.pt.
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
from experiments import b5_common as C  # noqa: E402
from experiments.b4_train import loss_fn  # noqa: E402  (cross-entropy + Dice, label -1 ignored)

BATCH, CROP, STEPS = 8, 224, 10000
LR = {"unet": 1e-3, "terramind": 5e-5}
CORNERS = ((0, 0), (0, 32), (32, 0), (32, 32))


class Net(torch.nn.Module):
    """Both networks as f(before, after) -> logits (B, 2, H, W)."""

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        if model == "unet":
            import segmentation_models_pytorch as smp
            self.net = smp.Unet("resnet18", encoder_weights=None, in_channels=14, classes=2)
        else:
            from experiments.b4_train import build
            self.net = build("terramind")

    def forward(self, before, after):
        if self.model == "unet":
            return self.net(torch.cat([before, after], dim=1))
        out = self.net(torch.stack([before, after], dim=2))   # (B, C, T=2, H, W)
        return out.output if hasattr(out, "output") else out


def packed():
    P = C.PACKED
    return (np.load(P / "train_before.npy", mmap_mode="r"), np.load(P / "train_after.npy", mmap_mode="r"),
            np.load(P / "train_y.npy", mmap_mode="r"))


def crops(data, rng):
    Bm, Am, Ym = data
    B, A, Y = [], [], []
    for k in rng.integers(0, len(Ym), BATCH):
        i, j = rng.integers(0, 256 - CROP + 1, 2)
        b = C.scale(np.asarray(Bm[k, :, i:i + CROP, j:j + CROP], np.float32))
        a = C.scale(np.asarray(Am[k, :, i:i + CROP, j:j + CROP], np.float32))
        y = np.asarray(Ym[k, i:i + CROP, j:j + CROP])
        r = rng.integers(4)
        b, a, y = np.rot90(b, r, (1, 2)), np.rot90(a, r, (1, 2)), np.rot90(y, r)
        if rng.random() < 0.5:
            b, a, y = b[:, :, ::-1], a[:, :, ::-1], y[:, ::-1]
        B.append(np.ascontiguousarray(b)), A.append(np.ascontiguousarray(a)), Y.append(np.ascontiguousarray(y))
    return np.stack(B), np.stack(A), np.stack(Y).astype(np.int64)


@torch.no_grad()
def predict(net, z: dict, dev) -> np.ndarray:
    """Probability of "burnt" (256, 256): the four corner windows of 224, averaged where they overlap."""
    b, a = C.scale(z["before"]), C.scale(z["after"])
    xb = torch.from_numpy(np.stack([b[:, i:i + CROP, j:j + CROP] for i, j in CORNERS])).to(dev)
    xa = torch.from_numpy(np.stack([a[:, i:i + CROP, j:j + CROP] for i, j in CORNERS])).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(net(xb, xa).float(), dim=1)[:, 1].cpu().numpy()
    prob, hits = np.zeros((256, 256), np.float32), np.zeros((256, 256), np.float32)
    for (i, j), pi in zip(CORNERS, p):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return prob / hits


def train_one(model: str, seed: int) -> dict:
    name = f"{model}_s{seed}"
    out_dir = C.RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = packed()
    net = Net(model).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR[model], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    losses, run = [], []
    net.train()
    for step in range(STEPS):
        b, a, y = crops(data, rng)
        yb = torch.from_numpy(y).to(dev)
        if (yb >= 0).sum() == 0:
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(torch.from_numpy(b).to(dev), torch.from_numpy(a).to(dev)).float()
        lo = loss_fn(logits, yb)
        opt.zero_grad(set_to_none=True)
        lo.backward()
        opt.step()
        sched.step()
        run.append(lo.item())
        if (step + 1) % 500 == 0:
            losses.append(float(np.mean(run)))
            run = []
            print(f"  [{name}] step {step + 1}/{STEPS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    test = C.built("test")
    probs = {n: np.round(predict(net, C.load(n), dev) * 255).astype(np.uint8) for n in test}
    np.savez_compressed(out_dir / "prob.npz", **probs)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    res = {"name": name, "model": model, "seed": seed, "lr": LR[model], "steps": STEPS,
           "n_train_patches": int(len(data[2])), "n_test_patches": len(test), "seconds": round(time.time() - t0),
           "loss_per_500_steps": losses, "parameters": sum(p.numel() for p in net.parameters())}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def run_all(gpus: list[str], per_gpu: int = 2) -> None:
    jobs = [(m, s) for m in ("terramind", "unet") for s in range(3)
            if not (C.RUNS / "runs" / f"{m}_s{s}" / "result.json").exists()]
    logs = C.RUNS / "logs"
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
            (m, s), g = jobs.pop(0), free.pop(0)
            p = subprocess.Popen([sys.executable, "-m", "experiments.b5_train", "--model", m, "--seed", str(s)],
                                 cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"train_{m}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(10)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(LR))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default="1,2,4")
    a = ap.parse_args()
    if a.all:
        run_all(a.gpus.split(","))
    else:
        train_one(a.model, a.seed)
