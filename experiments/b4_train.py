"""B4, the trained change detectors (protocol: docs/METHODS.md section 7, detectors 3 and 4).

One run = one network, one left-out group of cities, one seed. Training uses the median images of every
kept city outside that group (data_cache/oscd/pc/inputs, made by `python -m experiments.b4_common`).
Settings fixed in the protocol: 7 bands scaled with TerraMind's Level-2A statistics, missing values 0 and
left out of the loss; random 224 x 224 crops, flips and 90 degree rotations (the same for both images);
batch 8; AdamW, weight decay 0.05; learning rate 1e-3 (FC-Siam-diff) or 5e-5 (TerraMind), cosine
decrease; 3 000 steps; loss = cross-entropy + Dice loss of "changed"; bf16; no early stopping.

Prediction: the left-out group's cities, from 224 windows at stride 144, overlaps averaged; the
probability of "changed" is saved (x 255, uint8). b4_report averages the 3 seeds and marks "changed"
where the mean is above 0.5.

Group "ALL" trains on every kept city (the networks the pipeline uses); its weights are kept, and so are
those of group "paris_bercy" (the exam's Paris question is answered without the Paris cities).

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b4_train --model fsd --group beirut --seed 0
    .venv-ml/bin/python -m experiments.b4_train --all --gpus 1,2,4,5      # every run, 2 per GPU
Writes runs/b4/runs/<model>_<group>_s<seed>/ : result.json, prob.npz (and weights.pt for ALL, paris_bercy).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data.oscd_pc import group as group_of  # noqa: E402
from experiments import b4_common as C  # noqa: E402

BATCH, CROP, STRIDE, STEPS = 8, 224, 144, 3000
LR = {"fsd": 1e-3, "terramind": 5e-5}
KEEP_WEIGHTS = ("ALL", "paris_bercy")


def build(model: str) -> torch.nn.Module:
    if model == "fsd":
        from core.models.siamese import FCSiamDiff
        return FCSiamDiff(in_ch=7)
    from terratorch.models import EncoderDecoderFactory
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone="terramind_v1_base", backbone_modalities=["S2L2A"],
        backbone_bands={"S2L2A": C.TM_BANDS}, backbone_pretrained=True,
        backbone_use_temporal=True, backbone_temporal_pooling="diff", backbone_temporal_n_timestamps=2,
        decoder="UperNetDecoder", decoder_channels=256, num_classes=2, rescale=True, head_dropout=0.1,
        necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
               {"name": "ReshapeTokensToImage", "remove_cls_token": False},
               {"name": "LearnedInterpolateToPyramidal"}])


class Net(torch.nn.Module):
    """Both networks as f(before, after) -> logits (B, 2, H, W)."""

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.net = build(model)

    def forward(self, before, after):
        if self.model == "fsd":
            return self.net(before, after)
        out = self.net(torch.stack([before, after], dim=2))   # (B, C, T=2, H, W)
        return out.output if hasattr(out, "output") else out


def load(cities: list[str]) -> dict:
    data = {}
    for c in cities:
        z = C.inputs(c)
        ok = np.isfinite(z["before"]).all(0) & np.isfinite(z["after"]).all(0)
        y = np.where(ok, z["label"], -1).astype(np.int8)
        data[c] = (C.scale(z["before"]), C.scale(z["after"]), y)
    return data


def crops(data: dict, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = list(data)
    w = np.array([data[c][2].size for c in names], float)
    B, A, Y = [], [], []
    for _ in range(BATCH):
        b, a, y = data[names[rng.choice(len(names), p=w / w.sum())]]
        i, j = rng.integers(0, y.shape[0] - CROP + 1), rng.integers(0, y.shape[1] - CROP + 1)
        b, a, y = b[:, i:i + CROP, j:j + CROP], a[:, i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP]
        k = rng.integers(4)
        b, a, y = np.rot90(b, k, (1, 2)), np.rot90(a, k, (1, 2)), np.rot90(y, k)
        if rng.random() < 0.5:
            b, a, y = b[:, :, ::-1], a[:, :, ::-1], y[:, ::-1]
        B.append(np.ascontiguousarray(b)), A.append(np.ascontiguousarray(a)), Y.append(np.ascontiguousarray(y))
    return np.stack(B), np.stack(A), np.stack(Y).astype(np.int64)


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy + Dice loss of "changed", both over the pixels with a label (y >= 0)."""
    ce = F.cross_entropy(logits, y, ignore_index=-1)
    m = (y >= 0).float()
    p = torch.softmax(logits, 1)[:, 1] * m
    t = (y == 1).float() * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


@torch.no_grad()
def predict(net, z: dict, dev) -> np.ndarray:
    """Probability of "changed" (H, W) from CROP windows at STRIDE, overlaps averaged."""
    b, a = C.scale(z["before"]), C.scale(z["after"])
    H, W = b.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(H) for j in starts(W)]
    prob, hits = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for k in range(0, len(wins), 16):
        part = wins[k:k + 16]
        xb = torch.from_numpy(np.stack([b[:, i:i + CROP, j:j + CROP] for i, j in part])).to(dev)
        xa = torch.from_numpy(np.stack([a[:, i:i + CROP, j:j + CROP] for i, j in part])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(net(xb, xa).float(), dim=1)[:, 1].cpu().numpy()
        for (i, j), pi in zip(part, p):
            prob[i:i + CROP, j:j + CROP] += pi
            hits[i:i + CROP, j:j + CROP] += 1
    return prob / hits


def train_one(model: str, group: str, seed: int) -> dict:
    name = f"{model}_{group}_s{seed}"
    out_dir = C.RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    kept = C.kept()
    test = [c for c in kept if group_of(c) == group]
    train = [c for c in kept if group_of(c) != group]
    data = load(train)
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
        if (step + 1) % 250 == 0:
            losses.append(float(np.mean(run)))
            run = []
            print(f"  [{name}] step {step + 1}/{STEPS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = {c: np.round(predict(net, C.inputs(c), dev) * 255).astype(np.uint8) for c in test}
    np.savez_compressed(out_dir / "prob.npz", **probs)
    if group in KEEP_WEIGHTS:
        torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
                   out_dir / "weights.pt")
    res = {"name": name, "model": model, "group": group, "seed": seed, "lr": LR[model], "steps": STEPS,
           "train_cities": train, "test_cities": test, "seconds": round(time.time() - t0),
           "loss_per_250_steps": losses, "parameters": sum(p.numel() for p in net.parameters())}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def run_all(gpus: list[str], per_gpu: int = 2) -> None:
    groups = sorted({group_of(c) for c in C.kept()}) + ["ALL"]
    jobs = [(m, g, s) for m in ("terramind", "fsd") for g in groups for s in range(3)
            if not (C.RUNS / "runs" / f"{m}_{g}_s{s}" / "result.json").exists()]
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
            (m, gr, s), g = jobs.pop(0), free.pop(0)
            p = subprocess.Popen([sys.executable, "-m", "experiments.b4_train", "--model", m, "--group", gr, "--seed", str(s)],
                                 cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"train_{m}_{gr}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(10)
    print("all runs finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(LR))
    ap.add_argument("--group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--gpus", default="1,2,4,5")
    a = ap.parse_args()
    if a.all:
        run_all(a.gpus.split(","))
    else:
        train_one(a.model, a.group, a.seed)
