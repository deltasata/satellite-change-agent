"""B2, the trained networks (protocol: docs/METHODS.md section 4).

One run = one network, one input set, one kind of radar image, one fold, one seed:
    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b2_train --model unet18 --inp s1 --kind rtc --fold Spain
    ... --kind grd --fold official          # Part B: train on the official training part
    .venv-ml/bin/python -m experiments.b2_train --plan A --gpus 1,2,4,5   # every Part A run, 2 per GPU

Models:
  unet18, unet34   U-Net with a ResNet-18 / ResNet-34 encoder, trained from nothing
  terramind        TerraMind-1.0-base + UperNet decoder (modality S1RTC or S1GRD for radar, S2L1C for optical)

Folds:
  <event>   Part A: train on every other event, predict this one
  ALL       train on every event (the model the pipeline would use)
  official  Part B: train on the official training part, predict the validation, test and Bolivia parts

Writes runs/b2/runs/<name>/ : result.json (counts per predicted tile, loss per epoch, settings),
prob.npz (water probability x 255 per predicted tile), weights.pt (bf16).
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import sen1floods11 as s1f  # noqa: E402
from experiments import b2_common as B  # noqa: E402

BATCH, CROP, STRIDE = 16, 224, 144
DEFAULTS = {"unet18": dict(lr=1e-3, epochs=100), "unet34": dict(lr=1e-3, epochs=100),
            "terramind": dict(lr=5e-5, epochs=50)}
RADAR_MODALITY = {"rtc": "S1RTC", "grd": "S1GRD"}


def run_name(model, inp, kind, fold, seed, loss="ce", lr=None) -> str:
    lr = lr or DEFAULTS[model]["lr"]
    return f"{model}_{inp}_{kind}_{loss}_lr{lr:g}_{fold}_s{seed}"


def build(model: str, inp: str, kind: str):
    if model.startswith("unet"):
        import segmentation_models_pytorch as smp
        enc = {"unet18": "resnet18", "unet34": "resnet34"}[model]
        return smp.Unet(enc, encoder_weights=None, in_channels=B.CHANNELS[inp], classes=2)
    from terratorch.models import EncoderDecoderFactory
    mods = {"s1": [RADAR_MODALITY[kind]], "s2": ["S2L1C"], "s1s2": [RADAR_MODALITY[kind], "S2L1C"]}[inp]
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone="terramind_v1_base", backbone_modalities=mods, backbone_pretrained=True,
        decoder="UperNetDecoder", decoder_channels=256, num_classes=2, rescale=True, head_dropout=0.1,
        necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
               {"name": "ReshapeTokensToImage", "remove_cls_token": False},
               {"name": "LearnedInterpolateToPyramidal"}])


class Net(torch.nn.Module):
    """Takes the stacked input (radar channels first) and hands TerraMind a dictionary when it has two modalities."""

    def __init__(self, model, inp, kind):
        super().__init__()
        self.net = build(model, inp, kind)
        self.split = (model == "terramind" and inp == "s1s2")
        self.mods = [RADAR_MODALITY[kind], "S2L1C"]

    def forward(self, x):
        if self.split:
            x = {self.mods[0]: x[:, :2], self.mods[1]: x[:, 2:]}
        out = self.net(x)
        return out.output if hasattr(out, "output") else out


def loss_fn(logits, y, kind: str):
    ce = F.cross_entropy(logits, y, ignore_index=-1)
    if kind == "ce":
        return ce
    m = (y >= 0).float()
    p = torch.softmax(logits, 1)[:, 1] * m
    t = (y == 1).float() * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return ce + dice


def augment(x, y, rng):
    i, j = rng.integers(0, x.shape[1] - CROP + 1), rng.integers(0, x.shape[2] - CROP + 1)
    x, y = x[:, i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP]
    k = rng.integers(4)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


@torch.no_grad()
def predict(net, x: np.ndarray, dev, views: int = 1) -> np.ndarray:
    """Water probability on a full tile from CROP windows at STRIDE, overlaps averaged.
    views = 8: also the 3 rotations and the 4 mirrored versions, each turned back, averaged."""
    H, W = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(H) for j in starts(W)]
    xb = torch.from_numpy(np.stack([x[:, i:i + CROP, j:j + CROP] for i, j in wins])).to(dev)
    acc = torch.zeros(len(wins), CROP, CROP, device=dev)
    for v in range(views):
        k, flip = v % 4, v >= 4
        t = torch.rot90(xb, k, (2, 3))
        if flip:
            t = torch.flip(t, (3,))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(net(t).float(), dim=1)[:, 1]
        if flip:
            p = torch.flip(p, (2,))
        acc += torch.rot90(p, -k, (1, 2))
    acc = (acc / views).cpu().numpy()
    prob, hits = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    for (i, j), pi in zip(wins, acc):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return prob / hits


def split(fold: str, data: dict) -> tuple[list[str], list[str]]:
    if fold == "official":
        parts = B.official_parts()
        return [c for c in parts["train"] if c in data], [c for p in ("valid", "test", "bolivia") for c in parts[p] if c in data]
    if fold == "ALL":
        return list(data), []
    return [c for c in data if s1f.event(c) != fold], [c for c in data if s1f.event(c) == fold]


def train_one(model, inp, kind, fold, seed=0, loss="ce", lr=None, epochs=None) -> dict:
    name = run_name(model, inp, kind, fold, seed, loss, lr)
    out_dir = B.RUNS / "runs" / name
    if (out_dir / "result.json").exists():
        print(f"  {name}: done before")
        return json.loads((out_dir / "result.json").read_text())
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = {}
    for c in s1f.chips():
        s = B.scored(c, inp, kind)
        if s is not None:
            x, y, m = s
            data[c] = (x, np.where(m, y, -1).astype(np.int64))
    train, test = split(fold, data)
    net = Net(model, inp, kind).to(dev)
    lr = lr or DEFAULTS[model]["lr"]
    epochs = epochs or DEFAULTS[model]["epochs"]
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
            if len(batch) < 2:      # a last batch of one tile: the decoder's normalisation needs two
                continue
            xb = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
            yb = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            if (yb >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xb).float()
            lo = loss_fn(logits, yb, loss)
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
    res = {"name": name, "model": model, "inp": inp, "kind": kind, "fold": fold, "seed": seed, "loss": loss,
           "lr": lr, "epochs": epochs, "n_train_tiles": len(train), "n_predicted_tiles": len(test),
           "seconds": round(time.time() - t0), "loss_per_epoch": losses, "counts": per_chip}
    np.savez_compressed(out_dir / "prob.npz", **probs)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


# ---------------------------------------------------------------- job lists and the scheduler

def plan_a() -> list[dict]:
    events = sorted({s1f.event(p.stem) for p in (s1f.DIR / "S1RTC").glob("*.npz")})
    return [dict(model=m, inp="s1", kind="rtc", fold=e, seed=s)
            for s in range(3) for m in ("unet18", "terramind") for e in events]


def plan_b_select() -> list[dict]:
    jobs = [dict(model=m, inp=i, kind="grd", fold="official", seed=0, loss=l)
            for i in ("s1", "s2", "s1s2") for m in ("unet18", "unet34") for l in ("ce", "cedice")]
    jobs += [dict(model="terramind", inp=i, kind="grd", fold="official", seed=0, lr=lr)
             for i in ("s1", "s2", "s1s2") for lr in (5e-5, 1e-4)]
    return jobs


def plan_a_all() -> list[dict]:
    """The Part A winner trained on every event, 3 seeds: the radar model the pipeline uses for new places."""
    return [dict(model="terramind", inp="s1", kind="rtc", fold="ALL", seed=s) for s in range(3)]


def plan_b_final() -> list[dict]:
    """Seeds 1-4 of the configuration chosen on validation, per family and input set."""
    choice = json.loads((B.RUNS / "partB_choice.json").read_text())
    return [{**j, "seed": s} for j in choice.values() for s in range(1, 5)]


def done(job) -> bool:
    keys = ("model", "inp", "kind", "fold", "seed", "loss", "lr")
    name = run_name(*(job.get(k) for k in keys[:5]), job.get("loss", "ce"), job.get("lr"))
    return (B.RUNS / "runs" / name / "result.json").exists()


def run_jobs(jobs: list[dict], gpus: list[str], per_gpu: int = 2) -> None:
    jobs = [j for j in jobs if not done(j)]
    print(f"{len(jobs)} runs on GPUs {gpus}, {per_gpu} at a time per GPU", flush=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running: dict[int, subprocess.Popen] = {}
    logs = B.RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or running:
        for k, g in enumerate(slots):
            if k in running and running[k].poll() is not None:
                del running[k]
            if k not in running and jobs:
                j = jobs.pop(0)
                args = [sys.executable, "-m", "experiments.b2_train"]
                for key, val in j.items():
                    args += [f"--{key}", str(val)]
                name = run_name(j["model"], j["inp"], j["kind"], j["fold"], j["seed"], j.get("loss", "ce"), j.get("lr"))
                running[k] = subprocess.Popen(args, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": g},
                                              stdout=open(logs / f"{name}.log", "w"), stderr=subprocess.STDOUT)
                print(f"  GPU {g}: {name}", flush=True)
        time.sleep(15)
    print("all done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(DEFAULTS))
    ap.add_argument("--inp", choices=list(B.CHANNELS), default="s1")
    ap.add_argument("--kind", choices=["rtc", "grd"], default="rtc")
    ap.add_argument("--fold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "cedice"], default="ce")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--plan", help="A, B-select, or both joined by a comma")
    ap.add_argument("--gpus", default="1,2,4,5")
    ap.add_argument("--per-gpu", type=int, default=2)
    a = ap.parse_args()
    if a.plan:
        plans = {"A": plan_a, "A-all": plan_a_all, "B-select": plan_b_select, "B-final": plan_b_final}
        run_jobs([j for p in a.plan.split(",") for j in plans[p]()], a.gpus.split(","), a.per_gpu)
    else:
        train_one(a.model, a.inp, a.kind, a.fold, a.seed, a.loss, a.lr, a.epochs)
