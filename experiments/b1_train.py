"""B1, the trained models: fine-tune one model leaving one flood event out, then predict that event.

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b1_train --model prithvi --event Spain
    .venv-ml/bin/python -m experiments.b1_train --run-all --gpus 1,2,4,5      # all models x all events

Models (settings fixed in docs/METHODS.md section 3 before any score):
  prithvi    Prithvi-EO-2.0-300M-TL (NASA/IBM) + UperNet decoder: the published flood setup
  terramind  TerraMind-1.0-base (IBM/ESA) + UperNet decoder
  unet       a U-Net with a ResNet-18 encoder, trained from nothing (no pretrained numbers)

Writes runs/b1/folds/<model>/<event>.json (per-tile counts and the training loss curve) and
runs/b1/weights/<model>/<event>.pt (needed later, when the exam question of that event is answered).
"""

from __future__ import annotations

import argparse
import json
import math
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
from experiments import b1_common as C  # noqa: E402

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
# Each pretrained model is fed inputs scaled as in its pretraining (values given there in units of 1e-4 reflectance).
STATS = {
    "prithvi": ([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]),
    "terramind": ([1503.317, 1718.197, 1853.91, 3132.22, 2424.884, 1857.648],
                  [2141.107, 2038.973, 2134.138, 1753.829, 1434.261, 1334.311]),
}
STATS["unet"] = STATS["prithvi"]
SETTINGS = {  # fixed in advance
    "prithvi": dict(lr=5e-5, epochs=50),
    "terramind": dict(lr=5e-5, epochs=50),
    "unet": dict(lr=1e-3, epochs=100),
}
BATCH, CROP, STRIDE = 16, 224, 144


def build(model: str):
    if model == "unet":
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet18", encoder_weights=None, in_channels=6, classes=2)
    from terratorch.models import EncoderDecoderFactory
    common = dict(task="segmentation", decoder="UperNetDecoder", decoder_channels=256, num_classes=2,
                  rescale=True, head_dropout=0.1, backbone_pretrained=True)
    if model == "prithvi":
        return EncoderDecoderFactory().build_model(
            backbone="prithvi_eo_v2_300_tl", backbone_bands=BANDS, **common,
            necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]}, {"name": "ReshapeTokensToImage"},
                   {"name": "LearnedInterpolateToPyramidal"}])
    if model == "terramind":
        return EncoderDecoderFactory().build_model(
            backbone="terramind_v1_base", backbone_modalities=["S2L2A"], backbone_bands={"S2L2A": BANDS}, **common,
            necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                   {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                   {"name": "LearnedInterpolateToPyramidal"}])
    raise ValueError(model)


def forward(net, x):
    out = net(x)
    return out.output if hasattr(out, "output") else out


def prepare(chip: str, kind: str, model: str):
    """(x, y): standardised 6-band image with NaN -> 0, and the label with -1 wherever not scored."""
    s = C.scored(chip, kind)
    if s is None:
        return None
    x, y, m = s
    mean, std = (np.array(v, np.float32)[:, None, None] * 1e-4 for v in STATS[model])
    x = np.nan_to_num((x[C.MODEL_BANDS] - mean) / std, nan=0.0).astype(np.float32)
    y = np.where(m, y, -1).astype(np.int64)
    return x, y


def augment(x, y, rng):
    i, j = rng.integers(0, x.shape[1] - CROP + 1), rng.integers(0, x.shape[2] - CROP + 1)
    x, y = x[:, i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP]
    k = rng.integers(4)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


@torch.no_grad()
def predict(net, x: np.ndarray, dev) -> np.ndarray:
    """Water probability on a full tile, from CROP windows at STRIDE, overlaps averaged."""
    H, W = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    prob, hits = np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    wins = [(i, j) for i in starts(H) for j in starts(W)]
    xb = torch.from_numpy(np.stack([x[:, i:i + CROP, j:j + CROP] for i, j in wins])).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(forward(net, xb).float(), dim=1)[:, 1].cpu().numpy()
    for (i, j), pi in zip(wins, p):
        prob[i:i + CROP, j:j + CROP] += pi
        hits[i:i + CROP, j:j + CROP] += 1
    return prob / hits


def train_one(model: str, event: str, kind: str = "l2a", seed: int = 0) -> dict:
    t0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    chips = s1f.chips()
    data = {c: d for c in chips if (d := prepare(c, kind, model)) is not None}
    train = [c for c in data if s1f.event(c) != event]
    test = [c for c in data if s1f.event(c) == event]
    net = build(model).to(dev)
    cfg = SETTINGS[model]
    steps_per_epoch = math.ceil(len(train) / BATCH)
    total = cfg["epochs"] * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    losses = []
    net.train()
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(train)
        run = []
        for k in range(steps_per_epoch):
            batch = [augment(*data[c], rng) for c in order[k * BATCH:(k + 1) * BATCH]]
            xb = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
            yb = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            if (yb >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(forward(net, xb).float(), yb, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run.append(loss.item())
        losses.append(float(np.mean(run)) if run else float("nan"))
        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  [{model} {event}] epoch {epoch + 1}/{cfg['epochs']} loss {losses[-1]:.4f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
    net.eval()
    per_chip, preds = {}, {}
    for c in test:
        x, _ = data[c]
        p = predict(net, x, dev)
        _, y, m = C.scored(c, kind)
        per_chip[c] = C.counts(p > 0.5, y, m)
        preds[c] = (p * 255).astype(np.uint8)
    out = {"model": model, "event": event, "kind": kind, "n_train_tiles": len(train), "n_test_tiles": len(test),
           "settings": cfg, "seconds": round(time.time() - t0), "loss_per_epoch": losses, "counts": per_chip,
           "iou": C.pooled_iou(per_chip, test) if test else None}
    fold = C.RUNS / "folds" / model
    fold.mkdir(parents=True, exist_ok=True)
    (fold / f"{event}.json").write_text(json.dumps(out))
    np.savez_compressed(fold / f"{event}_prob.npz", **preds)
    wdir = C.RUNS / "weights" / model
    wdir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               wdir / f"{event}.pt")
    print(f"  [{model} {event}] done: {len(test)} test tiles, IoU {out['iou'] if out['iou'] is None else round(out['iou'], 3)}, "
          f"{out['seconds']} s", flush=True)
    return out


def run_all(gpus: list[str], models: list[str]) -> None:
    """Every (model, event) pair once; one job per GPU at a time; finished pairs are skipped."""
    from core.data.sen1floods11_l2a import OUT
    events = sorted({s1f.event(p.stem) for p in OUT.glob("*.npz")})
    jobs = [(m, e) for m in models for e in events if not (C.RUNS / "folds" / m / f"{e}.json").exists()]
    print(f"{len(jobs)} jobs on GPUs {gpus}")
    running: dict[str, subprocess.Popen] = {}
    logs = C.RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    while jobs or running:
        for g in gpus:
            if g in running and running[g].poll() is not None:
                del running[g]
            if g not in running and jobs:
                m, e = jobs.pop(0)
                running[g] = subprocess.Popen(
                    [sys.executable, "-m", "experiments.b1_train", "--model", m, "--event", e], cwd=ROOT,
                    env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": g},
                    stdout=open(logs / f"{m}_{e}.log", "w"), stderr=subprocess.STDOUT)
                print(f"  GPU {g}: {m} {e}", flush=True)
        time.sleep(20)
    print("all done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SETTINGS))
    ap.add_argument("--event", help="the event left out; ALL = train on every event (the model the pipeline uses)")
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--gpus", default="1,2,4,5")
    ap.add_argument("--models", default="unet,prithvi,terramind")
    a = ap.parse_args()
    if a.run_all:
        run_all(a.gpus.split(","), a.models.split(","))
    else:
        train_one(a.model, a.event)
