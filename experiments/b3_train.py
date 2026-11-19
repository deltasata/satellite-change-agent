"""B3, the trained cloud detectors (protocol: docs/METHODS.md section 6, detectors 4 and 5).

One run = one network, one seed. Training: the 8 490 high-quality training images of CloudSEN12+, as
stored in the data set (Level-2A made by Google Earth Engine), the pipeline's 7 bands, 4 classes
(0 clear, 1 thick cloud, 2 thin cloud, 3 shadow). Settings fixed in the protocol: random 224 x 224 crops,
90 degree rotations and flips; batch 16; AdamW, weight decay 0.05; learning rate 1e-3 (U-Net) or 5e-5
(TerraMind), cosine decrease; 20 epochs; cross-entropy (unscored pixels ignored); bf16; no early stopping;
no choice made on validation or test images.

Prediction: on every test image, in both kinds (es, gee), from 224 windows at stride 144, overlaps
averaged; the class probabilities are saved (x 255, uint8). b3_report averages the 3 seeds' probabilities
and takes, per pixel, the most probable class (fixed before any score: the same "most probable class" that
OmniCloudMask outputs). "Unusable" = classes 1-3.

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b3_train --model unet18 --seed 0
    .venv-ml/bin/python -m experiments.b3_train --all --gpus 1,2,4      # 2 networks x 3 seeds, 2 per GPU
Writes runs/b3/runs/<model>_s<seed>/ : result.json, prob_<es|gee>.npz, weights.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b3_common as C  # noqa: E402

BATCH, CROP, STRIDE, EPOCHS = 16, 224, 144, 20
LR = {"unet18": 1e-3, "terramind": 5e-5}


def build(model: str):
    if model == "unet18":
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet18", encoder_weights=None, in_channels=7, classes=4)
    from terratorch.models import EncoderDecoderFactory
    return EncoderDecoderFactory().build_model(
        task="segmentation", backbone="terramind_v1_base", backbone_modalities=["S2L2A"],
        backbone_bands={"S2L2A": C.TM_BANDS}, backbone_pretrained=True,
        decoder="UperNetDecoder", decoder_channels=256, num_classes=4, rescale=True, head_dropout=0.1,
        necks=[{"name": "SelectIndices", "indices": [2, 5, 8, 11]},
               {"name": "ReshapeTokensToImage", "remove_cls_token": False},
               {"name": "LearnedInterpolateToPyramidal"}])


class Net(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.net = build(model)

    def forward(self, x):
        out = self.net(x)
        return out.output if hasattr(out, "output") else out


def load_train() -> tuple[list[str], dict]:
    """Training images kept as uint16 (x 10 000) and int8 labels, -1 where not scored: about 15 GB."""
    ids = C.ids("train")

    def one(i):
        y, m = C.scored(i, "train")
        return i, (C.load(i, "train")["x"], np.where(m, y, -1).astype(np.int8))

    with ThreadPoolExecutor(16) as pool:
        data = dict(pool.map(one, ids))
    return ids, data


def augment(x, y, rng):
    i, j = rng.integers(0, x.shape[1] - CROP + 1), rng.integers(0, x.shape[2] - CROP + 1)
    x, y = x[:, i:i + CROP, j:j + CROP], y[i:i + CROP, j:j + CROP]
    k = rng.integers(4)
    x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    return C.scale(C.reflectance(np.ascontiguousarray(x))), np.ascontiguousarray(y).astype(np.int64)


@torch.no_grad()
def predict(net, x: np.ndarray, dev) -> np.ndarray:
    """Class probabilities (4, H, W) from CROP windows at STRIDE, overlaps averaged."""
    x = C.scale(x)
    H, W = x.shape[1:]
    starts = lambda n: sorted(set(list(range(0, n - CROP + 1, STRIDE)) + [n - CROP]))
    wins = [(i, j) for i in starts(H) for j in starts(W)]
    xb = torch.from_numpy(np.stack([x[:, i:i + CROP, j:j + CROP] for i, j in wins])).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = torch.softmax(net(xb).float(), dim=1).cpu().numpy()
    prob, hits = np.zeros((4, H, W), np.float32), np.zeros((H, W), np.float32)
    for (i, j), pi in zip(wins, p):
        prob[:, i:i + CROP, j:j + CROP] += pi
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
    train, data = load_train()
    print(f"  [{name}] {len(train)} training images loaded ({time.time() - t0:.0f} s)", flush=True)
    net = Net(model).to(dev)
    steps_per_epoch = math.ceil(len(train) / BATCH)
    opt = torch.optim.AdamW(net.parameters(), lr=LR[model], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    losses = []
    net.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(train)
        run = []
        for k in range(steps_per_epoch):
            batch = [augment(*data[i], rng) for i in order[k * BATCH:(k + 1) * BATCH]]
            if len(batch) < 2:      # a group of one image: the decoder's batch normalisation needs two (B2)
                continue
            xb = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
            yb = torch.from_numpy(np.stack([b[1] for b in batch])).to(dev)
            if (yb >= 0).sum() == 0:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(xb).float()
            lo = F.cross_entropy(logits, yb, ignore_index=-1)
            opt.zero_grad(set_to_none=True)
            lo.backward()
            opt.step()
            sched.step()
            run.append(lo.item())
        losses.append(float(np.mean(run)) if run else float("nan"))
        print(f"  [{name}] epoch {epoch + 1}/{EPOCHS} loss {losses[-1]:.4f} ({time.time() - t0:.0f} s)", flush=True)
    del data
    net.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    test = C.ids("test")
    for kind in ("es", "gee"):
        probs = {i: np.round(predict(net, C.image(i, "test", kind), dev) * 255).astype(np.uint8) for i in test}
        np.savez_compressed(out_dir / f"prob_{kind}.npz", **probs)
    torch.save({k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in net.state_dict().items()},
               out_dir / "weights.pt")
    res = {"name": name, "model": model, "seed": seed, "lr": LR[model], "epochs": EPOCHS,
           "n_train_images": len(train), "n_test_images": len(test), "seconds": round(time.time() - t0),
           "loss_per_epoch": losses}
    (out_dir / "result.json").write_text(json.dumps(res))   # written last: marks the run as finished
    print(f"  [{name}] done in {res['seconds']} s", flush=True)
    return res


def run_all(gpus: list[str], per_gpu: int = 2) -> None:
    jobs = [(m, s) for m in ("terramind", "unet18") for s in range(3)
            if not (C.RUNS / "runs" / f"{m}_s{s}" / "result.json").exists()]
    logs = C.RUNS / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = []
    while jobs or running:
        running = [(p, g) for p, g in running if p.poll() is None]
        free = [g for g in slots if sum(g == rg for _, rg in running) < slots.count(g)]
        while jobs and free:
            (m, s), g = jobs.pop(0), free.pop(0)
            print(f"  GPU {g}: {m}_s{s}", flush=True)
            p = subprocess.Popen([sys.executable, "-m", "experiments.b3_train", "--model", m, "--seed", str(s)],
                                 cwd=ROOT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": g},
                                 stdout=open(logs / f"train_{m}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            running.append((p, g))
        time.sleep(20)


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
