"""B8: every detector's map of "burnt" on the new fires of 2026 (protocol: docs/METHODS.md section 11).

    .venv-ml/bin/python -m experiments.b8_predict unet      # B5's U-Net (3 seeds), on the two medians
    .venv-ml/bin/python -m experiments.b8_predict utae      # B7's U-TAE (3 seeds), on every clear optical date
    .venv-ml/bin/python -m experiments.b8_predict utae_rd   # B8's U-TAE with radar and elevation (3 seeds)
TerraMind-base-Fire: experiments/b8_tmfire.py. Writes runs/b8/cems2026/<model>_s<seed>.npz (probability x 255
per patch). Only patches the B5 copy kept (not dropped) are predicted; the GPU is CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b5_common as C  # noqa: E402
from experiments import b8_train as T  # noqa: E402

OUT = T.RUNS / "cems2026"
CEMS = T.SOURCES["cems2026"]


def patches() -> list[str]:
    """New-fire patches that the B5 copy kept, with an optical series and a radar file."""
    names = [p["name"] for p in json.loads((CEMS / "patches.json").read_text())]
    out = []
    for n in names:
        f = CEMS / "es" / f"{n}.npz"
        if f.exists() and "dropped" not in np.load(f).files:
            out.append(n)
    return out


def load_es(name: str) -> dict:
    z = np.load(CEMS / "es" / f"{name}.npz")
    out = {k: z[k] for k in z.files}
    for k in ("before", "after"):
        out[k] = out[k].astype(np.float32)
    return out


@torch.no_grad()
def run(model: str) -> None:
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    names = patches()
    for s in range(3):
        dest = OUT / f"{model}_s{s}.npz"
        if dest.exists():
            continue
        if model == "unet":
            from experiments.b5_train import Net, predict
            net = Net("unet")
            w = C.RUNS / "runs" / f"unet_s{s}" / "weights.pt"
            f = lambda n: predict(net, load_es(n), dev)          # noqa: E731
        elif model == "utae":
            from core.models.utae import UTAE
            from experiments import b7_train as B7
            net = UTAE()
            w = B7.RUNS / "runs" / f"utae_s{s}" / "weights.pt"

            def f(n):
                meta = json.loads((CEMS / "ts" / f"{n}.json").read_text())
                x = B7.inputs(np.load(CEMS / "ts" / f"{n}.npy"))
                d, p = np.array(meta["days"]), np.array(meta["period"])
                X = torch.from_numpy(np.stack([x[:, :, i:i + T.CROP, j:j + T.CROP] for i, j in T.CORNERS])).to(dev)
                D = torch.from_numpy(np.repeat(d[None], 4, 0)).to(dev)
                P = torch.from_numpy(np.repeat(p[None], 4, 0)).to(dev)
                pad = torch.zeros(D.shape, dtype=torch.bool, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = torch.softmax(net(X, D, P, pad).float(), dim=1)[:, 1].cpu().numpy()
                prob, hits = np.zeros((256, 256), np.float32), np.zeros((256, 256), np.float32)
                for (i, j), pi in zip(T.CORNERS, pr):
                    prob[i:i + T.CROP, j:j + T.CROP] += pi
                    hits[i:i + T.CROP, j:j + T.CROP] += 1
                return prob / hits
        else:
            net = T.network()
            w = T.RUNS / "runs" / f"utae_rd_s{s}" / "weights.pt"
            f = lambda n: T.predict(net, n, "cems2026", dev)     # noqa: E731
        state = torch.load(w, map_location="cpu")
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
        net.to(dev).eval()
        probs = {n: np.round(f(n) * 255).astype(np.uint8) for n in names}
        np.savez_compressed(dest, **probs)
        print(f"{model}_s{s}: {len(probs)} patches", flush=True)


if __name__ == "__main__":
    run(sys.argv[1])
