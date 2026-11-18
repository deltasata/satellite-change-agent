"""B2, Part B final: predictions of the 5 seeds of each chosen configuration, with 1 and with 8 views.

The 8 views are the tile, its 3 rotations by 90 degrees, and the mirror image of each; each prediction
is turned back and the 8 are averaged. Saved per seed: runs/b2/runs/<name>/prob_views8.npz.
(The 1-view predictions were saved by the training run itself: prob.npz.)

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b2_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b2_common as B  # noqa: E402
from experiments.b2_train import Net, predict, run_name  # noqa: E402


def main() -> None:
    dev = torch.device("cuda")
    choice = json.loads((B.RUNS / "partB_choice.json").read_text())
    parts = B.official_parts()
    tiles = parts["valid"] + parts["test"] + parts["bolivia"]
    for j in choice.values():
        data = {}
        for s in range(5):
            d = B.RUNS / "runs" / run_name(j["model"], j["inp"], "grd", "official", s, j.get("loss", "ce"), j.get("lr"))
            if (d / "prob_views8.npz").exists():
                continue
            if not data:
                data = {c: B.scored(c, j["inp"], "grd")[0] for c in tiles}
            net = Net(j["model"], j["inp"], "grd")
            net.load_state_dict({k: v.float() for k, v in torch.load(d / "weights.pt", map_location="cpu").items()})
            net = net.to(dev).eval()
            out = {c: np.round(predict(net, data[c], dev, views=8) * 255).astype(np.uint8) for c in tiles}
            np.savez_compressed(d / "prob_views8.npz", **out)
            print(f"  {d.name}: 8-view predictions saved", flush=True)


if __name__ == "__main__":
    main()
