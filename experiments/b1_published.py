"""B1, detector 6: the published Prithvi-EO-2.0 Sen1Floods11 model, used as it is.

It was fine-tuned by NASA/IBM on Sen1Floods11's L1C images of the official training part (and chosen on
the validation part), so it is scored only on the official test part and Bolivia. Its inputs are scaled
with the Sen1Floods11 statistics of TerraTorch's data module, as in its training.

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b1_published
Writes runs/b1/counts_<kind>_published.json (test and Bolivia tiles only).
"""

from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from terratorch.datamodules.sen1floods11 import MEANS, STDS

from core.data import sen1floods11 as s1f
from experiments import b1_common as C
from experiments.b1_train import BANDS, build, predict

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"


def main():
    net = build("prithvi")
    ck = torch.load(hf_hub_download(REPO, "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt"), map_location="cpu", weights_only=False)
    missing, unexpected = net.load_state_dict({k.removeprefix("model."): v for k, v in ck["state_dict"].items()}, strict=False)
    print(f"loaded; missing {len(missing)}, unexpected {len(unexpected)}")
    dev = torch.device("cuda")
    net = net.to(dev).eval()
    mean = np.array([MEANS[b] for b in BANDS], np.float32)[:, None, None]
    std = np.array([STDS[b] for b in BANDS], np.float32)[:, None, None]
    parts = s1f.splits()
    chips = [c for c in s1f.chips() if parts.get(c) in ("test", "bolivia")]
    for kind in ("l1c", "l2a"):
        per_chip = {}
        for c in chips:
            s = C.scored(c, kind)
            if s is None:
                continue
            x, y, m = s
            xs = np.nan_to_num((x[C.MODEL_BANDS] - mean) / std, nan=0.0).astype(np.float32)
            per_chip[c] = C.counts(predict(net, xs, dev) > 0.5, y, m)
        C.save_counts("published", kind, per_chip)
        print(f"{kind}: {len(per_chip)} tiles, pooled IoU {C.pooled_iou(per_chip, per_chip):.3f}")


if __name__ == "__main__":
    main()
