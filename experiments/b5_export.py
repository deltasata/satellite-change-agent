"""B5: the burn networks the pipeline uses, saved as ONNX files (run on the CPU by core/models/change.py,
names "burn-<model>").

The 3 runs of the given network (trained on the training fires, without the fires at exam places). Each file
is checked against PyTorch on windows of a held-out patch.

    .venv-ml/bin/python -m experiments.b5_export --model unet        # or terramind
Writes data_cache/models/burn_<model>_s<seed>.onnx.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models.change import MODELS  # noqa: E402
from experiments import b5_common as C  # noqa: E402
from experiments.b5_train import CORNERS, CROP, Net  # noqa: E402


def export(model: str) -> None:
    import onnxruntime as ort
    z = C.load(C.built("test")[0])
    b = torch.from_numpy(np.stack([C.scale(z["before"])[:, i:i + CROP, j:j + CROP] for i, j in CORNERS[:3]]))
    a = torch.from_numpy(np.stack([C.scale(z["after"])[:, i:i + CROP, j:j + CROP] for i, j in CORNERS[:3]]))
    MODELS.mkdir(parents=True, exist_ok=True)
    for s in range(3):
        net = Net(model)
        sd = torch.load(C.RUNS / "runs" / f"{model}_s{s}" / "weights.pt", map_location="cpu")
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        net.eval()
        dest = MODELS / f"burn_{model}_s{s}.onnx"
        torch.onnx.export(net, (b[:2], a[:2]), str(dest), input_names=["before", "after"], output_names=["logits"],
                          dynamic_axes={"before": {0: "n"}, "after": {0: "n"}, "logits": {0: "n"}},
                          **({"dynamo": False, "opset_version": 17} if model == "unet" else {"dynamo": True}))
        with torch.no_grad():
            ref = net(b, a).numpy()
        out = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"]).run(None, {"before": b.numpy(), "after": a.numpy()})[0]
        print(f"  {dest.name}: largest difference to PyTorch {np.abs(out - ref).max():.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["unet", "terramind"])
    export(ap.parse_args().model)
