"""B4: the change networks the pipeline uses, saved as ONNX files (run on the CPU by core/models/change.py).

For the given network: the runs that trained on every city (group ALL) and those that left Paris and Bercy
out (for the exam's Paris question), 3 seeds each. Each file is checked against PyTorch on real windows.

    .venv-ml/bin/python -m experiments.b4_export --model fsd
Writes data_cache/models/change_<model>_<group>_s<seed>.onnx.
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
from experiments import b4_common as C  # noqa: E402
from experiments.b4_train import KEEP_WEIGHTS, Net  # noqa: E402


def export(model: str) -> None:
    import onnxruntime as ort
    z = C.inputs("montpellier")
    b = torch.from_numpy(np.stack([C.scale(z["before"])[:, i:i + 224, j:j + 224] for i, j in ((0, 0), (100, 150), (200, 200))]))
    a = torch.from_numpy(np.stack([C.scale(z["after"])[:, i:i + 224, j:j + 224] for i, j in ((0, 0), (100, 150), (200, 200))]))
    MODELS.mkdir(parents=True, exist_ok=True)
    for group in KEEP_WEIGHTS:
        for s in range(3):
            net = Net(model)
            sd = torch.load(C.RUNS / "runs" / f"{model}_{group}_s{s}" / "weights.pt", map_location="cpu")
            net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
            net.eval()
            dest = MODELS / f"change_{model}_{group}_s{s}.onnx"
            torch.onnx.export(net, (b[:2], a[:2]), str(dest), input_names=["before", "after"], output_names=["logits"],
                              dynamic_axes={"before": {0: "n"}, "after": {0: "n"}, "logits": {0: "n"}},
                              **({"dynamo": False, "opset_version": 17} if model == "fsd" else {"dynamo": True}))
            with torch.no_grad():
                ref = net(b, a).numpy()
            out = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"]).run(None, {"before": b.numpy(), "after": a.numpy()})[0]
            print(f"  {dest.name}: largest difference to PyTorch {np.abs(out - ref).max():.2e}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["fsd", "terramind"])
    export(ap.parse_args().model)
