"""B1: save the U-Net water models as ONNX files, so the pipeline can run them without PyTorch.

ONNX is a standard file format for a trained network: the layers and their numbers. The small library
onnxruntime runs such a file on the CPU. The pipeline's own Python environment then needs no PyTorch.

    .venv-ml/bin/python -m experiments.b1_export
Writes data_cache/models/water_unet_<ALL or event>.onnx (ALL = trained on every event: the pipeline's
model; <event> = trained with that event left out: used only to test on that event) and checks that
onnxruntime gives the same probabilities as PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments import b1_common as C
from experiments.b1_train import build

OUT = C.ROOT / "data_cache" / "models"


def main():
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, 6, 224, 224)
    for w in sorted((C.RUNS / "weights" / "unet").glob("*.pt")):
        net = build("unet")
        net.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
        net.eval()
        dest = OUT / f"water_unet_{w.stem}.onnx"
        torch.onnx.export(net, (x,), dest, input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, dynamo=False)
        with torch.no_grad():
            ref = net(x).numpy()
        got = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
        print(f"{dest.name}: largest difference to PyTorch {np.abs(got - ref).max():.2e}")


if __name__ == "__main__":
    main()
