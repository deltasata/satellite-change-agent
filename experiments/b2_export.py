"""B2: save the radar water models (TerraMind-1.0-base, input S1RTC) as ONNX files for the pipeline.

    .venv-ml/bin/python -m experiments.b2_export
Writes data_cache/models/radar_terramind_<fold>_s<seed>.onnx (+ .onnx.data, the weights), for fold ALL
(trained on every event: the pipeline's model) and for each left-out event (used only to test on that
event), seeds 0-2. Checks that onnxruntime gives the same output as PyTorch.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments import b2_common as B
from experiments.b2_train import Net, run_name

OUT = B.ROOT / "data_cache" / "models"


def main():
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    x = torch.randn(2, 2, 224, 224)
    folds = sorted({p.name.split("_")[-2] for p in (B.RUNS / "runs").glob("terramind_s1_rtc_ce_lr5e-05_*_s0")})
    for fold in folds:
        for s in range(3):
            dest = OUT / f"radar_terramind_{fold}_s{s}.onnx"
            if dest.exists():
                continue
            net = Net("terramind", "s1", "rtc")
            w = B.RUNS / "runs" / run_name("terramind", "s1", "rtc", fold, s) / "weights.pt"
            net.load_state_dict({k: v.float() for k, v in torch.load(w, map_location="cpu").items()})
            net.eval()
            torch.onnx.export(net, (x,), dest, input_names=["x"], output_names=["logits"],
                              dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, dynamo=True)
            with torch.no_grad():
                ref = net(x).numpy()
            got = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
            print(f"{dest.name}: largest difference to PyTorch {np.abs(got - ref).max():.2e}", flush=True)


if __name__ == "__main__":
    main()
