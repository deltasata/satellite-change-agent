"""B3: save OmniCloudMask's two networks (model version 4) as ONNX files for the pipeline.

    .venv-ml/bin/python -m experiments.b3_export
Writes data_cache/models/ocm_v4_<regnety_004|edgenext_small>.onnx, built exactly as the library builds them
(segmentation_models_pytorch U-Net, timm encoder, 3 inputs, 4 classes, the library's weight files), with
the image height and width free. Checks onnxruntime against PyTorch on random inputs of several sizes,
including sizes that are not multiples of 32 (a 5 km pipeline area is about 500 x 500 pixels).
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.b3_ocm import MODEL_DIR

OUT = MODEL_DIR.parent
NETS = {"regnety_004": ("tu-regnety_004", "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors"),
        "edgenext_small": ("tu-edgenext_small", "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors")}


def main() -> None:
    import onnxruntime as ort
    from omnicloudmask.model_utils import load_model_from_weights
    for short, (timm_name, fname) in NETS.items():
        net = load_model_from_weights(model_name=timm_name, weights_path=str(MODEL_DIR / fname), model_library="smp",
                                      device=torch.device("cpu"), dtype=torch.float32)
        dest = OUT / f"ocm_v4_{short}.onnx"
        torch.onnx.export(net, (torch.randn(1, 3, 512, 512),), str(dest), input_names=["x"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n", 2: "h", 3: "w"}, "logits": {0: "n", 2: "h", 3: "w"}},
                          opset_version=17, dynamo=False)
        sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        worst = 0.0
        for h, w in ((509, 509), (500, 502), (512, 512), (301, 333), (1000, 1000)):
            x = torch.randn(1, 3, h, w)
            with torch.no_grad():
                ref = net(x).numpy()
            got = sess.run(None, {"x": x.numpy()})[0]
            assert got.shape == ref.shape, (short, h, w, got.shape, ref.shape)
            worst = max(worst, float(np.abs(got - ref).max()))
        print(f"{dest.name}: largest difference to PyTorch over 5 sizes {worst:.2e}")


if __name__ == "__main__":
    main()
