"""B8: the adopted burn network for the pipeline, saved as ONNX files (run on the CPU by core/models/change.py,
name "burn-utae"; protocol docs/METHODS.md section 11, the exam step).

B7's U-TAE (3 seeds, trained on B5's training fires, without the fires at exam places). The number of dates
may differ from call to call. Each file is checked against PyTorch on a held-out patch.

    .venv-ml/bin/python -m experiments.b8_export
Writes data_cache/models/burn_utae_s<seed>.onnx.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.models.change import MODELS  # noqa: E402
from core.models.utae import UTAE  # noqa: E402
from experiments import b5_common as C  # noqa: E402
from experiments import b7_train as B7  # noqa: E402
from experiments.b5_train import CROP  # noqa: E402


def example(n_dates: int | None = None):
    """Inputs of one held-out patch (two windows of 224), as the network sees them."""
    from core.data.impactmesh_ts import TS
    name = C.built("test")[0]
    meta = json.loads((TS / f"{name}.json").read_text())
    x = B7.inputs(np.load(TS / f"{name}.npy"))
    k = slice(None) if n_dates is None else slice(0, n_dates)
    X = np.stack([x[k, :, 0:CROP, 0:CROP], x[k, :, 32:32 + CROP, 32:32 + CROP]])
    D = np.repeat(np.array(meta["days"])[None, k], 2, 0)
    P = np.repeat(np.array(meta["period"])[None, k], 2, 0)
    return (torch.from_numpy(X), torch.from_numpy(D).long(), torch.from_numpy(P).long(),
            torch.zeros(D.shape, dtype=torch.bool))


def export() -> None:
    import onnxruntime as ort
    MODELS.mkdir(parents=True, exist_ok=True)
    args = example()
    for s in range(3):
        net = UTAE()
        sd = torch.load(ROOT / "runs" / "b7" / "runs" / f"utae_s{s}" / "weights.pt", map_location="cpu")
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        net.eval()
        dest = MODELS / f"burn_utae_s{s}.onnx"
        torch.onnx.export(net, args, str(dest), input_names=["x", "days", "period", "pad"], output_names=["logits"],
                          dynamic_axes={"x": {0: "n", 1: "t"}, "days": {0: "n", 1: "t"}, "period": {0: "n", 1: "t"},
                                        "pad": {0: "n", 1: "t"}, "logits": {0: "n"}},
                          dynamo=False, opset_version=17)
        sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        for n in (None, 5):                                   # all dates, and fewer: the number of dates may vary
            a = example(n)
            with torch.no_grad():
                ref = net(*a).numpy()
            out = sess.run(None, {"x": a[0].numpy(), "days": a[1].numpy(), "period": a[2].numpy(), "pad": a[3].numpy()})[0]
            print(f"  {dest.name}: {a[0].shape[1]} dates, largest difference to PyTorch {np.abs(out - ref).max():.2e}",
                  flush=True)


if __name__ == "__main__":
    export()
