"""B3: does the pipeline's OmniCloudMask (core/models/cloud.py: ONNX on the CPU, the library's procedure
rewritten in NumPy) give the library's answer?

1. All test images (Earth Search copy): the port against the library's predictions saved by b3_ocm.py
   (made on the GPU). Pixel agreement, and the pooled "unusable" IoU of both.
2. Awkward shapes, the library on the CPU against the port: not square, no-data strips on the edges,
   larger than one window (several windows blended), small images.

    .venv-ml/bin/python -m experiments.b3_check          # writes runs/b3/check_port.json
"""

from __future__ import annotations

import json
import time

import numpy as np

from core.models import cloud
from experiments import b3_common as C
from experiments.b3_ocm import MODEL_DIR, OCM_BANDS


def library_cpu(x3: np.ndarray) -> np.ndarray:
    from omnicloudmask import predict_from_array
    a = np.nan_to_num(x3, nan=0.0).astype(np.float32)
    return predict_from_array(a, inference_device="cpu", mosaic_device="cpu", model_version=4.0,
                              model_download_source="google_drive", destination_model_dir=MODEL_DIR)[0]


def main() -> None:
    t0 = time.time()
    test = C.ids("test")
    agree, cm_port, cm_lib = [], {}, {}
    for i in test:
        y, m = C.scored(i, "test")
        x3 = C.image(i, "test", "es")[OCM_BANDS]
        p = cloud.classes(x3).astype(np.int64)
        lib = np.load(C.RUNS / "ocm_pred" / f"{i}.npz")["es"].astype(np.int64)
        agree.append(float((p == lib)[m].mean()))
        cm_port[i], cm_lib[i] = C.confusion(p, y, m), C.confusion(lib, y, m)
    a = np.array(agree)
    part1 = {"images": len(test), "pixel_agreement_median": float(np.median(a)), "pixel_agreement_min": float(a.min()),
             "images_below_99pct": int((a < 0.99).sum()),
             "iou_port": C.iou_unusable(np.sum(list(cm_port.values()), axis=0)),
             "iou_library_gpu": C.iou_unusable(np.sum(list(cm_lib.values()), axis=0))}
    print("1.", part1, f"({time.time() - t0:.0f} s)", flush=True)

    rng = np.random.default_rng(0)
    big = np.concatenate([C.image(i, "test", "es")[OCM_BANDS] for i in test[:3]], axis=2)   # 509 x 1527
    big = np.concatenate([big, big[:, ::-1]], axis=1)                                          # 1018 x 1527
    base = C.image(test[10], "test", "es")[OCM_BANDS]
    cases = {"509x509": base, "500x502": base[:, :500, :502], "301x333": base[:, 100:401, 50:383],
             "1018x1527 (several windows)": big}
    nd = base.copy()
    nd[:, :, :120] = np.nan                       # a no-data strip on the left (the window moves inward)
    cases["no-data strip left"] = nd
    nd2 = base.copy()
    nd2[:, 400:, :] = np.nan
    nd2[:, :, 450:] = np.nan                      # no data below and right
    cases["no-data below and right"] = nd2
    sparse = base.copy()
    sparse[:, rng.random(base.shape[1:]) < 0.4] = np.nan   # > 30 % no data: smaller windows
    cases["40 % scattered no data"] = sparse
    part2 = {}
    for name, x3 in cases.items():
        p, lib = cloud.classes(x3), library_cpu(x3)
        part2[name] = float((p == lib).mean())
        print(f"2. {name:28s} pixel agreement {part2[name]:.5f}", flush=True)
    (C.RUNS / "check_port.json").write_text(json.dumps({"test_images": part1, "shapes": part2}, indent=1))


if __name__ == "__main__":
    main()
