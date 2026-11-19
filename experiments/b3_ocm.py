"""B3, OmniCloudMask as published (protocol: docs/METHODS.md section 6, detector 3).

Library 1.8.1, model version 4 (the latest in the library; two networks, regnety_004 and edgenext_small,
whose outputs the library averages), all its own settings (patch 1000 px, overlap 300, float32, softmax).
Input: red (B04), green (B03), narrow near infrared (B8A), its own Sentinel-2 band choice, as reflectance;
the library normalises each image itself. Missing values become 0, its no-data value. Output: the 4
CloudSEN12 classes; "unusable" = classes 1-3. Nothing is fitted by us.

The weights are downloaded from Google Drive (the library's second source): Hugging Face refused this
machine's requests on 18 November after the data download (429 Too Many Requests).

    CUDA_VISIBLE_DEVICES=1 .venv-ml/bin/python -m experiments.b3_ocm
Writes runs/b3/cm_<es|gee>_ocm.json and runs/b3/ocm_pred/<id>.npz (the class maps, for figures).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from experiments import b3_common as C

MODEL_DIR = C.ROOT / "data_cache" / "models" / "omnicloudmask"
OCM_BANDS = [2, 1, 4]       # red, green, nir08 in our 7-band order (B02 B03 B04 B08 B8A B11 B12)
MODEL_VERSION = 4.0


def predict(x7: np.ndarray) -> np.ndarray:
    from omnicloudmask import predict_from_array
    a = np.nan_to_num(x7[OCM_BANDS], nan=0.0).astype(np.float32)
    out = predict_from_array(a, inference_device="cuda", mosaic_device="cuda", model_version=MODEL_VERSION,
                             model_download_source="google_drive", destination_model_dir=MODEL_DIR)
    return out[0].astype(np.int64)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    test = C.ids("test")
    out_dir = C.RUNS / "ocm_pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    cms = {"es": {}, "gee": {}}
    t0 = time.time()
    for k, i in enumerate(test, 1):
        y, m = C.scored(i, "test")
        preds = {}
        for kind in ("es", "gee"):
            preds[kind] = predict(C.image(i, "test", kind))
            cms[kind][i] = C.confusion(preds[kind], y, m)
        np.savez_compressed(out_dir / f"{i}.npz", **{k_: v.astype(np.uint8) for k_, v in preds.items()})
        if k % 100 == 0:
            print(f"  {k}/{len(test)} ({time.time() - t0:.0f} s)", flush=True)
    for kind, c in cms.items():
        C.save("ocm", kind, c)
    print(f"OmniCloudMask: {len(test)} test images, {time.time() - t0:.0f} s; torch {torch.__version__}")


if __name__ == "__main__":
    main()
