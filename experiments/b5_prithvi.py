"""B5: Prithvi-EO-2.0-300M-BurnScars as published (protocol: docs/METHODS.md section 8, detector 5, and the
setup check).

The model (NASA and IBM, Apache-2.0, Hugging Face `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`) was
fine-tuned on HLS Burn Scars: one image after the fire, 6 bands (B02, B03, B04, B8A, B11, B12) as
reflectance at 30 m. It is run as its own files say: no-data -> 0, then (x - mean) / spread with its
statistics, reflect-padded to a multiple of 512 and cut into 512 windows (its inference.py), "burnt" where
the probability is above 0.5.

  check   its own HLS test scenes (splits/test.txt of the model files): IoU of "burnt" (published 0.875)
  partA   the held-out patches' after median image, averaged to 30 m (3 x 3 pixels; the 256-pixel patch
          extended by one edge pixel to 258 = 86 x 3), and the map taken back to 10 m

    .venv-ml/bin/python -m experiments.b5_prithvi check
    .venv-ml/bin/python -m experiments.b5_prithvi partA
Writes runs/b5/prithvi_check.json and runs/b5/runs/prithvi/prob.npz (and result.json).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import b5_common as C  # noqa: E402

FILES = ROOT / "data_cache" / "models" / "prithvi_burnscars"
HLS = ROOT / "data_cache" / "burn" / "hls_dl" / "x"
MEAN = np.array([0.033349706741586264, 0.05701185520536176, 0.05889748132001316, 0.2323245113436119,
                 0.1972854853760658, 0.11944914225186566], np.float32)[:, None, None]
STD = np.array([0.02269135568823774, 0.026807560223070237, 0.04004109844362779, 0.07791732423672691,
                0.08708738838140137, 0.07241979477437814], np.float32)[:, None, None]
SIX = [0, 1, 2, 4, 5, 6]    # B02, B03, B04, B8A, B11, B12 among our 7 bands
WIN = 512


def build(dev) -> torch.nn.Module:
    from terratorch.models import EncoderDecoderFactory
    net = EncoderDecoderFactory().build_model(
        task="segmentation", backbone="prithvi_eo_v2_300", backbone_pretrained=False,
        backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        necks=[{"name": "SelectIndices", "indices": [5, 11, 17, 23]}, {"name": "ReshapeTokensToImage"},
               {"name": "LearnedInterpolateToPyramidal"}],
        decoder="UNetDecoder", decoder_channels=[512, 256, 128, 64], num_classes=2)
    sd = torch.load(FILES / "Prithvi_EO_V2_300M_BurnScars.pt", map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items() if k.startswith("model.")}, strict=True)
    return net.to(dev).eval()


@torch.no_grad()
def probability(net, x6: np.ndarray, dev) -> np.ndarray:
    """x6: (6, H, W) reflectance, NaN where missing. Returns the probability of "burnt" (H, W)."""
    x = (np.nan_to_num(x6, nan=0.0) - MEAN) / STD
    H, W = x.shape[1:]
    ph, pw = (WIN - H % WIN) % WIN, (WIN - W % WIN) % WIN
    x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect").astype(np.float32)
    out = np.zeros(x.shape[1:], np.float32)
    for i in range(0, x.shape[1], WIN):
        for j in range(0, x.shape[2], WIN):
            t = torch.from_numpy(x[None, :, i:i + WIN, j:j + WIN]).to(dev)
            o = net(t)
            o = o.output if hasattr(o, "output") else o
            out[i:i + WIN, j:j + WIN] = torch.softmax(o.float(), 1)[0, 1].cpu().numpy()
    return out[:H, :W]


def check() -> dict:
    import rasterio
    dev = torch.device("cuda")
    net = build(dev)
    tot = np.zeros(4)
    tokens = (FILES / "splits" / "test.txt").read_text().split()
    n = 0
    for tok in tokens:
        f = sorted(HLS.glob(f"*/subsetted_512x512_HLS.S30.{tok}*_merged.tif"))
        if not f:
            continue
        with rasterio.open(f[0]) as r:
            x = r.read().astype(np.float32)
        with rasterio.open(str(f[0]).replace("_merged.tif", ".mask.tif")) as r:
            y = r.read(1)
        p = probability(net, x, dev) > 0.5
        m, t = y >= 0, y == 1
        tot += [(p & t & m).sum(), (p & ~t & m).sum(), (~p & t & m).sum(), (~p & ~t & m).sum()]
        n += 1
    res = {"n_scenes": n, "iou_burnt": C.iou(tot), "published": 0.8752}
    C.RUNS.mkdir(parents=True, exist_ok=True)
    (C.RUNS / "prithvi_check.json").write_text(json.dumps(res, indent=1))
    print(res)
    return res


def to30(x: np.ndarray) -> np.ndarray:
    """(C, 256, 256) at 10 m -> (C, 86, 86) at 30 m: the edge pixel repeated once, then 3 x 3 means."""
    x = np.pad(x, ((0, 0), (0, 2), (0, 2)), mode="edge")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x.reshape(x.shape[0], 86, 3, 86, 3), axis=(2, 4))


def part_a() -> None:
    dev = torch.device("cuda")
    t0 = time.time()
    net = build(dev)
    out = {}
    for name in C.built("test"):
        z = C.load(name)
        p30 = probability(net, to30(z["after"][SIX]), dev)
        p = np.repeat(np.repeat(p30, 3, 0), 3, 1)[:256, :256]
        out[name] = np.round(p * 255).astype(np.uint8)
    d = C.RUNS / "runs" / "prithvi"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "prob.npz", **out)
    (d / "result.json").write_text(json.dumps({"name": "prithvi", "n_test_patches": len(out), "seconds": round(time.time() - t0),
                                               "parameters": sum(q.numel() for q in net.parameters())}))
    print(f"{len(out)} patches, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    {"check": check, "partA": part_a}[sys.argv[1]]()
