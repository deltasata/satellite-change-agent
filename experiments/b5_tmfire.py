"""B5 check: TerraMind-base-Fire as published, on ImpactMesh-Fire's own held-out images (protocol:
docs/METHODS.md section 8, "Checks of the setup"; reported, deciding nothing).

The model (IBM and ESA, Apache-2.0, Hugging Face `ibm-esa-geospatial/TerraMind-base-Fire`) reads four dates
of 12-band Sentinel-2 Level-2A, Sentinel-1 radar and elevation. It is run with the `impactmesh` package's
own script (`impactmesh/run_inference.py`) and its own configuration, changed only in where the data are:
the 42 held-out fires (`test_holdout`). It needs TerraTorch 1.2.14 and impactmesh 0.2.0, installed in their
own environment `.venv-fire` (the model card's TerraTorch branch "multimodal" no longer exists; the
released TerraTorch has the same parts).

    python -m experiments.b5_tmfire run      # predictions (calls .venv-fire), about 10 minutes on one GPU
    python -m experiments.b5_tmfire score    # IoU of "burnt" on every labelled pixel, and on Part A's scored pixels
Writes runs/b5/tmfire/<patch>_prediction.tif and runs/b5/tmfire_check.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import impactmesh_es as D  # noqa: E402
from experiments import b5_common as C  # noqa: E402

FILES = ROOT / "data_cache" / "models" / "terramind_fire"
OUT = C.RUNS / "tmfire"


def run(gpu: str = "1") -> None:
    cfg = (FILES / "terramind_v1_base_impactmesh_fire.yaml").read_text()
    cfg = cfg.replace("data/ImpactMesh-Fire/data", str(D.DIR / "fire_data"))
    for s in ("train", "val", "test"):
        cfg = cfg.replace(f"data/ImpactMesh-Fire/split/impactmesh_wildfire_{s}.txt", str(D.SPLITS / f"impactmesh_fire_{s}.txt"))
    cfg = cfg.replace("class_path: lightning.pytorch.loggers.WandbLogger", "class_path: lightning.pytorch.loggers.CSVLogger")
    cfg = cfg.replace("      project: ImpactMesh-Fire\n", "      save_dir: " + str(C.RUNS / "tmfire_logs") + "\n")
    cfg = "\n".join(line for line in cfg.splitlines() if not line.strip().startswith(("config:", "modalities: S1", "dataset: IM")))
    C.RUNS.mkdir(parents=True, exist_ok=True)
    cpath = C.RUNS / "tmfire_config.yaml"
    cpath.write_text(cfg)
    script = ROOT / "experiments" / "b5_tmfire_infer.py"     # the package's script, input moved to the GPU
    subprocess.run([str(ROOT / ".venv-fire" / "bin" / "python"), str(script), "-c", str(cpath),
                    "--ckpt", str(FILES / "TerraMind_v1_base_ImpactMesh_fire.pt"), "-o", str(OUT),
                    "--predict_split", str(D.SPLITS / "impactmesh_fire_test_holdout.txt")],
                   check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1"})


def score() -> dict:
    import rasterio
    names = (D.SPLITS / "impactmesh_fire_test_holdout.txt").read_text().split()
    every, ours, cms = np.zeros(4), np.zeros(4), {}
    n = 0
    for name in names:
        f = OUT / f"{name}_prediction.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as r:
            p = r.read(1) == 1
        y = D.label(name)
        m, t = y >= 0, y == 1
        every += [(p & t & m).sum(), (p & ~t & m).sum(), (~p & t & m).sum(), (~p & ~t & m).sum()]
        es = D.OUT / f"{name}.npz"
        if es.exists() and "dropped" not in np.load(es).files:
            z = C.load(name)
            cm = C.confusion(p, z)
            cms[name] = cm
            ours += cm
        n += 1
    C.save("tmfire", cms)
    res = {"n_patches": n, "iou_every_labelled_pixel": C.iou(every), "iou_part_a_scored_pixels": C.iou(ours),
           "n_patches_part_a": len(cms)}
    (C.RUNS / "tmfire_check.json").write_text(json.dumps(res, indent=1))
    print(res)
    return res


if __name__ == "__main__":
    {"run": run, "score": score}[sys.argv[1]]()
