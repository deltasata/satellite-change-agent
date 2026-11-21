"""B8: TerraMind-base-Fire as published, on inputs built by us (core/data/tmfire_inputs.py), protocol
docs/METHODS.md section 11, detector 4 and the check before any score on the new fires.

    python -m experiments.b8_tmfire run heldout     # the 42 held-out fires, our inputs (calls .venv-fire)
    python -m experiments.b8_tmfire check           # IoU on B5's scored pixels against 0.864 with ImpactMesh's
                                                    # own files; how often our dates are ImpactMesh's
    python -m experiments.b8_tmfire run cems2026    # the new fires
Writes runs/b8/tmfire_<set>/<patch>_prediction.tif and runs/b8/tmfire_check.json.
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
from experiments import b5_common as C  # noqa: E402

RUNS = ROOT / "runs" / "b8"
FILES = ROOT / "data_cache" / "models" / "terramind_fire"
SETS = {"heldout": ROOT / "data_cache" / "burn" / "impactmesh" / "tm4_fire",
        "cems2026": ROOT / "data_cache" / "burn" / "cems" / "tm4_fire"}


def run(which: str, gpu: str = "5") -> None:
    cfg = (C.RUNS / "tmfire_config.yaml").read_text()          # B5's copy of the model's own configuration
    old = str(ROOT / "data_cache" / "burn" / "impactmesh" / "fire_data")
    assert old in cfg
    cfg = cfg.replace(old, str(SETS[which]))
    RUNS.mkdir(parents=True, exist_ok=True)
    cpath = RUNS / f"tmfire_config_{which}.yaml"
    cpath.write_text(cfg)
    out = RUNS / f"tmfire_{which}"
    subprocess.run([str(ROOT / ".venv-fire" / "bin" / "python"), str(ROOT / "experiments" / "b5_tmfire_infer.py"),
                    "-c", str(cpath), "--ckpt", str(FILES / "TerraMind_v1_base_ImpactMesh_fire.pt"), "-o", str(out),
                    "--predict_split", str(SETS[which] / "split.txt")],
                   check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONNOUSERSITE": "1"})


def maps(which: str) -> dict:
    import rasterio
    out = {}
    for f in sorted((RUNS / f"tmfire_{which}").glob("*_prediction.tif")):
        with rasterio.open(f) as r:
            out[f.name.replace("_prediction.tif", "")] = r.read(1) == 1
    return out


def check() -> dict:
    """Our inputs against ImpactMesh's own, on the 42 held-out fires (reported, deciding nothing)."""
    ours = maps("heldout")
    own = C.read("tmfire")                                   # B5: ImpactMesh's own files, B5's scored pixels
    cms = {n: C.confusion(p, C.load(n)) for n, p in ours.items() if n in own}
    both = {n: own[n] for n in cms}
    res = {"n_patches": len(cms), "iou_own_files": C.iou(np.sum(list(both.values()), axis=0)),
           "iou_our_inputs": C.iou(np.sum(list(cms.values()), axis=0)),
           "our_minus_own": C.bootstrap_difference(both, cms)}
    res["same_dates"] = same_dates()
    (RUNS / "tmfire_check.json").write_text(json.dumps(res, indent=1))
    (RUNS / "cm_tmfire_ourinputs_heldout.json").write_text(json.dumps(cms))
    print(json.dumps(res, indent=1))
    return res


def same_dates() -> dict:
    """Per step, how often our optical and radar date is ImpactMesh's (needs .venv-fire's zarr; run there)."""
    code = r"""
import json, glob, numpy as np, zarr
D = 'data_cache/burn/impactmesh'
same = {'S2': np.zeros(4), 'S1': np.zeros(4)}; n = 0
for f in sorted(glob.glob(D + '/tm4/*.npz')):
    name = f.split('/')[-1][:-4]; z = np.load(f); n += 1
    for m, key in (('S2', 's2_days'), ('S1', 's1_days')):
        t = zarr.open(zarr.storage.ZipStore(f'{D}/{m}{"L2A" if m == "S2" else "RTC"}/{name}_{m}{"L2A" if m == "S2" else "RTC"}.zarr.zip', mode='r'), mode='r')
        td = [str(v)[:10] for v in t['time'][:].astype('datetime64[ns]').astype('datetime64[D]')]
        same[m] += [str(z[key][k]) == td[k] for k in range(4)]
print(json.dumps({'n': n, **{m: (v / n).round(3).tolist() for m, v in same.items()}}))
"""
    r = subprocess.run([str(ROOT / ".venv-fire" / "bin" / "python"), "-c", code], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
    return json.loads(r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], *(sys.argv[3:4]))
    else:
        check()
