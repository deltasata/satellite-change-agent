"""ImpactMesh-Fire as an Earth Search time series: every clear date, not two medians (milestone B7; protocol:
docs/METHODS.md (milestone B7, in progress)).

The same patches, items, grids, windows and cloud masks as the B5 copy (core/data/impactmesh_es.py), but
each date is kept. For each patch: the dates of its B5 window ("normal" or "wide", as B5 used) with at
least 1 % clear pixels in the patch; at most 12 per window, the 12 with the most clear pixels, in time
order. Patches that B5 dropped are not copied.

Per patch, in data_cache/burn/impactmesh/ts/:
  <name>.npy    (T, 7, 256, 256) float16 reflectance, NaN where not clear (cloud, shadow, no data)
  <name>.json   {"dates": [...], "days": [...] (days since the after window began; before dates negative),
                 "period": [0 before, 1 after], "stage": "normal" | "wide"}

    .venv-ml/bin/python -m core.data.impactmesh_ts     # about 2 hours, cloud masks on the GPUs (B5_GPUS)
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.data import impactmesh_es as E  # noqa: E402

TS = E.DIR / "ts"
MIN_CLEAR = 0.01
MAX_PER_WINDOW = 12


def select(dates: list[str], clear: np.ndarray, window: tuple[date, date]) -> list[int]:
    """Indices of the dates kept in one window: at least MIN_CLEAR clear, at most MAX_PER_WINDOW (the
    clearest), in time order."""
    t = [date.fromisoformat(d) for d in dates]
    idx = [k for k in range(len(dates)) if window[0] <= t[k] <= window[1] and clear[k] >= MIN_CLEAR]
    if len(idx) > MAX_PER_WINDOW:
        idx = sorted(sorted(idx, key=lambda k: -clear[k])[:MAX_PER_WINDOW])
    return idx


def _save(p: dict, dates: list[str], x: np.ndarray, stage: str) -> None:
    wb, wa = E._windows(date.fromisoformat(p["date"]), stage)
    clear = np.isfinite(x[:, 2]).mean(axis=(1, 2)) if len(dates) else np.zeros(0)
    kb, ka = select(dates, clear, wb), select(dates, clear, wa)
    keep = kb + ka
    meta = dict(dates=[dates[k] for k in keep], days=[(date.fromisoformat(dates[k]) - wa[0]).days for k in keep],
                period=[0] * len(kb) + [1] * len(ka), stage=stage)
    np.save(TS / f"{p['name']}.npy", x[keep].astype(np.float16))
    (TS / f"{p['name']}.json").write_text(json.dumps(meta))


def _stage(name: str) -> str | None:
    z = np.load(E.OUT / f"{name}.npz")
    return None if "dropped" in z.files else str(z["stage"])


def build_block(key: str) -> str:
    from core.data.cloud_mask import valid_mask
    from core.data.timeseries import FILE_FAULTS
    classes = E._classes_gpu if os.environ.get("CUDA_VISIBLE_DEVICES") else __import__("core.models.cloud", fromlist=["classes"]).classes
    by_stage = defaultdict(list)
    for p in E._blocks_cached()[key]:
        if (TS / f"{p['name']}.json").exists() or not (E.OUT / f"{p['name']}.npz").exists():
            continue
        s = _stage(p["name"])
        if s:
            by_stage[s].append(p)
    if not by_stage:
        return "cached"
    report = []
    for stage, todo in by_stage.items():
        gbox, (gx0, gy1) = E._grid(todo)
        d = date.fromisoformat(todo[0]["date"])
        (b0, _), (_, a1) = E._windows(d, stage)
        items = E._items(gbox.geographic_extent.boundingbox, b0, a1)
        dates, x, scl = E._load(items, gbox) if items else ([], None, None)
        for k in range(len(dates)):     # the same cloud mask as the B5 copy
            cc = classes(x[k][[2, 1, 4]])
            bad = ~(valid_mask(scl[k], FILE_FAULTS) & (cc == 0) & np.isfinite(x[k]).all(0))
            x[k][:, bad] = np.nan
        for p in todo:
            i, j = round((gy1 - p["y0"]) / E.PIX), round((p["x0"] - gx0) / E.PIX)
            xp = x[:, :, i:i + E.PATCH, j:j + E.PATCH] if dates else np.zeros((0, 7, E.PATCH, E.PATCH), np.float32)
            _save(p, dates, xp, stage)
        report.append(f"{stage}: {len(dates)} days, {len(todo)} patches")
    return "; ".join(report)


def main(workers: int = 16) -> None:
    TS.mkdir(parents=True, exist_ok=True)
    bl = E.blocks()
    keys = sorted(bl, key=lambda k: -len(bl[k]))
    log = E.DIR / "ts_log.jsonl"
    print(f"cloud masks on the GPUs {','.join(E.GPUS)}", flush=True)
    with ProcessPoolExecutor(workers, initializer=E._init, initargs=(True,)) as pool:
        futs = {k: pool.submit(build_block, k) for k in keys}
        for n, (k, f) in enumerate(futs.items(), 1):
            try:
                msg = f.result()
            except Exception as e:      # recorded; the block is retried on the next run
                msg = f"ERROR {type(e).__name__}: {str(e)[:300]}"
            with log.open("a") as fh:
                fh.write(json.dumps({"block": k, "msg": msg}) + "\n")
            if n % 50 == 0:
                print(f"{n}/{len(keys)} blocks", flush=True)


if __name__ == "__main__":
    main()
