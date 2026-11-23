"""B9, avenue 2: SpaceNet 7's areas as change-network inputs, made with B4's rule (protocol: docs/METHODS.md section 12).

Per area (core/data/spacenet7.py): windows of +- 60 days around the middle of the first and last labelled
month, widened once to +- 90 if fewer than half the pixels have >= 2 clear dates before and >= 1 after;
otherwise the area is dropped. Median images of the clear dates of each window (7 bands), the usable
pixels, the label (1 new building, 0 not, -1 no label: unusable Planet image or not usable here).

Alignment check (before any network sees the data): the label's new-building pixels must sit where the
Sentinel-2 images changed. For shifts of the label by -2..+2 pixels in each direction, the mean change
(root-mean-square over bands of after - before, in reflectance) on new-building pixels minus that on the
other usable pixels; the largest contrast should be at shift (0, 0).

    python -m experiments.b9_sn7_inputs
Writes data_cache/spacenet7/inputs/<area>.npz and runs/b9/sn7_inputs.json.
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.data import spacenet7 as S  # noqa: E402

OUT = S.DIR / "inputs"


def windows(d1: date, d2: date, half: int):
    w = timedelta(days=half)
    return (d1 - w, d1 + w), (d2 - w, d2 + w)


def make(area: str) -> dict:
    z = np.load(S.OUT / f"{area}.npz")
    x = z["x"].astype(np.float32)
    dates = np.array([str(d) for d in z["dates"]], dtype="datetime64[D]")
    d1, d2 = date.fromisoformat(str(z["d1"])), date.fromisoformat(str(z["d2"]))
    for half in (60, 90):
        (b0, b1), (a0, a1) = windows(d1, d2, half)
        ib = (dates >= np.datetime64(b0)) & (dates <= np.datetime64(b1))
        ia = (dates >= np.datetime64(a0)) & (dates <= np.datetime64(a1))
        nb, na = np.isfinite(x[ib, 2]).sum(0), np.isfinite(x[ia, 2]).sum(0)
        usable = (nb >= 2) & (na >= 1)
        if usable.mean() >= 0.5:
            break
    else:
        return {"area": area, "dropped": True, "usable_fraction": float(usable.mean())}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before, after = np.nanmedian(x[ib], axis=0), np.nanmedian(x[ia], axis=0)
    lab = z["label"].astype(np.int16)
    label = np.where(usable & (lab != 255), lab, -1).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{area}.npz", before=before.astype(np.float32), after=after.astype(np.float32),
                        usable=usable, label=label, half=half)
    return {"area": area, "dropped": False, "half": half, "usable_fraction": float(usable.mean()),
            "new_building_fraction": float((label == 1).sum() / max(1, (label >= 0).sum())), "shape": list(label.shape)}


def alignment(areas: list[str]) -> dict:
    """Contrast of Sentinel-2 change between new-building pixels and the rest, for label shifts."""
    num = {}
    for a in areas:
        z = np.load(OUT / f"{a}.npz")
        ch = np.sqrt(np.nanmean((z["after"] - z["before"]) ** 2, axis=0))
        lab = z["label"]
        for di in range(-2, 3):
            for dj in range(-2, 3):
                l = np.roll(np.roll(lab, di, 0), dj, 1)[2:-2, 2:-2]
                c = ch[2:-2, 2:-2]
                ok = np.isfinite(c) & (l >= 0)
                if (l[ok] == 1).sum() < 20:
                    continue
                num.setdefault((di, dj), []).append(float(c[ok & (l == 1)].mean() - c[ok & (l == 0)].mean()))
    mean = {f"{k[0]},{k[1]}": round(float(np.mean(v)), 5) for k, v in num.items()}
    best = max(mean, key=mean.get)
    return {"contrast_by_shift": mean, "best_shift": best, "n_areas": len(next(iter(num.values())))}


def main() -> None:
    res = [make(a) for a in S.areas() if (S.OUT / f"{a}.npz").exists()]
    kept = [r["area"] for r in res if not r["dropped"]]
    al = alignment(kept)
    out = {"n_areas": len(res), "n_kept": len(kept), "areas": res, "alignment": al}
    (ROOT / "runs" / "b9" / "sn7_inputs.json").write_text(json.dumps(out, indent=1))
    print(f"{len(kept)} of {len(res)} areas kept; median new-building fraction "
          f"{np.median([r['new_building_fraction'] for r in res if not r['dropped']]):.4f}")
    print("alignment, best shift:", al["best_shift"], "contrast at 0,0:", al["contrast_by_shift"]["0,0"],
          "at best:", al["contrast_by_shift"][al["best_shift"]])


if __name__ == "__main__":
    main()
