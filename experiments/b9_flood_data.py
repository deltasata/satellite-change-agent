"""B9, avenue 1: the before and after images of flooded places, exactly as the pipeline builds them for a
construction question (protocol: docs/METHODS.md section 12).

For every hand-labelled Sen1Floods11 chip (446 in 11 events) the question "was there construction?" is put
to `measure_change` with the flood question's periods (the event day d of the chip's event: before d-60 ..
d-10, after d-3 .. d+3), a 5 km square on the chip's centre. The change detector is replaced, for this
script only, by one that records what the pipeline hands to a change network: the before and after median
images (7 bands) and the pixels the pipeline counts as usable (at least 2 clear dates before, 1 after).
The 78 exam questions and the Conselice case (B4's Part C) are recorded the same way, from the image stacks
the pipeline stored in B4 (nothing is downloaded for them).

    python -m experiments.b9_flood_data chips        # all 446 chips (downloads the stacks not yet stored)
    python -m experiments.b9_flood_data chips --reverse   # a second process, from the other end
    python -m experiments.b9_flood_data partc        # B4's Part C questions
Writes data_cache/b9/flood/<chip>.npz and data_cache/b9/partc/<question id>.npz (before, after, usable).
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data_cache" / "b9"
S1F = ROOT / "data_cache" / "sen1floods11"
EVENT_NAME = {"Cambodia": "Mekong"}            # the chips of the Cambodian event are named "Mekong"


def event_days() -> dict[str, date]:
    g = json.loads((S1F / "metadata.geojson").read_text())
    out = {}
    for f in g["features"]:
        p = f["properties"]
        out[EVENT_NAME.get(p["location"], p["location"])] = date(*map(int, p["s2_date"].split("/")))
    return out


def chips() -> list[dict]:
    """Every hand-labelled chip: name, event, centre (lat, lon), event day."""
    import rasterio
    from rasterio.warp import transform_bounds
    days = event_days()
    out = []
    for f in sorted((S1F / "S2Hand").glob("*_S2Hand.tif")):
        name = f.name.removesuffix("_S2Hand.tif")
        with rasterio.open(f) as r:
            w, s, e, n = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
        ev = name.split("_")[0]
        out.append(dict(name=name, event=ev, lat=(s + n) / 2, lon=(w + e) / 2, day=days[ev].isoformat()))
    return out


def _record(before, after, name):
    """Stands in for core.models.change.changed: saves the median images and the usable pixels."""
    usable = inspect.currentframe().f_back.f_locals["usable"]       # detect_change_timeseries's own mask
    np.savez_compressed(os.environ["B9_DEST"], before=before.astype(np.float32), after=after.astype(np.float32),
                        usable=np.asarray(usable, bool))
    return np.zeros(before.shape[1:], bool)


def capture(lat: float, lon: float, periods: tuple, dest: Path) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    from core import tools
    from core.measure import measure_change
    from core.models import change
    change.changed = _record
    scratch = Path(tempfile.mkdtemp(prefix="b9_"))
    tools._result_path = lambda rid: scratch / f"{rid}.json"          # keep the stored results untouched
    tools.render_evidence = lambda *a, **k: type("E", (), {"path": None})()
    os.environ["B9_DEST"] = str(dest)
    (b0, b1), (a0, a1) = periods
    try:
        r = measure_change(lat=lat, lon=lon, kind="construction", size_km=5.0, change="b9record",
                           before_start=b0, before_end=b1, after_start=a0, after_end=a1)
        return dict(answer=r.answer, reason=r.reason, saved=dest.exists(),
                    clear_before=r.n_clear_images_before, clear_after=r.n_clear_images_after)
    except Exception as e:                                             # recorded; such a chip gives no example
        return dict(answer="error", reason=f"{type(e).__name__}: {str(e)[:200]}", saved=dest.exists())


def _chip_job(c: dict) -> dict:
    d = date.fromisoformat(c["day"])
    periods = ((d - timedelta(days=60), d - timedelta(days=10)), (d - timedelta(days=3), d + timedelta(days=3)))
    dest = OUT / "flood" / f"{c['name']}.npz"
    if dest.exists():
        return {**c, "saved": True, "answer": "stored before"}
    return {**c, **capture(c["lat"], c["lon"], periods, dest)}


def _q_job(q: dict) -> dict:
    ref = q["reference"]
    dest = OUT / "partc" / f"{q['id']}.npz"
    if dest.exists():
        return {"id": q["id"], "saved": True, "answer": "stored before"}
    periods = ((date.fromisoformat(ref["before"][0]), date.fromisoformat(ref["before"][1])),
               (date.fromisoformat(ref["after"][0]), date.fromisoformat(ref["after"][1])))
    return {"id": q["id"], "cluster": q["cluster"], **capture(ref["lat"], ref["lon"], periods, dest)}


def main(which: str, workers: int = 12, reverse: bool = False) -> None:
    (OUT / which.replace("chips", "flood")).mkdir(parents=True, exist_ok=True)
    if which == "chips":
        jobs, fn = chips()[::-1] if reverse else chips(), _chip_job
    else:
        from experiments.b4_exam import questions
        jobs, fn = questions("C"), _q_job
    res = []
    with ProcessPoolExecutor(workers) as pool:
        for k, r in enumerate(pool.map(fn, jobs), 1):
            res.append(r)
            if k % 20 == 0:
                print(f"  {which}: {k}/{len(jobs)}, saved {sum(x['saved'] for x in res)}", flush=True)
    (OUT / f"{which}{'_reverse' if reverse else ''}.json").write_text(json.dumps(res, indent=1))
    print(f"{which}: {len(res)} jobs, {sum(x['saved'] for x in res)} saved")


if __name__ == "__main__":
    main(sys.argv[1], reverse="--reverse" in sys.argv)
