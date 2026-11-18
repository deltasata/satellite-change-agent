"""B1 results table: pooled water IoU of every detector, and paired differences with event-bootstrap intervals.

    python -m experiments.b1_report            # after b1_rules, b1_train --run-all, b1_published
Writes runs/b1/report.json and prints the tables.
"""

from __future__ import annotations

import json

from core.data import sen1floods11 as s1f
from experiments import b1_common as C

RULES = ("ndwi0", "ndwi_fit", "mndwi_fit")
MODELS = ("unet", "prithvi", "terramind")
NAMES = {"ndwi0": "NDWI > 0 (pipeline rule)", "ndwi_fit": "NDWI > t, t fitted", "mndwi_fit": "MNDWI > t, t fitted",
         "unet": "U-Net from nothing", "prithvi": "Prithvi-EO-2.0-300M fine-tuned", "terramind": "TerraMind-1.0-base fine-tuned",
         "published": "Prithvi-EO-2.0 Sen1Floods11, published"}


def model_counts(model: str) -> dict | None:
    folder = C.RUNS / "folds" / model
    files = sorted(folder.glob("*.json"))
    if not files:
        return None
    out = {}
    for f in files:
        out.update(json.loads(f.read_text())["counts"])
    return out


def main():
    report = {}
    # ---- decision test: L2A, leave one event out
    res = {r: C.load_counts(r, "l2a") for r in RULES}
    for m in MODELS:
        c = model_counts(m)
        if c is not None:
            res[m] = c
    chips = sorted(res["ndwi0"])
    events = sorted({s1f.event(c) for c in chips})
    print(f"DECISION TEST: L2A, leave one event out ({len(chips)} tiles, {len(events)} events)")
    print(f"{'detector':34s} {'tiles':>5s} {'IoU':>6s}   vs NDWI>0 [95 % interval]      vs MNDWI fitted [95 % interval]")
    report["decision"] = {}
    for name, c in res.items():
        common = [k for k in chips if k in c]
        d0 = C.bootstrap_difference(res["ndwi0"], c, common)
        d1 = C.bootstrap_difference(res["mndwi_fit"], c, common)
        report["decision"][name] = dict(tiles=len(common), iou=C.pooled_iou(c, common), vs_ndwi0=d0, vs_mndwi_fit=d1,
                                        per_event={e: C.pooled_iou(c, [k for k in common if s1f.event(k) == e]) for e in events})
        note = "" if len(common) == len(chips) else "  (incomplete)"
        print(f"{NAMES[name]:34s} {len(common):5d} {C.pooled_iou(c, common):6.3f}   "
              f"{d0[0]:+.3f} [{d0[1]:+.3f}, {d0[2]:+.3f}]       {d1[0]:+.3f} [{d1[1]:+.3f}, {d1[2]:+.3f}]{note}")
    print("\nper event (IoU):")
    print(f"{'':12s}" + "".join(f"{n[:9]:>10s}" for n in res))
    for e in events:
        print(f"{e:12s}" + "".join(f"{report['decision'][n]['per_event'][e]:10.3f}" for n in res))

    # ---- official test part + Bolivia: comparison with the published model
    parts = s1f.splits()
    report["official"] = {}
    for kind in ("l1c", "l2a"):
        pub = C.load_counts("published", kind)
        ndwi = C.load_counts("ndwi0", kind)
        mndwi = C.load_counts("mndwi_fit", kind)
        tiles = sorted(pub)
        d = C.bootstrap_difference(ndwi, pub, tiles)
        print(f"\nOFFICIAL TEST PART + BOLIVIA, {kind.upper()} ({len(tiles)} tiles): NDWI>0 {C.pooled_iou(ndwi, tiles):.3f}, "
              f"MNDWI fitted {C.pooled_iou(mndwi, tiles):.3f}, published Prithvi {C.pooled_iou(pub, tiles):.3f}; "
              f"published minus NDWI>0 {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
        report["official"][kind] = dict(tiles=len(tiles), ndwi0=C.pooled_iou(ndwi, tiles),
                                        mndwi_fit=C.pooled_iou(mndwi, tiles), published=C.pooled_iou(pub, tiles),
                                        published_vs_ndwi0=d)
        for m in MODELS:   # our leave-one-event-out models on the same tiles (L2A only)
            c = res.get(m) if kind == "l2a" else None
            if c:
                common = [k for k in tiles if k in c]
                report["official"][kind][m] = C.pooled_iou(c, common)
                print(f"    {NAMES[m]} (left its event out) on the same tiles: {C.pooled_iou(c, common):.3f}")
    (C.RUNS / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
