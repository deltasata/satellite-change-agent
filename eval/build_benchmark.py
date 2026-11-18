"""Build eval/benchmark.jsonl: questions with known answers (yes / no / cannot_tell).

Sources
- Floods: Sen1Floods11 hand-labelled chips (Bonafilia et al. 2020). "yes" = chips where
  >= 15 % of labelled pixels are water that is NOT permanent water (JRC layer); "no" =
  chips from the same event and date with ~0 water. India (2016) predates the L2A
  archive we use, so India chips are "cannot_tell".
- Construction / vegetation loss / stable areas: curated sites, each verified visually
  on our own before/after composites.
- Designed "cannot_tell": no data can exist (future date, before Sentinel-2, open
  ocean, polar night) or clouds leave too little (verified with the catalog/pipeline).

Each question has:
  id, category, expected, question  - what an agent sees (only `question`!)
  reference                          - structured parameters for the no-LLM reference
                                       pipeline (true windows); agents never see this
  source                             - provenance of the expected answer

Run:  source activate.sh && python -m eval.build_benchmark [--v2]
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S1F = ROOT / "data_cache" / "sen1floods11" / "chip_stats.json"
OUT = ROOT / "eval" / "benchmark.jsonl"

EVENT_NAMES = {"Bolivia": "Bolivia", "Ghana": "Ghana", "India": "India", "Mekong": "Cambodia",
               "Nigeria": "Nigeria", "Pakistan": "Pakistan", "Paraguay": "Paraguay", "Somalia": "Somalia",
               "Spain": "Spain", "Sri-Lanka": "Sri Lanka", "USA": "the USA"}


def fmt_ll(lat: float, lon: float) -> str:
    return f"{abs(lat):.4f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.4f}°{'E' if lon >= 0 else 'W'}"


def flood_questions(n_yes: int = 2, n_no: int = 1, india_yes: int | None = None, india_no: int | None = None) -> list[dict]:
    """Per event: the n_yes most-flooded and the n_no most-completely-labelled dry chips.
    Rankings are fixed, so a larger cap gives a superset (v1 ids are preserved in v2)."""
    rows = json.loads(S1F.read_text())
    out = []
    for ev in sorted({r["event"] for r in rows}):
        ev_rows = [r for r in rows if r["event"] == ev]
        ny = india_yes if (ev == "India" and india_yes is not None) else n_yes
        nn = india_no if (ev == "India" and india_no is not None) else n_no
        yes = sorted([r for r in ev_rows if r["valid_frac"] >= 0.8 and r["flood_frac"] >= 0.15],
                     key=lambda r: -r["flood_frac"])[:ny]
        no = sorted([r for r in ev_rows if r["valid_frac"] >= 0.9 and r["flood_frac"] <= 0.005
                     and r["perm_frac"] <= 0.01 and r["water_frac"] <= 0.01],
                    key=lambda r: -r["valid_frac"])[:nn]
        for label, chosen in (("yes", yes), ("no", no)):
            for k, r in enumerate(chosen):
                lat = (r["bbox"][1] + r["bbox"][3]) / 2
                lon = (r["bbox"][0] + r["bbox"][2]) / 2
                d = date.fromisoformat(r["date"])
                expected = "cannot_tell" if ev == "India" else label
                out.append(dict(
                    id=f"flood_{ev.lower()}_{label}{k}", category="flood", expected=expected, cluster=f"flood_{ev}",
                    question=(f"Was the 5 km x 5 km area centred on {fmt_ll(lat, lon)} "
                              f"({EVENT_NAMES[ev]}) flooded around {d.isoformat()}, compared with "
                              f"the weeks before?"),
                    reference=dict(kind="flood", lat=round(lat, 5), lon=round(lon, 5), size_km=5.0,
                                   before=[str(d - timedelta(days=60)), str(d - timedelta(days=10))],
                                   after=[str(d - timedelta(days=3)), str(d + timedelta(days=3))]),
                    source=(f"Sen1Floods11 hand label {r['chip']}: flood (non-permanent water) fraction "
                            f"{r['flood_frac']:.3f}, permanent water {r['perm_frac']:.3f}"
                            + ("; no L2A imagery exists for India 2016 in the archive" if ev == "India" else ""))))
    return out


# Curated sites. Coordinates and dates from public records; each verified on our composites.
CURATED = [
    dict(id="build_tesla_gruenheide", category="construction", expected="yes",
         question="Was there new construction in the 3 km x 3 km area centred on 52.3950°N, 13.7950°E "
                  "(Grünheide, near Berlin) between summer 2019 and summer 2021?",
         reference=dict(kind="construction", lat=52.395, lon=13.795, size_km=3.0,
                        before=["2019-06-01", "2019-08-31"], after=["2021-06-01", "2021-08-31"]),
         source="Tesla Gigafactory Berlin-Brandenburg, built 2020-2021; verified in E03"),
    dict(id="build_tesla_austin", category="construction", expected="yes",
         question="Did large-scale construction take place in the 3 km x 3 km area centred on 30.2220°N, "
                  "97.6180°W (east of Austin, Texas) between summer 2019 and summer 2021?",
         reference=dict(kind="construction", lat=30.222, lon=-97.618, size_km=3.0,
                        before=["2019-06-01", "2019-08-31"], after=["2021-06-01", "2021-08-31"]),
         source="Tesla Gigafactory Texas, construction from mid-2020"),
    dict(id="build_hinkley_point_c", category="construction", expected="yes",
         question="Has the 3 km x 3 km area centred on 51.2066°N, 3.1325°W (Somerset coast, UK) changed "
                  "through construction between summer 2017 and summer 2020?",
         reference=dict(kind="construction", lat=51.2066, lon=-3.1325, size_km=3.0,
                        before=["2017-06-01", "2017-08-31"], after=["2020-06-01", "2020-08-31"]),
         source="Hinkley Point C nuclear power station, main construction 2017-"),
    dict(id="build_lusail_stadium", category="construction", expected="yes",
         question="Was a large structure built in the 3 km x 3 km area centred on 25.4207°N, 51.4903°E "
                  "(Lusail, Qatar) between early 2018 and early 2021?",
         reference=dict(kind="construction", lat=25.4207, lon=51.4903, size_km=3.0,
                        before=["2018-01-01", "2018-03-31"], after=["2021-01-01", "2021-03-31"]),
         source="Lusail Stadium, built 2017-2021"),
    dict(id="fire_lahaina", category="vegetation_loss", expected="yes",
         question="Was there major damage or loss of vegetation in the 3 km x 3 km area centred on "
                  "20.8783°N, 156.6825°W (Lahaina, Maui) in August 2023?",
         reference=dict(kind="vegetation_loss", lat=20.8783, lon=-156.6825, size_km=3.0,
                        before=["2023-06-01", "2023-08-05"], after=["2023-08-10", "2023-09-30"]),
         source="Lahaina wildfire, 8 August 2023"),
    dict(id="fire_lytton", category="vegetation_loss", expected="yes",
         question="Did a wildfire burn the 5 km x 5 km area centred on 50.2316°N, 121.5824°W "
                  "(Lytton, British Columbia) in summer 2021?",
         reference=dict(kind="vegetation_loss", lat=50.2316, lon=-121.5824, size_km=5.0,
                        before=["2021-05-15", "2021-06-28"], after=["2021-07-05", "2021-08-31"]),
         source="Lytton wildfire, 30 June 2021"),
    dict(id="stable_englischer_garten", category="stable", expected="no",
         question="Was there any significant land-surface change in the 2 km x 2 km area centred on "
                  "48.1575°N, 11.5950°E (Munich) between summer 2023 and summer 2024?",
         reference=dict(kind="construction", lat=48.1575, lon=11.595, size_km=2.0,
                        before=["2023-06-01", "2023-08-31"], after=["2024-06-01", "2024-08-31"]),
         source="Englischer Garten park; no known change; null test in E03"),
    dict(id="stable_central_park", category="stable", expected="no",
         question="Was there new construction in the 2 km x 2 km area centred on 40.7812°N, 73.9665°W "
                  "(New York City) between summer 2020 and summer 2022?",
         reference=dict(kind="construction", lat=40.7812, lon=-73.9665, size_km=2.0,
                        before=["2020-06-01", "2020-08-31"], after=["2022-06-01", "2022-08-31"]),
         source="Central Park; no known large construction"),
    dict(id="stable_rome_centre", category="stable", expected="no",
         question="Did the 2 km x 2 km area centred on 41.8986°N, 12.4769°E (Rome) change through "
                  "construction between summer 2019 and summer 2021?",
         reference=dict(kind="construction", lat=41.8986, lon=12.4769, size_km=2.0,
                        before=["2019-06-01", "2019-08-31"], after=["2021-06-01", "2021-08-31"]),
         source="historic centre of Rome; protected, no large construction"),
    dict(id="stable_kyoto_palace", category="stable", expected="no",
         question="Was there new construction in the 2 km x 2 km area centred on 35.0254°N, 135.7621°E "
                  "(Kyoto) between autumn 2019 and autumn 2021?",
         reference=dict(kind="construction", lat=35.0254, lon=135.7621, size_km=2.0,
                        before=["2019-10-01", "2019-11-30"], after=["2021-10-01", "2021-11-30"]),
         source="Kyoto Imperial Palace park; no known large construction"),
    # ---- designed "cannot tell"
    dict(id="ct_future", category="cannot_tell", expected="cannot_tell",
         question="Has the 5 km x 5 km area centred on 48.1575°N, 11.5950°E (Munich) flooded between "
                  "1 March 2027 and 1 May 2027?",
         reference=dict(kind="flood", lat=48.1575, lon=11.595, size_km=5.0,
                        before=["2027-01-15", "2027-02-28"], after=["2027-03-01", "2027-05-01"]),
         source="dates in the future: no imagery can exist"),
    dict(id="ct_before_sentinel2", category="cannot_tell", expected="cannot_tell",
         question="Was there new construction in the 3 km x 3 km area centred on 52.3950°N, 13.7950°E "
                  "(Grünheide, near Berlin) between summer 2010 and summer 2012?",
         reference=dict(kind="construction", lat=52.395, lon=13.795, size_km=3.0,
                        before=["2010-06-01", "2010-08-31"], after=["2012-06-01", "2012-08-31"]),
         source="Sentinel-2A launched June 2015: no imagery for these dates"),
    dict(id="ct_open_ocean", category="cannot_tell", expected="cannot_tell",
         question="Did anything get built in the 5 km x 5 km area centred on 35.0000°N, 40.0000°W "
                  "between summer 2022 and summer 2023?",
         reference=dict(kind="construction", lat=35.0, lon=-40.0, size_km=5.0,
                        before=["2022-06-01", "2022-08-31"], after=["2023-06-01", "2023-08-31"]),
         source="open Atlantic: Sentinel-2 does not image the open ocean"),
    dict(id="ct_polar_night", category="cannot_tell", expected="cannot_tell",
         question="Was the 5 km x 5 km area centred on 69.6490°N, 18.9550°E (Tromsø, Norway) flooded "
                  "between 5 and 31 December 2023, compared with November 2023?",
         reference=dict(kind="flood", lat=69.649, lon=18.955, size_km=5.0,
                        before=["2023-11-01", "2023-11-30"], after=["2023-12-05", "2023-12-31"]),
         source="polar night: no usable optical imagery (verified)"),
]


MONTHS = "January February March April May June July August September October November December".split()


def period(a: str, b: str, lat: float = 1.0) -> str:
    """Natural phrase for a window: 'summer 2019' for Jun-Aug in the north, else 'March-May 2018'
    (June-August is winter in the southern hemisphere)."""
    da, db = date.fromisoformat(a), date.fromisoformat(b)
    if (da.month, da.day, db.month, db.day) == (6, 1, 8, 31) and da.year == db.year and lat >= 0:
        return f"summer {da.year}"
    if da.year == db.year and da.month != db.month:
        return f"{MONTHS[da.month - 1]}-{MONTHS[db.month - 1]} {da.year}"
    if db.year == da.year + 1:
        return f"{MONTHS[da.month - 1]} {da.year}-{MONTHS[db.month - 1]} {db.year}"
    return f"{a} to {b}"


def curated_v2_questions() -> list[dict]:
    from eval.curated_v2 import CANDIDATES, CANNOT_TELL, REJECTED

    out = []
    for c in CANDIDATES:
        if c["id"] in REJECTED:
            continue
        where = f"the {c['size_km']:g} km x {c['size_km']:g} km area centred on {fmt_ll(c['lat'], c['lon'])} ({c['place']})"
        b, a = period(*c["before"], lat=c["lat"]), period(*c["after"], lat=c["lat"])
        if c["category"] == "vegetation_loss":
            ev = date.fromisoformat(c["before"][1]) + (date.fromisoformat(c["after"][0]) - date.fromisoformat(c["before"][1])) / 2
            q = f"Did a wildfire burn or damage {where} in {MONTHS[ev.month - 1]} {ev.year}?"
        elif c["category"] == "any_change":
            q = f"Did the land or water surface change substantially in {where} between {b} and {a}?"
        else:
            q = f"Was there new construction in {where} between {b} and {a}?"
        out.append(dict(id=c["id"], category=c["category"], expected=c["expected"], cluster=c["id"], question=q,
                        reference=dict(kind=c["kind"], lat=c["lat"], lon=c["lon"], size_km=float(c["size_km"]),
                                       before=list(c["before"]), after=list(c["after"])), source=c["source"]))
    for c in CANNOT_TELL:
        where = f"the {c['size_km']:g} km x {c['size_km']:g} km area centred on {fmt_ll(c['lat'], c['lon'])}" + \
                (f" ({c['place']})" if c["place"] else "")
        b, a = period(*c["before"], lat=c["lat"]), period(*c["after"], lat=c["lat"])
        if c["kind"] == "flood":
            q = f"Was {where} flooded between {c['after'][0]} and {c['after'][1]}, compared with {b}?"
        else:
            q = f"Was there new construction in {where} between {b} and {a}?"
        out.append(dict(id=c["id"], category="cannot_tell", expected="cannot_tell", cluster=c["id"], question=q,
                        reference=dict(kind=c["kind"], lat=c["lat"], lon=c["lon"], size_km=float(c["size_km"]),
                                       before=list(c["before"]), after=list(c["after"])), source=c["source"]))
    return out


def with_clusters(qs):
    for q in qs:
        q.setdefault("cluster", q["id"])
    return qs


def write(qs, out):
    from collections import Counter
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "duplicate ids"
    out.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs))
    print(f"{len(qs)} questions -> {out}")
    print("by expected:", dict(Counter(q["expected"] for q in qs)))
    print("by category:", dict(Counter(q["category"] for q in qs)))
    print("clusters:", len({q["cluster"] for q in qs}))


def main():
    import sys
    if "--v2" in sys.argv:
        qs = flood_questions(n_yes=6, n_no=6, india_yes=4, india_no=2) + CURATED + curated_v2_questions()
        write(with_clusters(qs), ROOT / "eval" / "benchmark_v2.jsonl")
    else:
        write(with_clusters(flood_questions() + CURATED), OUT)


if __name__ == "__main__":
    main()
