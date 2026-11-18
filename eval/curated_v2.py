"""Curated candidate sites for benchmark v2 (in addition to the 14 curated v1 questions).

Every candidate is a hypothesis until verified: eval/build_benchmark.py --v2 runs the no-LLM
reference on it, and each site is checked by eye on the before/after true-colour composites. Sites whose imagery does not clearly
show what the label claims are listed in REJECTED with the reason; they never enter the
benchmark. Coordinates and dates come from public records of each event or project.

Fields: id, category, expected, place, lat, lon, size_km, before, after, kind, phrase, source
"""

C, F, S, A = "construction", "vegetation_loss", "stable", "any_change"

CANDIDATES = [
    # ---------------- construction / large surface change (expected yes)
    dict(id="build_istanbul_airport", category=C, expected="yes", place="north of Istanbul, Türkiye",
         lat=41.262, lon=28.742, size_km=5, before=("2017-06-01", "2017-08-31"), after=("2019-06-01", "2019-08-31"),
         kind="construction", source="Istanbul Airport, construction 2015-2018, opened 2019"),
    dict(id="build_western_sydney_airport", category=C, expected="yes", place="Badgerys Creek, Sydney, Australia",
         lat=-33.885, lon=150.72, size_km=5, before=("2017-12-01", "2018-02-28"), after=("2021-12-01", "2022-02-28"),
         kind="construction", source="Western Sydney Airport, bulk earthworks 2018-2021"),
    dict(id="build_egypt_new_capital", category=C, expected="yes", place="New Administrative Capital, Egypt",
         lat=30.008, lon=31.698, size_km=5, before=("2018-03-01", "2018-05-31"), after=("2021-03-01", "2021-05-31"),
         kind="construction", source="Central business district of Egypt's New Administrative Capital, built 2018-2021"),
    dict(id="build_tsmc_arizona", category=C, expected="yes", place="north Phoenix, Arizona",
         lat=33.746, lon=-112.123, size_km=3, before=("2020-03-01", "2020-05-31"), after=("2023-03-01", "2023-05-31"),
         kind="construction", source="TSMC Arizona fab, construction from 2021"),
    dict(id="build_intel_ohio", category=C, expected="yes", place="New Albany, Ohio",
         lat=40.073, lon=-82.744, size_km=3, before=("2021-06-01", "2021-08-31"), after=("2024-06-01", "2024-08-31"),
         kind="construction", source="Intel Ohio site, earthworks and construction from 2022"),
    dict(id="build_samsung_taylor", category=C, expected="yes", place="Taylor, Texas",
         lat=30.553, lon=-97.441, size_km=3, before=("2021-06-01", "2021-08-31"), after=("2024-06-01", "2024-08-31"),
         kind="construction", source="Samsung Taylor fab, construction from 2022"),
    dict(id="build_tesla_shanghai", category=C, expected="yes", place="Lingang, Shanghai, China",
         lat=30.883, lon=121.774, size_km=3, before=("2018-10-01", "2018-12-31"), after=("2020-10-01", "2020-12-31"),
         kind="construction", source="Tesla Gigafactory Shanghai, built 2019"),
    dict(id="build_fehmarnbelt_rodby", category=C, expected="yes", place="Rødby, Lolland, Denmark",
         lat=54.645, lon=11.36, size_km=4, before=("2019-06-01", "2019-08-31"), after=("2023-06-01", "2023-08-31"),
         kind="construction", source="Fehmarnbelt tunnel element factory and work harbour, built from 2021"),
    dict(id="water_gerd_reservoir", category=A, expected="yes", place="Blue Nile, Benishangul-Gumuz, Ethiopia",
         lat=11.2, lon=35.12, size_km=5, before=("2020-01-01", "2020-03-31"), after=("2022-01-01", "2022-03-31"),
         kind="construction", source="Grand Ethiopian Renaissance Dam reservoir, filling from July 2020"),
    dict(id="water_kakhovka_drained", category=A, expected="yes", place="Kakhovka reservoir near Nikopol, Ukraine",
         lat=47.52, lon=34.35, size_km=5, before=("2023-04-15", "2023-06-01"), after=("2023-06-20", "2023-08-31"),
         kind="construction", source="Kakhovka dam destroyed 6 June 2023; reservoir drained"),
    dict(id="water_lake_mead_drop", category=A, expected="yes", place="Lake Mead, Nevada",
         lat=36.135, lon=-114.72, size_km=5, before=("2019-06-01", "2019-08-31"), after=("2022-06-01", "2022-08-31"),
         kind="construction", source="Lake Mead record low 2022 (water level ~30 m lower than 2019)"),
    # ---------------- fires (expected yes)
    dict(id="fire_paradise", category=F, expected="yes", place="Paradise, California",
         lat=39.76, lon=-121.62, size_km=5, before=("2018-09-01", "2018-11-07"), after=("2018-11-15", "2018-12-31"),
         kind="vegetation_loss", source="Camp Fire, 8 November 2018"),
    dict(id="fire_greenville_dixie", category=F, expected="yes", place="Greenville, California",
         lat=40.14, lon=-120.95, size_km=5, before=("2021-06-01", "2021-07-20"), after=("2021-08-10", "2021-09-30"),
         kind="vegetation_loss", source="Dixie Fire burned Greenville, 4 August 2021"),
    dict(id="fire_jasper", category=F, expected="yes", place="Jasper, Alberta",
         lat=52.873, lon=-118.08, size_km=5, before=("2024-06-01", "2024-07-20"), after=("2024-07-28", "2024-09-15"),
         kind="vegetation_loss", source="Jasper wildfire, 24 July 2024"),
    dict(id="fire_evia", category=F, expected="yes", place="northern Euboea (Evia), Greece",
         lat=38.88, lon=23.25, size_km=5, before=("2021-06-01", "2021-07-31"), after=("2021-08-20", "2021-10-15"),
         kind="vegetation_loss", source="Evia wildfires, 3-12 August 2021"),
    dict(id="fire_pedrogao", category=F, expected="yes", place="Pedrógão Grande, Portugal",
         lat=39.92, lon=-8.15, size_km=5, before=("2017-04-01", "2017-06-15"), after=("2017-06-25", "2017-08-15"),
         kind="vegetation_loss", source="Pedrógão Grande fire, 17 June 2017"),
    dict(id="fire_kangaroo_island", category=F, expected="yes", place="Flinders Chase, Kangaroo Island, Australia",
         lat=-35.85, lon=136.75, size_km=5, before=("2019-11-01", "2019-12-15"), after=("2020-01-25", "2020-03-15"),
         kind="vegetation_loss", source="Kangaroo Island bushfires, December 2019 - January 2020"),
    dict(id="fire_palisades", category=F, expected="yes", place="Pacific Palisades, Los Angeles",
         lat=34.05, lon=-118.54, size_km=4, before=("2024-11-01", "2025-01-05"), after=("2025-01-15", "2025-02-28"),
         kind="vegetation_loss", source="Palisades Fire, 7 January 2025"),
    dict(id="fire_eaton_altadena", category=F, expected="yes", place="Altadena, Los Angeles",
         lat=34.19, lon=-118.13, size_km=4, before=("2024-11-01", "2025-01-05"), after=("2025-01-15", "2025-02-28"),
         kind="vegetation_loss", source="Eaton Fire, 7 January 2025"),
    # ---------------- stable (expected no): historic centres and old parks, same season 2 years apart
    *[dict(id=f"stable_{k}", category=S, expected="no", place=p, lat=la, lon=lo, size_km=2,
           before=(f"{y}-06-01", f"{y}-08-31"), after=(f"{y + 2}-06-01", f"{y + 2}-08-31"), kind="construction",
           source="historic centre / old park; no known large construction")
      for k, p, la, lo, y in [
          ("hyde_park", "London", 51.507, -0.165, 2020), ("tiergarten", "Berlin", 52.514, 13.35, 2019),
          ("vienna_centre", "Vienna", 48.208, 16.372, 2020), ("prague_old_town", "Prague", 50.087, 14.421, 2019),
          ("national_mall", "Washington, D.C.", 38.889, -77.03, 2020),
          ("paris_luxembourg", "Paris", 48.846, 2.337, 2021), ("amsterdam_canals", "Amsterdam", 52.37, 4.885, 2020),
          ("florence_centre", "Florence", 43.771, 11.255, 2021), ("edinburgh_old_town", "Edinburgh", 55.949, -3.19, 2021),
          ("budapest_centre", "Budapest", 47.507, 19.046, 2020), ("madrid_retiro", "Madrid", 40.415, -3.684, 2021),
          ("boston_common", "Boston", 42.355, -71.066, 2020), ("sydney_botanic", "Sydney", -33.864, 151.216, 2020),
          ("zurich_old_town", "Zurich", 47.372, 8.543, 2021), ("stockholm_gamla_stan", "Stockholm", 59.325, 18.071, 2020),
          ("copenhagen_centre", "Copenhagen", 55.676, 12.578, 2021), ("lisbon_baixa", "Lisbon", 38.711, -9.137, 2020),
          ("barcelona_gothic", "Barcelona", 41.383, 2.177, 2021), ("montreal_mont_royal", "Montreal", 45.504, -73.587, 2020),
          ("sf_golden_gate_park", "San Francisco", 37.769, -122.483, 2021), ("munich_olympiapark", "Munich", 48.173, 11.553, 2020)]],
    dict(id="stable_tokyo_imperial", category=S, expected="no", place="Tokyo", lat=35.685, lon=139.753, size_km=2,
         before=("2019-10-01", "2019-11-30"), after=("2021-10-01", "2021-11-30"), kind="construction",
         source="Imperial Palace grounds; no known large construction"),
    # ---------------- more construction / surface change
    dict(id="build_northvolt_skelleftea", category=C, expected="yes", place="Skellefteå, Sweden",
         lat=64.75, lon=21.06, size_km=3, before=("2018-06-01", "2018-08-31"), after=("2022-06-01", "2022-08-31"),
         kind="construction", source="Northvolt Ett battery factory, built 2019-2022"),
    dict(id="build_bmw_debrecen", category=C, expected="yes", place="Debrecen, Hungary",
         lat=47.48, lon=21.62, size_km=3, before=("2019-06-01", "2019-08-31"), after=("2023-06-01", "2023-08-31"),
         kind="construction", source="BMW Debrecen plant, construction from 2019"),
    dict(id="build_daxing_airport", category=C, expected="yes", place="Daxing, Beijing, China",
         lat=39.509, lon=116.411, size_km=5, before=("2017-04-01", "2017-06-30"), after=("2019-04-01", "2019-06-30"),
         kind="construction", source="Beijing Daxing International Airport, completed 2019"),
    dict(id="water_hkia_third_runway", category=A, expected="yes", place="north of Hong Kong International Airport",
         lat=22.33, lon=113.92, size_km=4, before=("2017-10-01", "2017-12-31"), after=("2020-10-01", "2020-12-31"),
         kind="construction", source="Hong Kong airport third-runway land reclamation, 2016-2020 (no L2A data for 2016 here, so 2017 vs 2020)"),
    # ---------------- more fires
    dict(id="fire_mati", category=F, expected="yes", place="Mati, Attica, Greece",
         lat=38.05, lon=23.995, size_km=4, before=("2018-06-01", "2018-07-20"), after=("2018-07-28", "2018-09-15"),
         kind="vegetation_loss", source="Mati wildfire, 23 July 2018"),
    dict(id="fire_coffey_park", category=F, expected="yes", place="Santa Rosa, California",
         lat=38.476, lon=-122.745, size_km=3, before=("2017-08-15", "2017-10-07"), after=("2017-10-15", "2017-11-30"),
         kind="vegetation_loss", source="Tubbs Fire destroyed Coffey Park, 9 October 2017"),
    dict(id="fire_valparaiso", category=F, expected="yes", place="Viña del Mar, Chile",
         lat=-33.06, lon=-71.49, size_km=4, before=("2023-12-15", "2024-01-31"), after=("2024-02-10", "2024-03-31"),
         kind="vegetation_loss", source="Valparaíso region wildfires, 2-3 February 2024"),
]
# Sydney is in the southern hemisphere: June-August is winter there; same season in both years, so fine.

# Designed "cannot tell" (verified with the catalog: no imagery, or no usable imagery).
CANNOT_TELL = [
    dict(id="ct_future_paris", place="Paris", lat=48.856, lon=2.352, size_km=5, before=("2027-04-01", "2027-05-31"),
         after=("2027-06-01", "2027-07-01"), kind="flood", source="dates in the future"),
    dict(id="ct_future_rome", place="Rome", lat=41.9, lon=12.5, size_km=5, before=("2026-10-15", "2026-11-30"),
         after=("2026-12-01", "2027-01-15"), kind="flood", source="after window in the future"),
    dict(id="ct_pre_s2_austin", place="east of Austin, Texas", lat=30.222, lon=-97.618, size_km=3,
         before=("2013-06-01", "2013-08-31"), after=("2014-06-01", "2014-08-31"), kind="construction",
         source="before Sentinel-2A launch (June 2015)"),
    dict(id="ct_pre_s2_lusail", place="Lusail, Qatar", lat=25.4207, lon=51.4903, size_km=3,
         before=("2008-01-01", "2008-03-31"), after=("2011-01-01", "2011-03-31"), kind="construction",
         source="before Sentinel-2A launch (June 2015)"),
    dict(id="ct_ocean_pacific", place="", lat=0.0, lon=-140.0, size_km=5, before=("2021-06-01", "2021-08-31"),
         after=("2023-06-01", "2023-08-31"), kind="construction", source="open Pacific: not imaged"),
    dict(id="ct_ocean_indian", place="", lat=-30.0, lon=80.0, size_km=5, before=("2021-06-01", "2021-08-31"),
         after=("2023-06-01", "2023-08-31"), kind="construction", source="open Indian Ocean: not imaged"),
    dict(id="ct_ocean_south_atlantic", place="", lat=-40.0, lon=-20.0, size_km=5, before=("2021-06-01", "2021-08-31"),
         after=("2023-06-01", "2023-08-31"), kind="construction", source="open South Atlantic: not imaged"),
    dict(id="ct_polar_longyearbyen", place="Longyearbyen, Svalbard", lat=78.223, lon=15.65, size_km=5,
         before=("2022-11-15", "2022-12-15"), after=("2022-12-16", "2023-01-20"), kind="flood",
         source="polar night"),
    dict(id="ct_polar_utqiagvik", place="Utqiagvik, Alaska", lat=71.29, lon=-156.79, size_km=5,
         before=("2022-11-25", "2022-12-20"), after=("2022-12-21", "2023-01-15"), kind="flood",
         source="polar night"),
    dict(id="ct_antarctic_winter", place="McMurdo Station, Antarctica", lat=-77.85, lon=166.67, size_km=5,
         before=("2023-05-01", "2023-06-15"), after=("2023-06-16", "2023-07-31"), kind="flood",
         source="Antarctic winter (polar night)"),
]

# Filled in after visual verification (id -> reason). Rejected sites never enter the benchmark.
REJECTED: dict[str, str] = {}
