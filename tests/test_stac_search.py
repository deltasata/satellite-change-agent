"""Search tests. Offline ones use real catalog items saved in tests/fixtures/."""

import json
from datetime import date
from pathlib import Path

import pytest

from core.data.stac_search import (
    deduplicate,
    item_to_scene,
    scenes_from_items,
    search_imagery,
)
from core.schemas import BBox, SearchRequest

FIXTURE = Path(__file__).parent / "fixtures" / "earthsearch_items.json"
MUNICH = BBox(west=11.50, south=48.10, east=11.60, north=48.17)


@pytest.fixture
def items():
    return json.loads(FIXTURE.read_text())


def test_item_to_scene_reads_calibration(items):
    by_id = {it["id"]: item_to_scene(it) for it in items}
    old = by_id["S2B_32UPU_20210630_0_L2A"]  # original processing, baseline 03.01
    new = by_id["S2B_32UPU_20210630_1_L2A"]  # reprocessed, baseline 05.00
    # The catalog advertises offset -0.1 for the new one, but Earth Search already
    # removed it from the pixels (flag boa_offset_applied). Effective offset: 0 for both.
    assert new.offset_already_in_pixels and not old.offset_already_in_pixels
    assert old.reflectance_offset == new.reflectance_offset == 0.0
    assert old.reflectance_scale == new.reflectance_scale == pytest.approx(1e-4)
    assert old.tile == new.tile == "MGRS-32UPU"


def test_deduplicate_keeps_latest_processing(items):
    scenes = [item_to_scene(it) for it in items]
    kept = deduplicate(scenes)
    assert len(kept) == 2  # the 2021 pair collapses to one; 2024 stays
    kept_2021 = [s for s in kept if s.datetime.year == 2021]
    assert kept_2021[0].id == "S2B_32UPU_20210630_1_L2A"


def test_scenes_from_items_accounts_for_everything(items):
    req = SearchRequest(bbox=MUNICH, start=date(2021, 1, 1), end=date(2024, 12, 31), max_cloud_cover=20)
    res = scenes_from_items(items, req)
    # conservation: found = duplicates + too cloudy + kept
    assert res.n_found == res.n_duplicates_removed + res.n_too_cloudy + len(res.scenes)
    assert all(s.cloud_cover <= 20 for s in res.scenes)
    assert [s.datetime for s in res.scenes] == sorted(s.datetime for s in res.scenes)


@pytest.mark.network
def test_live_search_munich_2024():
    req = SearchRequest(bbox=MUNICH, start=date(2024, 3, 1), end=date(2024, 9, 30))
    res = search_imagery(req)
    assert res.n_found > 40  # at least the naive 5-day revisit
    assert all(date(2024, 3, 1) <= s.datetime.date() <= date(2024, 9, 30) for s in res.scenes)
    assert len({(s.tile, s.datetime.date()) for s in res.scenes}) == len(res.scenes)  # no duplicates


def test_unreadable_duplicate_loses_to_readable_copy(items):
    """Same tile, day and baseline: sequence 1 only in the requester-pays bucket must lose."""
    import copy

    base = next(it for it in items if it["id"].endswith("20240929_0_L2A"))
    paid = copy.deepcopy(base)
    paid["id"] = base["id"].replace("_0_", "_1_")
    paid["properties"]["s2:sequence"] = "1"
    for k in ("blue", "green", "red", "nir", "swir16", "scl"):
        paid["assets"][k]["href"] = "s3://sentinel-s2-l2a/tiles/x.jp2"
    kept = deduplicate([item_to_scene(base), item_to_scene(paid)])
    assert [s.id for s in kept] == [base["id"]]
    req = SearchRequest(bbox=MUNICH, start=date(2024, 9, 1), end=date(2024, 9, 30))
    res = scenes_from_items([paid], req)  # only the paid copy exists -> dropped and counted
    assert res.scenes == [] and res.n_unreadable == 1


def test_unflagged_2022_tile_gets_no_offset(items):
    """Baseline 04.00 tiles NOT flagged as corrected are also offset-free in the pixels (measured)."""
    import copy

    it = copy.deepcopy(next(i for i in items if i["id"].endswith("20240929_0_L2A")))
    it["properties"]["earthsearch:boa_offset_applied"] = False
    it["properties"]["s2:processing_baseline"] = "04.00"
    s = item_to_scene(it)
    assert s.reflectance_offset == 0.0 and not s.offset_already_in_pixels
