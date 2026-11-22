"""measure_change and the two code checks (no Claude, no model server)."""

from datetime import date

import pytest

from agents.chat import checks
from core.measure import event_periods, measure_change


# ---------------------------------------------------------------- measure_change, no images needed

def test_event_periods_follow_the_written_rule():
    (b0, b1), (a0, a1) = event_periods(date(2023, 5, 23))
    assert (b0, b1, a0, a1) == (date(2023, 3, 24), date(2023, 5, 13), date(2023, 5, 20), date(2023, 5, 26))
    (b0, _), (_, a1) = event_periods(date(2023, 5, 23), wider=True)
    assert (b0, a1) == (date(2023, 1, 23), date(2023, 5, 30))


def test_future_is_cannot_tell_without_downloading():
    r = measure_change(lat=48.1, lon=11.6, kind="flood", event_date="2099-03-15")
    assert r.answer == "cannot_tell" and "future" in r.reason and r.n_images_before is None


def test_before_sentinel2_is_cannot_tell():
    r = measure_change(lat=52.4, lon=13.8, kind="construction", before_start="2010-06-01",
                       before_end="2010-08-31", after_start="2012-06-01", after_end="2012-08-31")
    assert r.answer == "cannot_tell" and "2015" in r.reason


def test_missing_dates_are_refused_with_a_useful_message():
    with pytest.raises(ValueError, match="event_date, or all four"):
        measure_change(lat=44.5, lon=11.8, kind="flood")


def test_any_is_not_a_kind():
    with pytest.raises(ValueError):
        measure_change(lat=44.5, lon=11.8, kind="any", event_date="2023-05-23")


def test_periods_longer_than_120_days_are_refused():
    with pytest.raises(ValueError, match="limit is 120"):
        measure_change(lat=44.5, lon=11.8, kind="construction", before_start="2019-01-01",
                       before_end="2019-12-31", after_start="2021-06-01", after_end="2021-08-31")


def _flood_networks_present() -> bool:
    """The flood path needs the cloud networks (B3) and the water U-Net (B1) as ONNX files. They are made
    from trained weights, which the public copy does not contain."""
    from core.models import cloud, water
    files = [cloud.MODELS / f"ocm_v4_{n}.onnx" for n in cloud.NETS] + [water.MODELS / "water_unet_ALL.onnx"]
    return all(f.exists() for f in files)


@pytest.mark.network
@pytest.mark.skipif(not _flood_networks_present(), reason="cloud or water ONNX files not made (experiments/b1_export, b3_export)")
def test_conselice_flood_is_measured_yes():
    r = measure_change(lat=44.5063, lon=11.8276, kind="flood", event_date="2023-05-23")
    assert r.answer == "yes" and r.before_period == ["2023-03-24", "2023-05-13"] and r.figure


# ---------------------------------------------------------------- the number check

RESULT = ('{"answer": "yes", "rule": "new water = 0.1486 - 0.0012 = 0.1474, which is at least the threshold 0.05", '
          '"before_period": ["2023-03-24", "2023-05-13"], "n_images_before": 11, "n_clear_images_before": 3, '
          '"measurement": {"changed_area_km2": 1.062}}')


@pytest.mark.parametrize("reply", [
    "Answer: yes\nWater rose from 0.12% to 14.9% of the area (new water 14.7%, threshold 5%).",
    "Answer: yes\nWater fraction went from 0.0012 to 0.15; 3 clear images before 2023-05-13.",
    "Answer: yes\nAbout 1.06 km² changed, seen by Sentinel-2 in band B11.",
    "Answer: yes\nAbout 15 % of the area was under water.",
])
def test_numbers_from_the_result_pass(reply):
    assert checks.unsupported_numbers(reply, [RESULT]) == []


def test_an_invented_number_is_caught():
    assert checks.unsupported_numbers("Answer: yes\nWater covered 23.5% of the area.", [RESULT]) == ["23.5"]


def test_a_misused_number_is_not_caught():
    """A known limit: 11 is in the result (11 images), so '11 clear images' passes, though only 3 were clear."""
    assert checks.unsupported_numbers("Answer: yes\nIt rests on 11 clear images.", [RESULT]) == []


def test_answer_must_equal_the_last_result():
    assert checks.answer_mismatch("Answer: no\n...", {"answer": "yes"})
    assert checks.answer_mismatch("Answer: cannot tell\n...", {"answer": "cannot_tell"}) is None
    assert "never called" in checks.answer_mismatch("Answer: yes", None)


def test_fallback_reply_uses_only_the_result():
    r = {"answer": "yes", "rule": "new water = 0.1486 - 0.0012 = 0.1474, which is at least the threshold 0.05",
         "before_period": ["2023-03-24", "2023-05-13"], "after_period": ["2023-05-20", "2023-05-26"],
         "n_clear_images_before": 3, "n_clear_images_after": 1, "figure": "x.png"}
    text = checks.fallback_reply(r)
    assert text.startswith("Answer: yes.") and checks.unsupported_numbers(text, [str(r)]) == []


def test_todays_date_is_allowed():
    """The one false alarm of the exam: Claude wrote 'today is 17 November 2026' from its instructions."""
    assert checks.unsupported_numbers("Today is 17 November 2026.", ["2026-11-17"]) == []
