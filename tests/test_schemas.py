from datetime import date

import pytest
from pydantic import ValidationError

from core.schemas import BBox, SearchRequest

MUNICH = dict(west=11.50, south=48.10, east=11.60, north=48.17)


def test_valid_bbox():
    assert BBox(**MUNICH).as_list() == [11.50, 48.10, 11.60, 48.17]


@pytest.mark.parametrize(
    "bad",
    [
        dict(MUNICH, west=11.7),  # west east of east
        dict(MUNICH, south=48.2),  # south north of north
        dict(MUNICH, north=95.0),  # latitude out of range
    ],
)
def test_invalid_bbox_rejected(bad):
    with pytest.raises(ValidationError):
        BBox(**bad)


def test_dates_must_be_ordered():
    with pytest.raises(ValidationError):
        SearchRequest(bbox=BBox(**MUNICH), start=date(2024, 9, 1), end=date(2024, 3, 1))


def test_cloud_cover_range():
    with pytest.raises(ValidationError):
        SearchRequest(
            bbox=BBox(**MUNICH), start=date(2024, 3, 1), end=date(2024, 9, 1), max_cloud_cover=150
        )


def test_strings_from_an_llm_are_parsed():
    # An LLM will send JSON strings; pydantic converts them to typed values or refuses.
    req = SearchRequest.model_validate(
        {"bbox": MUNICH, "start": "2024-03-01", "end": "2024-09-30", "max_cloud_cover": "40"}
    )
    assert req.start == date(2024, 3, 1) and req.max_cloud_cover == 40.0
