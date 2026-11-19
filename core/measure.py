"""measure_change: the whole measurement as one plain Python function, with no language model inside.

Whoever calls it (the chat agent, a test, or a person) supplies the place, the kind of change and
the dates. Everything else is fixed code: choose the periods, check them, get and cloud-mask the
images, measure, apply the decision rule, retry once with wider periods if clouds left too few
clear images, and draw the figure. No language model is involved, so the same inputs always give
the same result.

    from core.measure import measure_change
    r = measure_change(lat=44.5063, lon=11.8276, kind="flood", event_date="2023-05-23")
    r.answer   # "yes"
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from core import tools
from core.models.change import NAMES as CHANGE_NAMES
from core.models.water import BANDS7, NAMES as WATER_NAMES, RADAR_NAME

# The decision rules, frozen on 2026-11-13 and shared with every agent variant.
FLOOD_MIN_NEW_WATER = 0.05
CHANGE_MIN_FRACTION = 0.03
SENTINEL2_FIRST_DATA = date(2015, 7, 1)  # Sentinel-2A sent its first images in June 2015
# Which pixels are water, for flood questions. Milestone B1 (18 November 2026): the U-Net beat NDWI > 0
# on held-out flood events (water IoU 0.833 against 0.690) and on the exam's flood questions, so it
# replaced the rule here. The other kinds of change are unchanged.
FLOOD_WATER_DETECTOR = "unet"
# Milestone B2 (18 November 2026): when clouds (or a lack of optical images) leave a flood question
# without an answer, radar answers it. On the exam's 72 flood questions outside India this raised the
# accuracy from 0.722 to 0.889 (+0.167, interval +0.031..+0.346). Radar water: TerraMind (core.models.water).
MIN_RADAR_USABLE = 0.5
# Which pixels are cloud or shadow (core/data/timeseries.py). Milestone B3 (18 November 2026): OmniCloudMask
# beat ESA's scene classification (SCL) on 953 expert-labelled test images (IoU of "unusable" 0.885 against
# 0.644, +0.241, interval +0.221..+0.261), and on exam v2 the answers did not become worse (150 questions:
# 0.853 -> 0.860, +0.007, interval -0.015..+0.029). The lower-level functions (core.tools) keep "scl" as
# their default, so earlier recorded results still describe the code.
CLOUD_DETECTOR = "ocm"
# Which pixels changed, for construction questions. Milestone B4 (19 November 2026), protocol in
# docs/METHODS.md section 7: a Siamese TerraMind, fine-tuned on OSCD's urban-change labels, beat the rule
# (change larger than 4 times the before images' date-to-date scatter) on 24 cities left out one group at a
# time (IoU of "changed" 0.301 against 0.188, +0.114, interval +0.052..+0.180). On exam v2 the answers did not become worse
# (0.860 -> 0.860), and on the 78 flood questions asked as construction questions it gave 21 false "yes"
# instead of 25 (interval of the difference includes zero). "Vegetation loss" keeps the rule.
CHANGE_DETECTOR = "terramind"

Kind = Literal["flood", "construction", "vegetation_loss"]


def event_periods(d: date, wider: bool = False) -> tuple[tuple[date, date], tuple[date, date]]:
    """The periods for an event on day d (the rule written in agents/common.py, step 2).
    before: d-60 .. d-10 days; after: d-3 .. d+3. Wider (the one retry): d-120 .. d-10 and d-3 .. d+7."""
    before = (d - timedelta(days=120 if wider else 60), d - timedelta(days=10))
    after = (d - timedelta(days=3), d + timedelta(days=7 if wider else 3))
    return before, after


class MeasureInput(BaseModel):
    lat: float = Field(ge=-85, le=85, description="latitude of the area centre, degrees north")
    lon: float = Field(ge=-180, le=180, description="longitude of the area centre, degrees east")
    kind: Kind = Field(description="flood, construction, or vegetation_loss (fire, clearing)")
    event_date: date | None = Field(None, description="the day of the event, if the question names one")
    before_start: date | None = None
    before_end: date | None = None
    after_start: date | None = None
    after_end: date | None = None
    size_km: float = Field(5.0, ge=0.5, le=20, description="side of the square area, km")


class MeasureResult(BaseModel):
    answer: Literal["yes", "no", "cannot_tell"]
    kind: Kind
    rule: str | None = Field(description="the decision rule written out with the measured numbers")
    reason: str | None = Field(description="why the answer is cannot_tell, if it is")
    retried: bool = Field(description="true if the periods were widened once because of clouds")
    before_period: list[str] | None = Field(description="first and last day actually used")
    after_period: list[str] | None
    n_images_before: int | None
    n_clear_images_before: int | None = Field(description="images with at least 80 % of the area cloud-free")
    n_images_after: int | None
    n_clear_images_after: int | None
    measurement: dict | None = Field(description="all numbers from detect_change")
    figure: str | None = Field(description="path of the evidence figure (PNG)")
    water_detector: str | None = Field(None, description="how water pixels were found (flood questions)")
    cloud_detector: str | None = Field(None, description="how cloud and shadow pixels were found (optical images)")
    change_detector: str | None = Field(None, description="how changed pixels were found (construction questions)")
    sensor: str | None = Field(None, description="'optical' (Sentinel-2) or 'radar' (Sentinel-1, used for floods "
                                                 "when clouds hide the ground)")
    n_radar_days_before: int | None = None
    n_radar_days_after: int | None = None
    steps: list[str] = Field(description="what was done, in order")


def decide(kind: str, change: tools.ChangeOutput) -> tuple[str, str]:
    """The decision rule. Returns the answer and the rule written out with the numbers."""
    if change.status != "ok":
        return "cannot_tell", f"too few clear images ({change.reason})"
    if kind == "flood":
        wb, wa = change.water_fraction_before, change.water_fraction_after
        d = wa - wb
        ans = "yes" if d >= FLOOD_MIN_NEW_WATER else "no"
        return ans, (f"new water = {wa:.4f} - {wb:.4f} = {d:.4f}, which is "
                     f"{'at least' if ans == 'yes' else 'below'} the threshold {FLOOD_MIN_NEW_WATER}")
    cf = change.changed_fraction
    ans = "yes" if cf >= CHANGE_MIN_FRACTION else "no"
    return ans, (f"changed fraction = {cf:.4f}, which is "
                 f"{'at least' if ans == 'yes' else 'below'} the threshold {CHANGE_MIN_FRACTION}")


def _periods(x: MeasureInput, wider: bool) -> tuple[tuple[date, date], tuple[date, date]]:
    """The before and after periods: given ones win; missing ones come from the event date."""
    if x.event_date is not None:
        (b0, b1), (a0, a1) = event_periods(x.event_date, wider)
    else:
        b0 = b1 = a0 = a1 = None
    b0, b1 = x.before_start or b0, x.before_end or b1
    a0, a1 = x.after_start or a0, x.after_end or a1
    if None in (b0, b1, a0, a1):
        raise ValueError("give event_date, or all four of before_start, before_end, after_start, after_end")
    return (b0, b1), (a0, a1)


def radar_answer(out: dict, steps: list[str], bbox: list[float], periods, fold: str = "ALL") -> None:
    """Flood questions only: answer from Sentinel-1 radar over the given periods, if radar can tell.
    Changes `out` in place; leaves it unchanged (cannot tell) if radar cannot tell either."""
    from core.data.radar import load_radar
    from core.models.water import radar_water_fraction
    (b0, b1), (a0, a1) = periods
    rb, ra = load_radar(bbox, b0, b1), load_radar(bbox, a0, a1)
    out.update(n_radar_days_before=len(rb.days), n_radar_days_after=len(ra.days))
    steps.append(f"radar: before {b0}..{b1}: {len(rb.days)} days; after {a0}..{a1}: {len(ra.days)} days")
    if rb.x is None or ra.x is None:
        steps.append("radar: no image in at least one period -> cannot tell")
        return
    fb, fa, usable = radar_water_fraction(rb.x, ra.x, fold)
    if usable < MIN_RADAR_USABLE:
        steps.append(f"radar: only {usable:.0%} of the area has radar values in both periods -> cannot tell")
        return
    d = fa - fb
    ans = "yes" if d >= FLOOD_MIN_NEW_WATER else "no"
    rule = (f"radar: new water = {fa:.4f} - {fb:.4f} = {d:.4f}, which is "
            f"{'at least' if ans == 'yes' else 'below'} the threshold {FLOOD_MIN_NEW_WATER}")
    steps.append(f"{rule} -> {ans}")
    out.update(answer=ans, rule=rule, reason=None, sensor="radar", water_detector=RADAR_NAME,
               before_period=[b0.isoformat(), b1.isoformat()], after_period=[a0.isoformat(), a1.isoformat()],
               measurement=dict(water_fraction_before=fb, water_fraction_after=fa, usable_fraction=usable,
                                radar_days_before=rb.days, radar_days_after=ra.days))


def measure_change(water: str | None = None, radar: str | None = "ALL", cloud: str | None = None,
                   change: str | None = None, **kwargs) -> MeasureResult:
    """Measure one change at one place. `water` overrides the flood water detector (for tests only:
    'ndwi', or 'unet-without-<event>' for a question about a Sen1Floods11 event). `radar` is the radar
    model for floods under cloud: 'ALL' (default), '<event>' (that event left out; tests only), or None (off).
    `cloud` overrides the cloud detector ('scl' or 'ocm'; default CLOUD_DETECTOR). `change` overrides the
    change detector of construction questions ('rule', 'fsd', 'terramind', or '<model>-without-<group>' for
    a place inside an OSCD city; default CHANGE_DETECTOR)."""
    cloud = cloud or CLOUD_DETECTOR
    x = MeasureInput(**kwargs)
    water = (water or FLOOD_WATER_DETECTOR) if x.kind == "flood" else "ndwi"
    change = (change or CHANGE_DETECTOR) if x.kind == "construction" else "rule"
    bands = BANDS7 if (water != "ndwi" or change != "rule") else tools.DEFAULT_BANDS
    area = dict(lat=x.lat, lon=x.lon, size_km=x.size_km)
    steps: list[str] = []
    out = dict(answer="cannot_tell", kind=x.kind, rule=None, reason=None, retried=False,
               before_period=None, after_period=None, n_images_before=None, n_clear_images_before=None,
               n_images_after=None, n_clear_images_after=None, measurement=None, figure=None, steps=steps,
               water_detector=WATER_NAMES.get(water, water) if x.kind == "flood" else None,
               change_detector=CHANGE_NAMES.get(change.split("-without-")[0], change) if x.kind == "construction" else None,
               cloud_detector={"scl": "ESA scene classification (SCL)", "ocm": "OmniCloudMask (milestone B3)"}[cloud])

    # The one retry with wider periods is allowed only when the periods came from an event date.
    may_widen = x.event_date is not None and not (x.before_start or x.before_end or x.after_start or x.after_end)
    first_periods = None
    for wider in ((False, True) if may_widen else (False,)):
        (b0, b1), (a0, a1) = _periods(x, wider)
        today = date.today()
        out.update(before_period=[b0.isoformat(), b1.isoformat()], after_period=[a0.isoformat(), a1.isoformat()],
                   retried=wider)
        if a0 > today:
            out["reason"] = "the after period is in the future, so no images can exist yet"
            steps.append("dates checked: in the future -> cannot tell")
            return MeasureResult(**out)
        a1 = min(a1, today)
        if b1 >= a0:
            raise ValueError(f"the before period must end before the after period starts ({b1} >= {a0})")
        for name, (s, e) in (("before", (b0, b1)), ("after", (a0, a1))):
            if e < s:
                raise ValueError(f"the {name} period ends before it starts ({s} .. {e})")
            if (e - s).days > tools.MAX_WINDOW_DAYS:
                raise ValueError(f"the {name} period {s} .. {e} is {(e - s).days} days long; "
                                 f"the limit is {tools.MAX_WINDOW_DAYS}")
        out["after_period"][1] = a1.isoformat()
        first_periods = first_periods or ((b0, b1), (a0, a1))
        if a1 < SENTINEL2_FIRST_DATA or b1 < SENTINEL2_FIRST_DATA:
            out["reason"] = "a period lies before July 2015, when Sentinel-2 began taking images"
            steps.append("dates checked: before Sentinel-2 -> cannot tell")
            return MeasureResult(**out)

        b = tools.cloud_mask_and_stack(start=b0, end=b1, bands=bands, cloud=cloud, **area)
        a = tools.cloud_mask_and_stack(start=a0, end=a1, bands=bands, cloud=cloud, **area)
        out.update(n_images_before=b.n_dates, n_clear_images_before=b.n_usable_dates,
                   n_images_after=a.n_dates, n_clear_images_after=a.n_usable_dates)
        steps.append(f"images: before {b0}..{b1}: {b.n_dates} ({b.n_usable_dates} clear); "
                     f"after {a0}..{a1}: {a.n_dates} ({a.n_usable_dates} clear)")
        if b.cube_id is None or a.cube_id is None:
            out["reason"] = "no images exist for this place in at least one period (for example open sea)"
            break

        c = tools.detect_change(b.cube_id, a.cube_id, water=water, change=change)
        out["measurement"] = c.model_dump(mode="json", exclude={"result_id", "before_window", "after_window"})
        ans, rule = decide(x.kind, c)
        steps.append(f"measured: status {c.status}; rule: {rule} -> {ans}")
        out.update(answer=ans, rule=rule, reason=None if c.status == "ok" else rule)
        if c.status == "ok":
            out["sensor"] = "optical"
            out["figure"] = tools.render_evidence(c.result_id, f"{x.kind} at {x.lat:.4f}, {x.lon:.4f}").path
            return MeasureResult(**out)
        if may_widen and not wider:
            steps.append("too few clear images -> one retry with wider periods")
    # Optical images could not tell. For floods, radar over the first (not widened) periods, as tested in B2.
    if x.kind == "flood" and radar and first_periods:
        bb = tools.Area(**area).bbox()
        radar_answer(out, steps, [bb.west, bb.south, bb.east, bb.north], first_periods, radar)
    return MeasureResult(**out)
