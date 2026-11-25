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
# instead of 25 (interval of the difference includes zero).
# Milestone B9, avenue 1 (22 November 2026), protocol in docs/METHODS.md section 12: the same network's map minus
# the pixels the water network (B1) sees as water in the after image. On the 78 flood questions asked as
# construction questions, false "yes" fell from 21 to 6 (-0.192, interval -0.333..-0.072; each flood event
# scored with a water network that never saw it); urban change on OSCD unchanged (IoU 0.304 against 0.301,
# +0.003, interval -0.007..+0.017; allowed loss 0.03); exam: one answer fewer right (the GERD reservoir, an
# "any change" question where new water is the change), within the allowed 1.
CHANGE_DETECTOR = "terramind+water"
# Which pixels lost vegetation (burnt), for "vegetation loss" questions. Milestone B5, protocol in
# docs/METHODS.md section 8: a U-Net trained from nothing on ImpactMesh-Fire (CEMS burnt-area labels, our
# own Earth Search median images, fires near exam fires left out) beat the rule on 42 held-out fires (IoU of
# "burnt" 0.772 against 0.050, +0.722, interval +0.671..+0.773). A Siamese TerraMind was within its interval
# (0.793), so the cheaper network was taken. Exam v2: 0.860 -> 0.887 (all 13 fires right instead of 9); on
# 570 unburnt patches 3.9 % false "yes" instead of 5.1 % (interval of the difference includes zero).
# Milestone B8 (21 November 2026), protocol in docs/METHODS.md section 11: on 44 new fires of 2026, later than
# every training fire (CEMS maps, labels by ImpactMesh's rule), B7's U-TAE on every clear date beat the U-Net
# (IoU of "burnt" 0.859 against 0.838, +0.021, interval +0.012..+0.035); a U-TAE that also sees radar and
# elevation did as well (0.859) but not better, and has more parameters and inputs, so the smaller one was
# taken. False "yes" on 701 unburnt new patches 0.6 % against 0.7 %; exam v2 unchanged (0.887, all 13 fires
# right). It runs on the CPU in about 10-25 s per 5 km area instead of 0.5 s.
VEGETATION_DETECTOR = "burn-utae"

# If at least this share of the measured pixels was seen cloud-free only once after the event, the answer
# carries a caution (Claude's choice, 24 November 2026; a reporting rule, it does not change any answer).
ONE_LOOK_CAUTION = 0.5

# Milestone B10 (docs/METHODS.md section 13): which images a fire question asked about a day (or a month) uses.
# "fixed": the flood periods of event_periods (before day 1 of B10); "training": the burn network's training
# periods, then its wide ones; "nearest8": within the wide ones, the 8 useful images nearest the fire on each side.
# B10 result (25 November 2026), 228 fire questions with known answers from 55 fires: right 0.855 with "fixed",
# 0.996 with "training" and with "nearest8" (+0.140 [+0.076, +0.221]); false "yes" 0.000 -> 0.009 (upper end
# +0.028, margin 0.03). Both passed and gave the same answers, so by the protocol the 8-nearest rule was taken.
FIRE_SELECTION = "nearest8"
FIRE_PERIODS = {"normal": ((-60, -3), (7, 60)), "wide": ((-120, -3), (7, 90))}   # days from D1 (before), D2 (after)
NEAREST_N, NEAREST_MIN_CLEAR = 8, 0.10

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
    event_date: date | None = Field(None, description="the day of the event, if the question names one; the first "
                                                      "day of its span if the question names a span (a month)")
    event_end: date | None = Field(None, description="the last day of the event's span (a month), if any")
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
    n_clear_images_before: int | None = Field(description="mostly clear images: at least 80 % of the area cloud-free. "
                                              "A summary only; the measurement itself works pixel by pixel, so it can "
                                              "succeed with none (clouds in different places on different days)")
    n_images_after: int | None
    n_clear_images_after: int | None
    measurement: dict | None = Field(description="all numbers from detect_change")
    figure: str | None = Field(description="path of the evidence figure (PNG)")
    water_detector: str | None = Field(None, description="how water pixels were found (flood questions)")
    cloud_detector: str | None = Field(None, description="how cloud and shadow pixels were found (optical images)")
    change_detector: str | None = Field(None, description="how changed pixels were found (construction and vegetation-loss questions)")
    sensor: str | None = Field(None, description="'optical' (Sentinel-2) or 'radar' (Sentinel-1, used for floods "
                                                 "when clouds hide the ground)")
    n_radar_days_before: int | None = None
    n_radar_days_after: int | None = None
    caution: str | None = Field(None, description="a weakness of the evidence the answer should mention, if any")
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


def _fire_measure(x: MeasureInput, out: dict, steps: list[str], area: dict, bands, cloud: str, change: str,
                  selection: str) -> MeasureResult:
    """Milestone B10: a fire question about a day D0 (D1 = D2 = D0) or a span D1..D2, with the images chosen by
    `selection` ('training' or 'nearest8', docs/METHODS.md section 13)."""
    d1 = x.event_date
    d2 = x.event_end or d1
    (wb0, wb1), (wa0, wa1) = FIRE_PERIODS["wide"]
    b0, b1 = d1 + timedelta(days=wb0), d1 + timedelta(days=wb1)
    a0, a1 = d2 + timedelta(days=wa0), min(d2 + timedelta(days=wa1), date.today())
    out.update(before_period=[b0.isoformat(), b1.isoformat()], after_period=[a0.isoformat(), a1.isoformat()])
    if a0 > date.today():
        out["reason"] = "too early: a burn scar is measured from 7 days after the fire, and that day is in the future"
        steps.append("dates checked: after period in the future -> cannot tell")
        return MeasureResult(**out)
    if b1 < SENTINEL2_FIRST_DATA:
        out["reason"] = "a period lies before July 2015, when Sentinel-2 began taking images"
        steps.append("dates checked: before Sentinel-2 -> cannot tell")
        return MeasureResult(**out)
    b = tools.cloud_mask_and_stack(start=b0, end=b1, bands=bands, cloud=cloud, **area)
    a = tools.cloud_mask_and_stack(start=a0, end=a1, bands=bands, cloud=cloud, **area)
    steps.append(f"images searched: before {b0}..{b1}: {b.n_dates}; after {a0}..{a1}: {a.n_dates}")
    if b.cube_id is None or a.cube_id is None:
        out["reason"] = "no images exist for this place in at least one period (for example open sea)"
        return MeasureResult(**out)
    tb, ta = tools.cube_dates(b.cube_id), tools.cube_dates(a.cube_id)
    if selection == "nearest8":
        kb = sorted([d for d, v in tb if v >= NEAREST_MIN_CLEAR], key=lambda d: (d1 - d).days)[:NEAREST_N]
        ka = sorted([d for d, v in ta if v >= NEAREST_MIN_CLEAR], key=lambda d: (d - d2).days)[:NEAREST_N]
        tries = [(kb, ka, f"the {NEAREST_N} images nearest the fire with at least {NEAREST_MIN_CLEAR:.0%} of the "
                          f"area cloud-free (fewer if fewer exist)")]
    elif selection == "training":
        (nb0, _), (_, na1) = FIRE_PERIODS["normal"]
        nb, na = d1 + timedelta(days=nb0), d2 + timedelta(days=na1)
        tries = [([d for d, _ in tb if d >= nb], [d for d, _ in ta if d <= na], "every image of the burn network's "
                  "training periods (day -60 to -3 and +7 to +60)"),
                 ([d for d, _ in tb], [d for d, _ in ta], "every image of its wide periods (day -120 to -3 and +7 to +90)")]
    else:
        raise ValueError(f"unknown image selection {selection!r}")
    for k, (kb, ka, words) in enumerate(tries):
        sb, sa = tools.subset_cube(b.cube_id, kb), tools.subset_cube(a.cube_id, ka)
        out.update(retried=k > 0, n_images_before=sb.n_dates, n_clear_images_before=sb.n_usable_dates,
                   n_images_after=sa.n_dates, n_clear_images_after=sa.n_usable_dates)
        if kb and ka:
            out.update(before_period=[min(kb).isoformat(), max(kb).isoformat()],
                       after_period=[min(ka).isoformat(), max(ka).isoformat()])
        steps.append(f"images used: {words}: before {len(kb)}, of which {sb.n_usable_dates} mostly clear; "
                     f"after {len(ka)}, of which {sa.n_usable_dates} mostly clear")
        if sb.cube_id is None or sa.cube_id is None:
            out.update(answer="cannot_tell", rule=None, reason="no useful image in at least one period")
            continue
        c = tools.detect_change(sb.cube_id, sa.cube_id, water="ndwi", change=change)
        out["measurement"] = c.model_dump(mode="json", exclude={"result_id", "before_window", "after_window"})
        steps.append(f"pixel by pixel: {c.fraction_pixels_usable:.0%} of the ground was seen cloud-free at least "
                     f"twice before and at least once after (at least 50 % is needed)")
        ans, rule = decide(x.kind, c)
        steps.append(f"measured: status {c.status}; rule: {rule} -> {ans}")
        out.update(answer=ans, rule=rule, reason=None if c.status == "ok" else rule)
        if c.status == "ok":
            if (c.fraction_one_look_after or 0) >= ONE_LOOK_CAUTION:
                out["caution"] = (f"thin evidence after the fire: {c.fraction_one_look_after:.0%} of the measured "
                                  f"ground was seen cloud-free only once, so thin cloud or haze missed by the cloud "
                                  f"mask could hide a change or fake one")
                steps.append("caution: " + out["caution"])
            out["sensor"] = "optical"
            out["figure"] = tools.render_evidence(c.result_id, f"{x.kind} at {x.lat:.4f}, {x.lon:.4f}").path
            return MeasureResult(**out)
    return MeasureResult(**out)


def measure_change(water: str | None = None, radar: str | None = "ALL", cloud: str | None = None,
                   change: str | None = None, selection: str | None = None, **kwargs) -> MeasureResult:
    """Measure one change at one place. `water` overrides the flood water detector (for tests only:
    'ndwi', or 'unet-without-<event>' for a question about a Sen1Floods11 event). `radar` is the radar
    model for floods under cloud: 'ALL' (default), '<event>' (that event left out; tests only), or None (off).
    `cloud` overrides the cloud detector ('scl' or 'ocm'; default CLOUD_DETECTOR). `change` overrides the
    change detector of construction questions ('rule', 'fsd', 'terramind', or '<model>-without-<group>' for
    a place inside an OSCD city; default CHANGE_DETECTOR), and of vegetation-loss questions ('rule',
    'burn-unet', 'burn-terramind', 'burn-utae'; default VEGETATION_DETECTOR). `selection` chooses the images of
    a vegetation-loss question asked about a day or a span ('fixed', 'training', 'nearest8'; default
    FIRE_SELECTION; milestone B10)."""
    cloud = cloud or CLOUD_DETECTOR
    x = MeasureInput(**kwargs)
    water = (water or FLOOD_WATER_DETECTOR) if x.kind == "flood" else "ndwi"
    change = ((change or CHANGE_DETECTOR) if x.kind == "construction" else
              (change or VEGETATION_DETECTOR) if x.kind == "vegetation_loss" else "rule")
    bands = BANDS7 if (water != "ndwi" or change != "rule") else tools.DEFAULT_BANDS
    area = dict(lat=x.lat, lon=x.lon, size_km=x.size_km)
    steps: list[str] = []
    out = dict(answer="cannot_tell", kind=x.kind, rule=None, reason=None, retried=False,
               before_period=None, after_period=None, n_images_before=None, n_clear_images_before=None,
               n_images_after=None, n_clear_images_after=None, measurement=None, figure=None, steps=steps,
               water_detector=WATER_NAMES.get(water, water) if x.kind == "flood" else None,
               change_detector=CHANGE_NAMES.get(change.split("-without-")[0], change) if x.kind != "flood" else None,
               cloud_detector={"scl": "ESA scene classification (SCL)", "ocm": "OmniCloudMask (milestone B3)"}[cloud])

    # The one retry with wider periods is allowed only when the periods came from an event date.
    may_widen = x.event_date is not None and not (x.before_start or x.before_end or x.after_start or x.after_end)
    selection = selection or FIRE_SELECTION
    if x.kind == "vegetation_loss" and may_widen and selection != "fixed":
        return _fire_measure(x, out, steps, area, bands, cloud, change, selection)
    if x.event_end is not None and x.event_end != x.event_date:
        x = x.model_copy(update={"event_date": x.event_date + (x.event_end - x.event_date) / 2})   # the middle day
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
        steps.append(f"images: before {b0}..{b1}: {b.n_dates}, of which {b.n_usable_dates} mostly clear; "
                     f"after {a0}..{a1}: {a.n_dates}, of which {a.n_usable_dates} mostly clear "
                     f"(mostly clear: at least {tools.USABLE_CLEAR_FRACTION:.0%} of the area cloud-free)")
        if b.cube_id is None or a.cube_id is None:
            out["reason"] = "no images exist for this place in at least one period (for example open sea)"
            break

        c = tools.detect_change(b.cube_id, a.cube_id, water=water, change=change)
        out["measurement"] = c.model_dump(mode="json", exclude={"result_id", "before_window", "after_window"})
        steps.append(f"pixel by pixel: {c.fraction_pixels_usable:.0%} of the ground was seen cloud-free at least "
                     f"twice before and at least once after (at least 50 % is needed)")
        ans, rule = decide(x.kind, c)
        steps.append(f"measured: status {c.status}; rule: {rule} -> {ans}")
        if c.status == "ok" and (c.fraction_one_look_after or 0) >= ONE_LOOK_CAUTION:
            out["caution"] = (f"thin evidence after the event: {c.fraction_one_look_after:.0%} of the measured ground "
                              f"was seen cloud-free only once, so thin cloud or haze missed by the cloud mask "
                              f"could hide a change or fake one")
            steps.append("caution: " + out["caution"])
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
