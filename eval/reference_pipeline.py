"""The no-LLM reference: answers each benchmark question with fixed rules,
using the TRUE parameters stored in question["reference"] (agents never see them).

It measures the ceiling the tools allow (if the reference gets a question wrong,
no agent using these tools should be expected to get it right), and it lets us
debug the benchmark and runner without spending any Claude quota.

Decision rules: the same ones written in detect_change's tool description.
"""

from __future__ import annotations

from datetime import date

from core import tools

# Rules FROZEN on 2026-11-13, before any agent run. History: the construction rule first also
# required delta_ndvi <= -0.1; the curated Lusail (desert) site showed that fails where there is
# no vegetation to lose, so it was dropped.
FLOOD_MIN_NEW_WATER = 0.05
CHANGE_MIN_FRACTION = 0.03


def decide(kind: str, r: tools.ChangeOutput) -> str:
    if r.status != "ok":
        return "cannot_tell"
    if kind == "flood":
        return "yes" if (r.water_fraction_after - r.water_fraction_before) >= FLOOD_MIN_NEW_WATER else "no"
    return "yes" if r.changed_fraction >= CHANGE_MIN_FRACTION else "no"


def answer(q: dict, evidence: bool = True) -> dict:
    ref = q["reference"]
    area = dict(lat=ref["lat"], lon=ref["lon"], size_km=ref["size_km"])
    trace = []
    try:
        b = tools.cloud_mask_and_stack(start=date.fromisoformat(ref["before"][0]),
                                       end=date.fromisoformat(ref["before"][1]), **area)
        trace.append(("cloud_mask_and_stack", "before", b.n_dates, b.n_usable_dates))
        a = tools.cloud_mask_and_stack(start=date.fromisoformat(ref["after"][0]),
                                       end=date.fromisoformat(ref["after"][1]), **area)
        trace.append(("cloud_mask_and_stack", "after", a.n_dates, a.n_usable_dates))
    except ValueError as e:  # e.g. future dates
        return dict(answer="cannot_tell", reason=str(e), trace=trace)
    if b.cube_id is None or a.cube_id is None:
        return dict(answer="cannot_tell", reason="no imagery in at least one window", trace=trace)
    r = tools.detect_change(b.cube_id, a.cube_id)
    trace.append(("detect_change", r.status))
    out = dict(answer=decide(ref["kind"], r), change=r.model_dump(mode="json"), trace=trace)
    if evidence and r.status == "ok":
        out["evidence"] = tools.render_evidence(r.result_id, f"{q['id']}  (expected: {q['expected']})").path
    return out
