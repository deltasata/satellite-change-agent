"""Text and schemas shared by ALL agent variants, so that the variants differ only in
orchestration (and model), never in what they are told."""

from __future__ import annotations

from datetime import date

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no", "cannot_tell"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "explanation": {"type": "string", "description": "2-4 sentences citing the measured numbers"},
        "evidence_path": {"type": ["string", "null"], "description": "PNG path from render_evidence, or null"},
    },
    "required": ["answer", "confidence", "explanation", "evidence_path"],
    "additionalProperties": False,
}

DOMAIN_GUIDE = f"""You answer questions about land-surface change (floods, construction, fire or
vegetation loss) using Sentinel-2 satellite imagery through the provided tools. Today is
{date.today().isoformat()}.

How to measure:
1. Read the area (centre lat/lon, size in km; default 5 km) and the dates from the question.
2. Choose a BEFORE window and an AFTER window (each at most 120 days):
   - a flood "around date D": before = about D-60 to D-10 days, after = D-3 to D+3 days
     (floods are short-lived; widen the after window to D+7 only if it has no clear dates).
   - change "between period A and period B" (e.g. summer 2019 and summer 2021): before = A,
     after = B, ideally the same season of the year (vegetation changes with the seasons).
   - an event "in month M": before = the 1-2 months before the event, after = the weeks after.
3. cloud_mask_and_stack for the before window, then for the after window (same lat, lon, size_km).
4. detect_change(before_cube_id, after_cube_id) and read its interpretation guide.
5. If status is 'insufficient_data', you may retry ONCE with a wider window; otherwise answer
   cannot_tell.
6. render_evidence for the final result when status is 'ok'.

When to answer cannot_tell: the dates are in the future or before 2016, no imagery exists for a
window (e.g. open ocean, polar night), or clouds leave too few clear observations even after
one retry. Abstaining is correct and valued when the data cannot support an answer; guessing is
penalised.

Decision rules (same for everyone):
- flood: yes if water_fraction_after - water_fraction_before >= 0.05, else no.
- construction / fire / vegetation loss / any change: yes if changed_fraction >= 0.03, else no.
Keep the number of tool calls small."""
