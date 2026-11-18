"""A scripted stand-in for an LLM (a 'mock'): regexes and the decision rules, no model.
Used to test the graph's wiring for free. It is NOT a variant and is never benchmarked
as one."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta

SEASONS = {"spring": (3, 1, 5, 31), "summer": (6, 1, 8, 31), "autumn": (10, 1, 11, 30), "winter": (1, 1, 2, 28)}


class FakeLLM:
    name = "fake"

    def __init__(self):
        self.n_calls = 0

    def raw_json(self, system: str, prompt: str, json_schema: dict) -> str:
        self.n_calls += 1
        task = system.split("TASK:")[-1]
        if "procedure above can measure" in task:
            on = bool(re.search(r"°[NS],", prompt))  # scripted: a question with coordinates is on topic
            return json.dumps({"reason": "scripted (fake backend)", "on_topic": "yes" if on else "no"})
        if "measurement plan" in task:
            return json.dumps(self._plan(prompt))
        if "decide from the measurement" in task:
            return json.dumps(self._assess(prompt))
        return json.dumps({"confidence": "medium", "explanation": "scripted explanation (fake backend)"})

    def _plan(self, q: str) -> dict:
        lat, ns, lon, ew = re.search(r"([\d.]+)°([NS]),\s*([\d.]+)°([EW])", q).groups()
        lat = float(lat) * (1 if ns == "N" else -1)
        lon = float(lon) * (1 if ew == "E" else -1)
        m = re.search(r"([\d.]+) km x", q)
        size = float(m.group(1)) if m else 5.0
        kind = "flood" if "flood" in q else "construction"
        m = re.search(r"around (\d{4}-\d{2}-\d{2})", q)
        if m:
            d = date.fromisoformat(m.group(1))
            b0, b1, a0, a1 = d - timedelta(60), d - timedelta(10), d - timedelta(3), d + timedelta(3)
        else:
            s = re.findall(r"(spring|summer|autumn|winter) (\d{4})", q)
            if len(s) == 2:
                (sa, ya), (sb, yb) = s
                ma = SEASONS[sa]; mb = SEASONS[sb]
                b0, b1 = date(int(ya), ma[0], ma[1]), date(int(ya), ma[2], ma[3])
                a0, a1 = date(int(yb), mb[0], mb[1]), date(int(yb), mb[2], mb[3])
            else:
                b0, b1, a0, a1 = date(2023, 6, 1), date(2023, 8, 31), date(2024, 6, 1), date(2024, 8, 31)
        return dict(lat=lat, lon=lon, size_km=size, change_type=kind,
                    event_date=m.group(1) if (m := re.search(r"around (\d{4}-\d{2}-\d{2})", q)) else None,
                    before_start=str(b0), before_end=str(b1),
                    after_start=str(a0), after_end=str(a1))

    def _assess(self, prompt: str) -> dict:
        c = json.loads(re.search(r"detect_change result: (\{.*\})", prompt).group(1))
        plan = json.loads(re.search(r"Plan: (\{.*\})", prompt).group(1))
        left = int(re.search(r"Retries left: (\d+)", prompt).group(1))
        if c["status"] != "ok":
            if left > 0:
                b0 = date.fromisoformat(plan["before_start"]) - timedelta(30)
                return dict(decision="refine", answer=None, confidence="low", new_before_start=str(b0),
                            new_before_end=plan["before_end"], new_after_start=plan["after_start"],
                            new_after_end=plan["after_end"], rationale="insufficient data; widen before window")
            return dict(decision="abstain", answer=None, confidence="medium", rationale="insufficient data")
        if plan["change_type"] == "flood":
            yes = c["water_fraction_after"] - c["water_fraction_before"] >= 0.05
        else:
            yes = c["changed_fraction"] >= 0.03
        return dict(decision="answer", answer="yes" if yes else "no", confidence="medium", rationale="rule")
