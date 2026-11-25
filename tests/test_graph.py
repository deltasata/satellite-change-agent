"""Graph wiring tests with the scripted FakeLLM (no model, no quota)."""

import json

import pytest

from agents.llm_backends.base import complete
from agents.llm_backends.fake import FakeLLM
from agents.graph.graph import Plan


def test_backend_contract_validates_and_repairs():
    class Flaky:
        name, n_calls = "flaky", 0

        def raw_json(self, system, prompt, schema):
            self.n_calls += 1
            if self.n_calls == 1:
                return '{"lat": 999}'  # invalid: lat out of range, fields missing
            return json.dumps(dict(lat=1, lon=2, size_km=5, change_type="flood", before_start="2020-01-01",
                                   before_end="2020-02-01", after_start="2020-03-01", after_end="2020-03-05"))

    llm = Flaky()
    p = complete(llm, "sys", "q", Plan)
    assert p.lat == 1 and llm.n_calls == 2  # one repair call


def test_fake_plan_parsing():
    q = ("Was the 5 km x 5 km area centred on 44.5200°N, 11.8300°E (Italy) flooded around "
         "2023-05-23, compared with the weeks before?")
    p = FakeLLM()._plan(q)
    assert p["change_type"] == "flood" and p["after_start"] == "2023-05-20" and p["lat"] == 44.52


@pytest.fixture
def version1(monkeypatch):
    """The Phase A graph (four step tools), kept for experiments E06 and E08."""
    from agents.graph import graph
    monkeypatch.setattr(graph, "GRAPH_VERSION", 1)


@pytest.mark.network
def test_graph_end_to_end_with_fake_llm(version1):
    from agents.graph.run import answer_question

    r = answer_question("Was there new construction in the 3 km x 3 km area centred on 52.3950°N, "
                        "13.7950°E (Grünheide, near Berlin) between summer 2019 and summer 2021?", backend="fake")
    assert r["answer"] == "yes"
    assert r["path"] == ["check_question", "parse_request", "validate_plan", "stack", "detect", "assess_evidence", "render",
                         "write_answer"]
    assert [t["tool"] for t in r["tool_calls"]] == ["cloud_mask_and_stack", "cloud_mask_and_stack",
                                                    "detect_change", "render_evidence"]
    assert r["n_llm_calls"] == 4


@pytest.mark.network
def test_graph_future_dates_abstain_without_tools():
    from agents.graph.run import answer_question

    r = answer_question("Was the 5 km x 5 km area centred on 48.1575°N, 11.5950°E (Munich) flooded around "
                        "2027-03-15, compared with the weeks before?", backend="fake")
    assert r["answer"] == "cannot_tell" and r["tool_calls"] == []


def test_refine_goes_through_validation(version1):
    """A retry must pass validate_plan again (bug found in the first local run)."""
    from agents.graph.graph import build_graph

    g = build_graph(None, None).get_graph()
    targets = {e.target for e in g.edges if e.source == "assess_evidence"}
    assert "validate_plan" in targets and "stack" not in targets


def test_v2_event_windows_follow_the_guide():
    from datetime import date

    from agents.graph.graph import event_windows

    w = event_windows(date(2023, 5, 23))
    assert (str(w["before_start"]), str(w["before_end"]), str(w["after_start"]), str(w["after_end"])) == \
        ("2023-03-24", "2023-05-13", "2023-05-20", "2023-05-26")
    r = event_windows(date(2023, 5, 23), retry=True)
    assert str(r["before_start"]) == "2023-01-23" and str(r["after_end"]) == "2023-05-30"


def test_rule_check_v3():
    from agents.graph.graph import rule_check
    s = rule_check("flood", {"status": "ok", "water_fraction_before": 0.0021, "water_fraction_after": 0.0035,
                             "changed_fraction": 0.12})
    assert "NOT met" in s and "+0.0014" in s
    assert "MET -> the rule says yes" in rule_check("construction", {"status": "ok", "changed_fraction": 0.05})
    assert rule_check("flood", {"status": "insufficient_data"}) is None


def test_off_topic_question_stops_at_the_first_box():
    from agents.graph.run import answer_question

    r = answer_question("How much do I weigh?", backend="fake")
    assert r["answer"] == "out_of_scope"
    assert r["path"] == ["check_question"] and r["tool_calls"] == [] and r["n_llm_calls"] == 1


def test_no_imagery_reason_reaches_the_answer(version1):
    """When a window has no images, the reason must be in the final state (it used to be written
    from inside an arrow function, where LangGraph does not keep it)."""
    import asyncio

    from agents.graph.graph import build_graph
    from agents.llm_backends.fake import FakeLLM

    async def no_images(state, name, args):  # stands in for the MCP server: every window is empty
        state.setdefault("tool_calls", []).append({"tool": name, "input": args, "is_error": False})
        return {"cube_id": None, "n_dates": 0, "n_usable_dates": 0}

    q = ("Was the 5 km x 5 km area centred on 0.0000°N, 0.0000°E flooded around 2023-05-23, "
         "compared with the weeks before?")
    s = asyncio.run(build_graph(FakeLLM(), no_images).ainvoke({"question": q, "tool_calls": [], "path": []}))
    assert s["path"][-2:] == ["stack", "write_answer"]
    assert s["answer"] == "cannot_tell"
    assert s.get("abstain_reason") == "no imagery exists in at least one window"


# ---------------------------------------------------------------- version 4 (the adopted measurement)

def _fake_measure(answer, reason=None, rule=None):
    """Stands in for the measure MCP server: records the call and returns a MeasureResult-like dict."""
    async def call(state, name, args):
        state.setdefault("tool_calls", []).append({"tool": name, "input": args, "is_error": False})
        return {"answer": answer, "reason": reason, "rule": rule, "figure": None, "steps": []}
    return call


def _run(question, tools):
    import asyncio

    from agents.graph.graph import build_graph
    return asyncio.run(build_graph(FakeLLM(), tools).ainvoke({"question": question, "tool_calls": [], "path": []}))


def test_v4_is_the_default_and_measures_in_one_call():
    from agents.graph import graph
    assert graph.GRAPH_VERSION == 4
    s = _run("Was the 5 km x 5 km area centred on 44.5200°N, 11.8300°E (Italy) flooded around 2023-05-23, "
             "compared with the weeks before?", _fake_measure("yes", rule="new water = 0.30 - 0.01 = 0.29"))
    assert s["path"] == ["check_question", "parse_request", "validate_plan", "measure", "write_answer"]
    assert [t["tool"] for t in s["tool_calls"]] == ["measure_change"]
    args = s["tool_calls"][0]["input"]
    # a day wins over the windows the model may also write: measure_change chooses the periods itself
    assert args == {"lat": 44.52, "lon": 11.83, "size_km": 5.0, "kind": "flood", "event_date": "2023-05-23"}
    assert s["answer"] == "yes"


def test_v4_two_periods_and_any_change():
    s = _run("Was there new construction in the 3 km x 3 km area centred on 52.3950°N, 13.7950°E (Grünheide, "
             "near Berlin) between summer 2019 and summer 2021?", _fake_measure("no"))
    args = s["tool_calls"][0]["input"]
    assert args["kind"] == "construction" and "event_date" not in args
    assert (args["before_start"], args["after_end"]) == ("2019-06-01", "2021-08-31")


def test_v4_cannot_tell_keeps_the_reason():
    s = _run("Was the 5 km x 5 km area centred on 0.0000°N, 0.0000°E flooded around 2023-05-23, compared with "
             "the weeks before?", _fake_measure("cannot_tell", reason="no images exist for this place"))
    assert s["answer"] == "cannot_tell" and s["abstain_reason"] == "no images exist for this place"


def test_v4_future_abstains_without_tools():
    s = _run("Was the 5 km x 5 km area centred on 48.1575°N, 11.5950°E (Munich) flooded around 2099-03-15, "
             "compared with the weeks before?", _fake_measure("yes"))
    assert s["tool_calls"] == [] and s["answer"] == "cannot_tell"
    assert s["path"][-2:] == ["validate_plan", "write_answer"]
