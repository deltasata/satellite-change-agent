"""Milestone B6: the fitted thresholds, the logistic regression and leaving one group out, on made-up
questions (no data, no network)."""

import numpy as np

from experiments import b6_fit as F


def q(i, kind, value, expected, cluster=None, **extra):
    return dict(id=str(i), cluster=cluster or str(i), expected=expected, answer="yes" if value >= F.TODAY[kind] else "no",
                m=dict(kind=kind, value=value, **extra))


def test_threshold_keeps_todays_value_when_it_already_answers_everything():
    rows = [q(1, "construction", 0.10, "yes"), q(2, "construction", 0.001, "no")]
    assert F.fit_thresholds(rows)["construction"] == 0.03


def test_threshold_moves_when_it_helps():
    rows = [q(i, "construction", 0.04, "no") for i in range(3)] + [q(9, "construction", 0.30, "yes")]
    th = F.fit_thresholds(rows)["construction"]
    assert 0.04 < th <= 0.30 and th == 0.05      # the smallest change of today's value that fixes all three


def test_only_yes_questions_keep_todays_threshold():
    rows = [q(i, "vegetation_loss", v, "yes") for i, v in enumerate((0.2, 0.5, 0.9))]
    assert F.fit_thresholds(rows)["vegetation_loss"] == 0.03


def test_logistic_regression_separates_a_clear_case():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2, 0.5, 50), rng.normal(2, 0.5, 50)])[:, None]
    y = np.r_[np.zeros(50), np.ones(50)]
    w = F.logistic(x, y)
    assert w[1] > 0 and abs(w[0]) < 1.0


def test_unmeasured_questions_keep_their_answer():
    r = dict(id="a", cluster="a", expected="cannot_tell", answer="cannot_tell", m=None)
    assert F.answer_thresholds(r, F.TODAY) == "cannot_tell"
    assert F.answer_logistic(r, {}) == "cannot_tell"


def test_leave_one_group_out_never_fits_on_the_answered_group():
    seen = []

    def fit(train):
        seen.append({r["cluster"] for r in train})
        return F.TODAY
    rows = [q(1, "construction", 0.1, "yes", "g1"), q(2, "construction", 0.1, "yes", "g1"), q(3, "construction", 0.0, "no", "g2")]
    F.cross_validated(rows, fit, F.answer_thresholds)
    assert seen == [{"g2"}, {"g1"}]


def test_flood_measurement_is_new_water():
    r = dict(kind="flood", sensor="radar", answer="yes", measurement=dict(water_fraction_before=0.1, water_fraction_after=0.3))
    m = F.measured(r)
    assert m["kind"] == "flood_radar" and abs(m["value"] - 0.2) < 1e-12
