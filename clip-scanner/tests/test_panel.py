"""The consensus rule: a merely-good clip must never come out GREAT."""

import math
import random

import pytest

from clip_scanner.config import GATE_NAMES, RUBRIC_CRITERIA, Config
from clip_scanner.judging.gates import GateResult
from clip_scanner.judging.panel import (
    JudgeOutcome,
    assess_judge,
    decide,
    judge_rejects,
    near_miss_key,
    rank_great,
    rubric_mean,
)
from clip_scanner.judging.runner import collect_panel

CFG = Config()
PERSONAS = CFG.judging.judges
DET_OK = [GateResult("length", True, "ok"), GateResult("hook_onset", True, "ok")]
SURVIVES = {"argument_convincing": False, "survival_confidence": 8.0, "decisive_issue": "none"}


def verdict(score=9.0, overall=None, gates_fail=(), vote=None, **crit):
    scores = {c: {"justification": "x", "score": crit.get(c, score)} for c in RUBRIC_CRITERIA}
    overall = score if overall is None else overall
    gates = {g: {"evidence": "e", "passed": g not in gates_fail} for g in GATE_NAMES}
    if vote is None:
        vote = "GREAT" if overall >= 8.5 and not gates_fail else "NOT_GREAT"
    return {"first_two_seconds": "x", "gates": gates, "scores": scores, "biggest_weakness": "x",
            "overall_justification": "x", "overall": overall, "hook_line": "x", "verdict": vote}


def panel(*verdicts):
    return [JudgeOutcome(persona=p, verdict=v) for p, v in zip(PERSONAS, verdicts)]


def test_unanimous_great_survives_devil():
    d = decide("c1", DET_OK, panel(verdict(9), verdict(9.2), verdict(8.8)), SURVIVES, {}, CFG)
    assert d.great and d.stage == "great"
    assert d.min_effective == pytest.approx(8.8)


def test_threshold_is_inclusive_at_exactly_8_5():
    d = decide("c1", DET_OK, panel(verdict(8.5), verdict(8.5), verdict(8.5)), SURVIVES, {}, CFG)
    assert d.great


def test_one_judge_just_below_bar_kills_clip():
    d = decide("c1", DET_OK, panel(verdict(9.5), verdict(9.5), verdict(8.4)), SURVIVES, {}, CFG)
    assert not d.great and d.stage == "panel"
    assert any("overall 8.4" in r for r in d.reasons)


def test_average_above_bar_is_not_enough():
    # mean overall = 8.9 > 8.5, but one judge is at 7.0: unanimity, not averaging
    d = decide("c1", DET_OK, panel(verdict(10), verdict(9.7), verdict(7.0)), SURVIVES, {}, CFG)
    assert not d.great


@pytest.mark.parametrize("gate", GATE_NAMES)
def test_any_single_gate_failure_by_any_judge_kills(gate):
    v_bad = verdict(9.8, gates_fail=(gate,), vote="GREAT")  # even if that judge still voted GREAT
    d = decide("c1", DET_OK, panel(verdict(9.8), v_bad, verdict(9.8)), SURVIVES, {}, CFG)
    assert not d.great
    assert any(gate.split("_")[0] in r.replace(" ", "_") or "gate" in r for r in d.reasons)


def test_majority_gate_policy_tolerates_single_dissent_only():
    cfg = Config.model_validate({"judging": {"gate_policy": "majority"}})
    one = panel(verdict(9.5), verdict(9.5, gates_fail=("standalone",), vote="GREAT"), verdict(9.5))
    two = panel(verdict(9.5, gates_fail=("standalone",), vote="GREAT"),
                verdict(9.5, gates_fail=("standalone",), vote="GREAT"), verdict(9.5))
    assert decide("c1", DET_OK, one, SURVIVES, {}, cfg).great
    assert not decide("c1", DET_OK, two, SURVIVES, {}, cfg).great


def test_inflated_overall_not_backed_by_rubric_is_rejected():
    # Holistic 9.5 but rubric mean ~6.9: the judge's own rubric says "solid, not great".
    v = verdict(6.9, overall=9.5, vote="GREAT")
    d = decide("c1", DET_OK, panel(verdict(9.5), v, verdict(9.5)), SURVIVES, {}, CFG)
    assert not d.great
    assert any("rubric" in r for r in d.reasons)


def test_effective_score_is_min_of_overall_and_rubric_plus_allowance():
    # Hook and payoff clear their floors; the rest are mediocre, so the weighted mean is about 7.6.
    v = verdict(7.11, overall=9.4, vote="GREAT", hook_strength=8.5, payoff=8.5)
    mean = rubric_mean(v["scores"], CFG.judging.weights)
    assert mean == pytest.approx(7.6, abs=0.01)
    a = assess_judge(JudgeOutcome("jaded_scroller", v), CFG)
    assert a.effective == pytest.approx(min(9.4, mean + 1.0), abs=1e-3)
    assert a.ok  # 8.6 effective clears 8.5 and mean clears 7.5
    v2 = verdict(6.85, overall=9.4, vote="GREAT", hook_strength=8.5, payoff=8.5)  # mean ~7.43
    a2 = assess_judge(JudgeOutcome("jaded_scroller", v2), CFG)
    assert not a2.ok and a2.effective < 8.5


def test_judge_voting_not_great_kills_even_with_high_numbers():
    d = decide("c1", DET_OK, panel(verdict(9.5), verdict(9.5, vote="NOT_GREAT"), verdict(9.5)), SURVIVES, {}, CFG)
    assert not d.great and any("NOT_GREAT" in r for r in d.reasons)


@pytest.mark.parametrize("crit", ["hook_strength", "payoff"])
def test_criterion_floors(crit):
    v = verdict(9.6, **{crit: 7.9})
    d = decide("c1", DET_OK, panel(verdict(9.6), v, verdict(9.6)), SURVIVES, {}, CFG)
    assert not d.great and any("floor" in r for r in d.reasons)


def test_missing_judge_is_rejection_not_pass():
    d = decide("c1", DET_OK, panel(verdict(9.8), verdict(9.8)), SURVIVES, {}, CFG)
    assert not d.great
    assert d.stage == "error"


def test_judge_error_or_refusal_is_rejection():
    outs = panel(verdict(9.8), verdict(9.8)) + [JudgeOutcome(PERSONAS[2], error="refusal: declined")]
    d = decide("c1", DET_OK, outs, SURVIVES, {}, CFG)
    assert not d.great and d.stage == "error"


def test_error_plus_real_rejection_is_reported_as_panel_rejection():
    outs = [JudgeOutcome(PERSONAS[0], verdict=verdict(6.0)), JudgeOutcome(PERSONAS[1], error="api: 500"),
            JudgeOutcome(PERSONAS[2], verdict=verdict(9.0))]
    d = decide("c1", DET_OK, outs, SURVIVES, {}, CFG)
    assert not d.great and d.stage == "panel"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10.5, -1.0, "9", None, True])
def test_malformed_scores_fail_closed(bad):
    v = verdict(9.5)
    v["overall"] = bad
    d = decide("c1", DET_OK, panel(verdict(9.5), v, verdict(9.5)), SURVIVES, {}, CFG)
    assert not d.great


def test_malformed_rubric_entry_fails_closed():
    v = verdict(9.5)
    v["scores"]["payoff"]["score"] = math.nan
    assert not decide("c1", DET_OK, panel(verdict(9.5), v, verdict(9.5)), SURVIVES, {}, CFG).great
    v2 = verdict(9.5)
    del v2["gates"]["standalone"]
    assert not decide("c1", DET_OK, panel(verdict(9.5), v2, verdict(9.5)), SURVIVES, {}, CFG).great


def test_unexpected_extra_judge_fails_closed():
    outs = panel(verdict(9), verdict(9), verdict(9)) + [JudgeOutcome("mystery_judge", verdict(9))]
    assert not decide("c1", DET_OK, outs, SURVIVES, {}, CFG).great


def test_deterministic_gate_failure_is_final():
    det = [GateResult("dead_air", False, "longest dead air 2.1s")]
    d = decide("c1", det, panel(verdict(10), verdict(10), verdict(10)), SURVIVES, {}, CFG)
    assert not d.great and d.stage == "deterministic_gates"


@pytest.mark.parametrize("ruling, survives", [
    ({"argument_convincing": True, "survival_confidence": 9.0, "decisive_issue": "needs context"}, False),
    ({"argument_convincing": False, "survival_confidence": 6.9, "decisive_issue": ""}, False),
    ({"argument_convincing": False, "survival_confidence": 7.0, "decisive_issue": ""}, True),
    ({"argument_convincing": "no", "survival_confidence": 9.0}, False),
    ({"argument_convincing": False, "survival_confidence": float("nan")}, False),
    (None, False),
])
def test_devils_advocate_rule(ruling, survives):
    d = decide("c1", DET_OK, panel(verdict(9.5), verdict(9.5), verdict(9.5)), ruling, {}, CFG)
    assert d.great is survives
    if not survives:
        assert d.stage == "devils_advocate"


def test_devil_disabled_skips_that_stage():
    cfg = Config.model_validate({"devils_advocate": {"enabled": False}})
    assert decide("c1", DET_OK, panel(verdict(9), verdict(9), verdict(9)), None, None, cfg).great


def test_rank_and_max_clips_never_fill_with_non_great():
    good = decide("a", DET_OK, panel(verdict(9.0), verdict(9.1), verdict(9.4)), SURVIVES, {}, CFG)
    better = decide("b", DET_OK, panel(verdict(9.6), verdict(9.5), verdict(9.7)), SURVIVES, {}, CFG)
    near = decide("c", DET_OK, panel(verdict(8.4), verdict(9.9), verdict(9.9)), SURVIVES, {}, CFG)
    ranked = rank_great([good, near, better])
    assert [d.candidate_id for d in ranked] == ["b", "a"]
    assert rank_great([near]) == []  # a weak video yields nothing, not "the best of the bunch"


def test_near_miss_ordering_prefers_later_stage():
    devil_kill = decide("d", DET_OK, panel(verdict(9), verdict(9), verdict(9)),
                        {"argument_convincing": True, "survival_confidence": 2.0}, {}, CFG)
    panel_kill = decide("p", DET_OK, panel(verdict(8.4), verdict(9), verdict(9)), SURVIVES, {}, CFG)
    gate_kill = decide("g", [GateResult("length", False, "70s")], None, None, None, CFG)
    order = sorted([gate_kill, panel_kill, devil_kill], key=near_miss_key)
    assert [d.candidate_id for d in order] == ["d", "p", "g"]


def test_rubric_mean_weights():
    scores = {c: {"score": 10.0} for c in RUBRIC_CRITERIA}
    scores["hook_strength"]["score"] = 0.0
    w = CFG.judging.weights
    assert rubric_mean(scores, w) == pytest.approx(10 * (sum(w.values()) - w["hook_strength"]) / sum(w.values()))


# ---- short-circuit must never change the outcome ------------------------------------------------

def _random_verdict(rng):
    if rng.random() < 0.08:
        return None
    base = rng.choice([6.0, 7.5, 8.2, 8.5, 8.8, 9.3])
    crit = {c: min(10.0, max(0.0, base + rng.uniform(-1.2, 1.2))) for c in RUBRIC_CRITERIA}
    fails = tuple(g for g in GATE_NAMES if rng.random() < 0.06)
    v = verdict(base, overall=min(10, base + rng.uniform(-0.5, 0.8)), gates_fail=fails, **crit)
    if rng.random() < 0.1:
        v["verdict"] = "GREAT" if v["verdict"] == "NOT_GREAT" else "NOT_GREAT"
    return v


@pytest.mark.parametrize("policy", ["any", "majority"])
def test_short_circuit_equivalent_to_full_panel(policy):
    cfg = Config.model_validate({"judging": {"gate_policy": policy}})
    rng = random.Random(1234)
    for trial in range(800):
        table = {p: _random_verdict(rng) for p in PERSONAS}

        def judge_fn(p):
            v = table[p]
            return JudgeOutcome(p, verdict=v) if v is not None else JudgeOutcome(p, error="api: 529")

        full = collect_panel(cfg, judge_fn, full_panel=True)
        short = collect_panel(cfg, judge_fn, full_panel=False)
        d_full = decide("c", DET_OK, full, SURVIVES, {}, cfg)
        d_short = decide("c", DET_OK, short, SURVIVES, {}, cfg)
        assert d_full.great == d_short.great, (trial, table)
        if d_short.great:
            assert all(not o.skipped for o in short)


def test_judge_rejects_matches_assessment():
    assert judge_rejects(JudgeOutcome("jaded_scroller", verdict(8.0)), CFG)
    assert not judge_rejects(JudgeOutcome("jaded_scroller", verdict(9.0)), CFG)
    assert judge_rejects(JudgeOutcome("jaded_scroller", error="x"), CFG)


# ---- regressions from the adversarial review -----------------------------------------------------

def test_verdict_from_fallback_model_never_counts():
    """A refusal answered by a different model is not the judge's verdict (review finding #1)."""
    outs = panel(verdict(9.5), verdict(9.5), verdict(9.5))
    outs[1].served_model = "claude-sonnet-5-5"
    d = decide("c1", DET_OK, outs, SURVIVES, {}, CFG)
    assert not d.great and d.stage == "error"
    outs2 = panel(verdict(9.5), verdict(9.5), verdict(9.5))
    outs2[2].fallback = True
    assert not decide("c1", DET_OK, outs2, SURVIVES, {}, CFG).great


def test_dated_snapshot_of_configured_model_is_accepted():
    outs = panel(verdict(9.5), verdict(9.5), verdict(9.5))
    for o in outs:
        o.served_model = "claude-opus-5-5-20260901"
    assert decide("c1", DET_OK, outs, SURVIVES, {}, CFG).great


def test_duplicate_judge_verdicts_fail_closed():
    outs = panel(verdict(9.5), verdict(9.5), verdict(9.5)) + [JudgeOutcome(PERSONAS[0], verdict=verdict(9.6))]
    assert not decide("c1", DET_OK, outs, SURVIVES, {}, CFG).great


def test_majority_policy_counts_judges_failing_any_gate():
    cfg = Config.model_validate({"judging": {"gate_policy": "majority"}})
    outs = panel(verdict(9.5, gates_fail=("standalone",), vote="GREAT"),
                 verdict(9.5, gates_fail=("monetization_safe",), vote="GREAT"), verdict(9.5))
    assert not decide("c1", DET_OK, outs, SURVIVES, {}, cfg).great


def test_devils_advocate_failure_is_incomplete_not_a_quality_verdict():
    d = decide("c1", DET_OK, panel(verdict(9.5), verdict(9.5), verdict(9.5)), None, None, CFG,
               devil_error="arbiter failed: api: 529")
    assert not d.great and d.stage == "error"


def test_panel_rejection_keeps_devil_ruling_for_calibration_sweeps():
    d = decide("c1", DET_OK, panel(verdict(8.0), verdict(9), verdict(9)), SURVIVES, {"x": 1}, CFG)
    assert not d.great and d.arbiter == SURVIVES and d.devil == {"x": 1}
