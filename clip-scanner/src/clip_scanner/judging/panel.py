"""The consensus rule. Pure functions, unit-tested.

A clip is GREAT only if ALL of the following hold:
  1. every deterministic gate passes;
  2. every configured judge returned a valid verdict (missing, error or refusal means reject);
  3. no judge failed any LLM gate (gate_policy="any"; "majority" lets a single dissent on a
     gate through, but every other check stays unanimous);
  4. every judge voted GREAT, and every judge's overall is >= overall_min, its weighted rubric
     mean is >= rubric_mean_min, its effective score min(overall, rubric_mean + allowance) is
     >= overall_min, and every per-criterion floor is met;
  5. the devil's advocate ran, and the independent arbiter found the case against NOT
     convincing, with survival confidence >= min_survival_confidence.
Anything unexpected (NaN, out of range, missing field) is a rejection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from ..config import GATE_NAMES, RUBRIC_CRITERIA, Config
from .gates import GateResult

GATE_LABELS = {
    "hook_in_first_2s": "hook in first 2s",
    "standalone": "standalone",
    "clear_payoff": "clear payoff",
    "no_dead_air": "no dead air",
    "monetization_safe": "monetization/TOS safe",
}


@dataclass
class JudgeOutcome:
    persona: str
    verdict: dict | None = None      # validated JudgeVerdict.model_dump()
    error: str | None = None
    served_model: str = ""
    skipped: bool = False            # not run because an earlier judge already rejected (short-circuit)


@dataclass
class JudgeAssessment:
    persona: str
    ok: bool
    overall: float | None
    rubric_mean: float | None
    effective: float | None
    failed_gates: list[str]
    reasons: list[str]
    unavailable: bool = False        # no verdict at all (error/refusal/skipped), as opposed to a low verdict


@dataclass
class Decision:
    candidate_id: str
    great: bool
    stage: str                       # deterministic_gates | panel | devils_advocate | great | error
    reasons: list[str]               # why it was killed (first is the decisive one)
    deterministic: list[dict] = field(default_factory=list)
    judges: list[dict] = field(default_factory=list)
    assessments: list[dict] = field(default_factory=list)
    devil: dict | None = None
    arbiter: dict | None = None
    min_effective: float | None = None
    mean_effective: float | None = None
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 10.0


def rubric_mean(scores: dict, weights: dict[str, float]) -> float | None:
    total_w = sum(weights[c] for c in RUBRIC_CRITERIA)
    acc = 0.0
    for c in RUBRIC_CRITERIA:
        entry = scores.get(c)
        s = entry.get("score") if isinstance(entry, dict) else None
        if not _finite(s):
            return None
        acc += weights[c] * s
    return acc / total_w if total_w > 0 else None


def failed_gates(verdict: dict) -> list[str] | None:
    gates = verdict.get("gates")
    if not isinstance(gates, dict):
        return None
    failed = []
    for g in GATE_NAMES:
        entry = gates.get(g)
        if not isinstance(entry, dict) or not isinstance(entry.get("passed"), bool):
            return None
        if not entry["passed"]:
            failed.append(g)
    return failed


def assess_judge(outcome: JudgeOutcome, cfg: Config) -> JudgeAssessment:
    """Score-side checks for one judge. Gate failures are reported, and the policy applies them."""
    j = cfg.judging
    p = outcome.persona
    if outcome.verdict is None:
        why = outcome.error or ("not run" if outcome.skipped else "no verdict")
        return JudgeAssessment(p, False, None, None, None, [], [f"{p}: no valid verdict ({why})"], unavailable=True)
    v = outcome.verdict
    reasons: list[str] = []
    overall = v.get("overall")
    scores = v.get("scores") if isinstance(v.get("scores"), dict) else {}
    mean = rubric_mean(scores, j.weights)
    fg = failed_gates(v)
    if fg is None:
        return JudgeAssessment(p, False, None, None, None, [], [f"{p}: malformed gates"])
    if not _finite(overall) or mean is None:
        return JudgeAssessment(p, False, None, mean, None, fg, [f"{p}: malformed or out-of-range scores"])
    effective = min(overall, mean + j.consistency_allowance)
    if v.get("verdict") != "GREAT":
        reasons.append(f"{p}: voted NOT_GREAT (overall {overall:.1f})")
    if overall < j.overall_min:
        reasons.append(f"{p}: overall {overall:.1f} < {j.overall_min:.1f}")
    if mean < j.rubric_mean_min:
        reasons.append(f"{p}: rubric mean {mean:.2f} < {j.rubric_mean_min:.2f}")
    if effective < j.overall_min and overall >= j.overall_min:
        reasons.append(f"{p}: overall {overall:.1f} not backed by its own rubric (mean {mean:.2f})")
    for crit, floor in j.criterion_floors.items():
        s = scores[crit]["score"]
        if s < floor:
            reasons.append(f"{p}: {crit.replace('_', ' ')} {s:.1f} < floor {floor:.1f}")
    return JudgeAssessment(p, not reasons, overall, round(mean, 3), round(effective, 3), fg, reasons)


def judge_rejects(outcome: JudgeOutcome, cfg: Config) -> bool:
    """Would this single verdict already doom the clip? (Used for short-circuiting the panel.)"""
    a = assess_judge(outcome, cfg)
    if not a.ok:
        return True
    return bool(a.failed_gates) and cfg.judging.gate_policy == "any"


def panel_consensus(outcomes: list[JudgeOutcome], cfg: Config) -> tuple[bool, list[str], list[JudgeAssessment]]:
    expected = list(cfg.judging.judges)
    by_persona = {o.persona: o for o in outcomes}
    reasons: list[str] = []
    assessments: list[JudgeAssessment] = []
    for p in expected:
        o = by_persona.get(p) or JudgeOutcome(persona=p, error="missing")
        assessments.append(assess_judge(o, cfg))
    unknown = set(by_persona) - set(expected)
    if unknown:
        reasons.append(f"unexpected judges in panel: {sorted(unknown)}")

    # Gates first (they are the hard rule), then scores.
    gate_fails: dict[str, list[str]] = {}
    for a in assessments:
        for g in a.failed_gates:
            gate_fails.setdefault(g, []).append(a.persona)
    n = len(expected)
    for g in GATE_NAMES:
        who = gate_fails.get(g, [])
        fatal = bool(who) if cfg.judging.gate_policy == "any" else len(who) * 2 > n
        if fatal:
            o = by_persona.get(who[0])
            ev = (o.verdict or {}).get("gates", {}).get(g, {}).get("evidence", "") if o else ""
            reasons.append(f"gate '{GATE_LABELS[g]}' failed ({', '.join(who)}): {ev}".rstrip(": "))
    for a in assessments:
        if not a.unavailable:
            reasons.extend(a.reasons)
    skipped = {o.persona for o in outcomes if o.skipped}
    for a in assessments:
        # A judge skipped by the short-circuit only matters if nothing else rejected the clip.
        if a.unavailable and (a.persona not in skipped or not reasons):
            reasons.extend(a.reasons)
    return not reasons, reasons, assessments


def devils_verdict(arbiter: dict | None, error: str | None, cfg: Config) -> tuple[bool, list[str]]:
    """True if the clip SURVIVES the devil's advocate."""
    if not cfg.devils_advocate.enabled:
        return True, []
    if arbiter is None:
        return False, [f"devil's advocate/arbiter unavailable ({error or 'no ruling'}), so the clip is rejected"]
    conf = arbiter.get("survival_confidence")
    convincing = arbiter.get("argument_convincing")
    if not isinstance(convincing, bool) or not _finite(conf):
        return False, ["arbiter ruling malformed, so the clip is rejected"]
    reasons = []
    if convincing:
        reasons.append(f"devil's advocate convinced the arbiter: {arbiter.get('decisive_issue', '')}".rstrip(": "))
    if conf < cfg.devils_advocate.min_survival_confidence:
        reasons.append(f"arbiter survival confidence {conf:.1f} < {cfg.devils_advocate.min_survival_confidence:.1f}")
    return not reasons, reasons


def decide(candidate_id: str, det: list[GateResult], outcomes: list[JudgeOutcome] | None,
           arbiter: dict | None, devil: dict | None, cfg: Config, devil_error: str | None = None) -> Decision:
    det_json = [g.to_json() for g in det]
    det_fail = [g for g in det if not g.passed]
    if det_fail:
        return Decision(candidate_id, False, "deterministic_gates",
                        [f"gate '{g.name}' failed: {g.detail}" for g in det_fail], deterministic=det_json)
    outcomes = outcomes or []
    ok, reasons, assessments = panel_consensus(outcomes, cfg)
    effs = [a.effective for a in assessments if a.effective is not None]
    common = dict(deterministic=det_json, judges=[asdict(o) for o in outcomes],
                  assessments=[asdict(a) for a in assessments],
                  min_effective=min(effs) if len(effs) == len(cfg.judging.judges) else None,
                  mean_effective=round(sum(effs) / len(effs), 3) if effs else None,
                  errors=[f"{o.persona}: {o.error}" for o in outcomes if o.error])
    if not ok:
        # If the only problems are judges that never produced a verdict (API error, refusal),
        # this is an incomplete judgment, not a quality verdict. It is still not GREAT.
        real = [a for a in assessments if not a.unavailable and (a.reasons or a.failed_gates)]
        errored = any(o.error for o in outcomes) or len(outcomes) < len(cfg.judging.judges)
        stage = "error" if not real and errored else "panel"
        return Decision(candidate_id, False, stage, reasons, **common)
    survives, why = devils_verdict(arbiter, devil_error, cfg)
    if not survives:
        return Decision(candidate_id, False, "devils_advocate", why, devil=devil, arbiter=arbiter, **common)
    return Decision(candidate_id, True, "great", [], devil=devil, arbiter=arbiter, **common)


def rank_great(decisions: list[Decision]) -> list[Decision]:
    great = [d for d in decisions if d.great]
    return sorted(great, key=lambda d: (-(d.min_effective or 0), -(d.mean_effective or 0), d.candidate_id))


def near_miss_key(d: Decision) -> tuple:
    """Order rejected candidates by how close they came: later stage first, then score."""
    stage_rank = {"devils_advocate": 0, "panel": 1, "error": 2, "deterministic_gates": 3}.get(d.stage, 4)
    effs = [a.get("effective") for a in d.assessments if a.get("effective") is not None]
    weakest = min(effs) if effs else -1.0  # the harshest judge that actually ran
    return (stage_rank, -weakest, d.candidate_id)
