"""Runs the precision stage: deterministic gates → blind panel → devil's advocate + arbiter.

Each candidate is judged in isolation (never side by side), so scores cannot become relative.
Judges run one after another per candidate. Under the unanimity rule the first rejecting judge
settles the outcome, so with short_circuit on the remaining judges are skipped (same verdict,
less spend). Candidates are processed concurrently.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..candidates import Candidate
from ..config import Config, load_style_guide
from ..cost import estimate_tokens, image_tokens
from ..llm import LLM, LLMError
from ..signals import Signals
from ..transcribe import Transcript
from . import prompts
from .gates import GateResult, clip_measurements, deterministic_gates
from .panel import Decision, JudgeOutcome, decide, judge_rejects, near_miss_key, rank_great
from .schemas import ArbiterRuling, DevilsCase, Diagnosis, JudgeVerdict, Packaging

# Rough output-token budgets (thinking + JSON) per call, used only for the pre-run estimate.
OUTPUT_TOKENS = {"low": 1500, "medium": 2500, "high": 4000, "xhigh": 6000, "max": 9000}
# Assumed per-judge pass rate for the expected-cost estimate (short-circuit).
ASSUMED_PASS_RATE = 0.3


@dataclass
class JudgeContext:
    cfg: Config
    llm: LLM
    transcript: Transcript
    signals: Signals
    frames: dict[str, list]
    handbook: str
    video_text: str
    title: str = ""


@dataclass
class JudgingResult:
    decisions: list[Decision]
    great: list[Decision]
    packaging: dict[str, dict] = field(default_factory=dict)
    diagnosis: str = ""
    status: str = "complete"  # complete | incomplete | unavailable
    notes: list[str] = field(default_factory=list)


def build_context(cfg: Config, llm: LLM, transcript: Transcript, signals: Signals, frames, title: str,
                  uploader: str, duration: float) -> JudgeContext:
    return JudgeContext(cfg=cfg, llm=llm, transcript=transcript, signals=signals, frames=frames,
                        handbook=prompts.handbook(load_style_guide()),
                        video_text=prompts.video_block(title, uploader, duration, transcript, cfg), title=title)


def _meta(c: Candidate, ctx: JudgeContext, **extra) -> dict:
    """Plain-data description of the request for the scripted (offline) backend."""
    return {"candidate_id": c.id, "clip_text": " ".join(w.text for w in c.words(ctx.transcript)),
            "duration": c.duration, **extra}


def _judge_once(c: Candidate, persona: str, packet: list[dict], ctx: JudgeContext) -> JudgeOutcome:
    j = ctx.cfg.judging
    system = prompts.system_blocks(ctx.handbook, ctx.video_text, persona, j.cache_ttl)
    try:
        res = ctx.llm.call(purpose="judge", model=j.model, system=system,
                           messages=[{"role": "user", "content": packet}], output=JudgeVerdict, effort=j.effort,
                           max_tokens=j.max_output_tokens, meta=_meta(c, ctx, persona=persona))
        return JudgeOutcome(persona=persona, verdict=res.data, served_model=res.served_model)
    except LLMError as e:
        return JudgeOutcome(persona=persona, error=str(e))


def _panel_summary(outcomes: list[JudgeOutcome]) -> str:
    parts = []
    for i, o in enumerate(outcomes, 1):
        v = o.verdict or {}
        sc = v.get("scores", {})
        parts.append(f"Judge {i}: overall {v.get('overall')}; hook {sc.get('hook_strength', {}).get('score')}, "
                     f"payoff {sc.get('payoff', {}).get('score')}. Why: {v.get('overall_justification', '')} "
                     f"Self-identified weakness: {v.get('biggest_weakness', '')}")
    return "\n".join(parts)


def _devil_and_arbiter(c: Candidate, packet: list[dict], outcomes: list[JudgeOutcome], ctx: JudgeContext):
    j = ctx.cfg.judging
    panel_case = _panel_summary(outcomes)
    try:
        devil = ctx.llm.call(
            purpose="devils_advocate", model=j.model,
            system=prompts.system_blocks(ctx.handbook, ctx.video_text, "devils_advocate", j.cache_ttl),
            messages=[{"role": "user", "content": packet + [
                {"type": "text", "text": f"THE PANEL'S CASE FOR THIS CLIP:\n{panel_case}\n\nMake the case it flops."}]}],
            output=DevilsCase, effort=j.effort, max_tokens=j.max_output_tokens,
            meta=_meta(c, ctx, role="devils_advocate")).data
    except LLMError as e:
        return None, None, f"devil's advocate failed: {e}"
    try:
        ruling = ctx.llm.call(
            purpose="arbiter", model=j.model,
            system=prompts.system_blocks(ctx.handbook, ctx.video_text, "arbiter", j.cache_ttl),
            messages=[{"role": "user", "content": packet + [{"type": "text", "text": (
                f"THE PANEL'S CASE FOR THIS CLIP:\n{panel_case}\n\n"
                f"THE DEVIL'S ADVOCATE'S CASE AGAINST IT:\n{json.dumps(devil, indent=1)}\n\nRule.")}]}],
            output=ArbiterRuling, effort=j.effort, max_tokens=j.max_output_tokens,
            meta=_meta(c, ctx, role="arbiter", devil=devil)).data
    except LLMError as e:
        return devil, None, f"arbiter failed: {e}"
    return devil, ruling, None


def collect_panel(cfg: Config, judge_fn, *, full_panel: bool = False) -> list[JudgeOutcome]:
    """Run the judges in order. Once one verdict alone dooms the clip, skip the rest (short-circuit).
    The final decision is identical either way, because the rule is unanimous (tested)."""
    outcomes: list[JudgeOutcome] = []
    doomed = False
    for persona in cfg.judging.judges:
        if doomed and not full_panel and cfg.judging.short_circuit:
            outcomes.append(JudgeOutcome(persona=persona, skipped=True))
            continue
        o = judge_fn(persona)
        outcomes.append(o)
        doomed = doomed or judge_rejects(o, cfg)
    return outcomes


def judge_candidate(c: Candidate, det: list[GateResult], measurements: dict, ctx: JudgeContext,
                    *, full_panel: bool = False) -> Decision:
    cfg = ctx.cfg
    if any(not g.passed for g in det):
        return decide(c.id, det, None, None, None, cfg)
    packet = prompts.clip_packet(c, ctx.transcript, ctx.signals, measurements, ctx.frames.get(c.id, []), cfg)
    outcomes = collect_panel(cfg, lambda persona: _judge_once(c, persona, packet, ctx), full_panel=full_panel)
    pre = decide(c.id, det, outcomes, None, None, cfg)
    panel_passed = pre.stage in ("devils_advocate", "great")  # decide() got no ruling yet
    # Only clips the whole panel passed face the devil's advocate. In calibration (full_panel) every
    # fully judged clip does too, so thresholds can be swept offline from cached results.
    run_devil = cfg.devils_advocate.enabled and (panel_passed or (full_panel and pre.stage == "panel"))
    if not run_devil:
        return pre
    devil, ruling, err = _devil_and_arbiter(c, packet, outcomes, ctx)
    return decide(c.id, det, outcomes, ruling, devil, cfg, devil_error=err)


def estimate_cost(cands: list[Candidate], dets: dict[str, list[GateResult]], ctx: JudgeContext) -> dict:
    cfg = ctx.cfg
    j = cfg.judging
    price = cfg.price(j.model)
    eligible = [c for c in cands if all(g.passed for g in dets[c.id])]
    # Requests already in the on-disk cache cost nothing.
    uncached = [c for c in eligible if not _first_request_cached(c, j.judges[0], ctx)]
    prefix_tokens = estimate_tokens(ctx.handbook) + estimate_tokens(ctx.video_text)
    role_tokens = 450
    per_clip_text = 1800
    img = sum(image_tokens(j.keyframe_width, int(j.keyframe_width * 9 / 16)) for _ in range(j.keyframes_per_candidate))
    out_tokens = OUTPUT_TOKENS[j.effort]
    write_rate = price.cache_write_1h if j.cache_ttl == "1h" else price.cache_write_5m

    def call_cost(n_calls: int) -> float:
        if n_calls == 0:
            return 0.0
        fresh = n_calls * ((role_tokens + per_clip_text + img) * price.input + out_tokens * price.output)
        cache = prefix_tokens * write_rate + (n_calls - 1) * prefix_tokens * price.cache_read
        return (fresh + cache) / 1e6

    n = len(uncached)
    k = len(j.judges)
    expected_judges = sum(ASSUMED_PASS_RATE ** i for i in range(k)) if j.short_circuit else k
    expected_finalists = n * ASSUMED_PASS_RATE ** k
    expected = call_cost(round(n * expected_judges + 2 * expected_finalists * cfg.devils_advocate.enabled))
    worst = call_cost(n * k + 2 * n * cfg.devils_advocate.enabled)
    return {"candidates": len(cands), "eligible": len(eligible), "uncached": n, "expected_usd": round(expected, 2),
            "worst_case_usd": round(worst, 2), "prefix_tokens": prefix_tokens, "model": j.model,
            "effort": j.effort}


def _first_request_cached(c: Candidate, persona: str, ctx: JudgeContext) -> bool:
    j = ctx.cfg.judging
    measurements = clip_measurements(c, ctx.transcript, ctx.signals)
    packet = prompts.clip_packet(c, ctx.transcript, ctx.signals, measurements, ctx.frames.get(c.id, []), ctx.cfg)
    system = prompts.system_blocks(ctx.handbook, ctx.video_text, persona, j.cache_ttl)
    from ..llm import strict_schema

    rh = ctx.llm.request_hash(model=j.model, system=system, messages=[{"role": "user", "content": packet}],
                              schema=strict_schema(JudgeVerdict), effort=j.effort)
    if ctx.llm.backend == "scripted":
        rh = ctx.llm.scripted_hash("judge", _meta(c, ctx, persona=persona), rh)
    return ctx.llm.cached_result("judge", rh) is not None


def confirm_cost(est: dict, cfg: Config, assume_yes: bool, console) -> bool:
    if est["uncached"] == 0 or assume_yes or est["expected_usd"] <= cfg.judging.confirm_above_usd:
        return True
    if not sys.stdin.isatty():
        console.print(f"[yellow]Estimated ${est['expected_usd']:.2f} exceeds confirm_above_usd "
                      f"(${cfg.judging.confirm_above_usd:.2f}); re-run with --yes to proceed.[/yellow]")
        return False
    ans = console.input(f"Proceed with judging (est. ${est['expected_usd']:.2f}, worst case "
                        f"${est['worst_case_usd']:.2f})? [y/N] ")
    return ans.strip().lower() in ("y", "yes")


def package(c: Candidate, decision: Decision, ctx: JudgeContext) -> dict:
    j = ctx.cfg.judging
    measurements = clip_measurements(c, ctx.transcript, ctx.signals)
    packet = prompts.clip_packet(c, ctx.transcript, ctx.signals, measurements, ctx.frames.get(c.id, []), ctx.cfg)
    hooks = [jd.get("verdict", {}) or {} for jd in decision.judges]
    notes = "\n".join(f"- {h.get('hook_line', '')} | {h.get('overall_justification', '')}" for h in hooks if h)
    try:
        return ctx.llm.call(
            purpose="packaging", model=j.model,
            system=prompts.system_blocks(ctx.handbook, ctx.video_text, "packaging", j.cache_ttl),
            messages=[{"role": "user", "content": packet + [
                {"type": "text", "text": f"PANEL NOTES:\n{notes}\n\nWrite the packaging."}]}],
            output=Packaging, effort="medium", max_tokens=8000, meta=_meta(c, ctx, role="packaging")).data
    except LLMError as e:
        first = (hooks[0] if hooks else {}) or {}
        return {"hook_line": first.get("hook_line", ""), "why_viral": first.get("overall_justification", ""),
                "caption_title": " ".join(first.get("hook_line", "").split()[:8]), "posting_hook_text": "",
                "hashtags": ["#joebartolozzi"], "error": str(e)}


def diagnose(decisions: list[Decision], ctx: JudgeContext) -> str:
    j = ctx.cfg.judging
    stages: dict[str, int] = {}
    for d in decisions:
        stages[d.stage] = stages.get(d.stage, 0) + 1
    top = sorted(decisions, key=near_miss_key)[:8]
    summary = "\n".join(f"- {d.candidate_id} ({d.stage}): {d.reasons[0] if d.reasons else ''}" for d in top)
    text = (f"Video: {ctx.title}\nCandidates judged: {len(decisions)}; rejected at stage: {stages}\n"
            f"Closest candidates and why they died:\n{summary}")
    try:
        return ctx.llm.call(
            purpose="diagnosis", model=j.model,
            system=prompts.system_blocks(ctx.handbook, ctx.video_text, "diagnosis", j.cache_ttl),
            messages=[{"role": "user", "content": text}], output=Diagnosis, effort="low", max_tokens=4000,
            meta={"role": "diagnosis", "stages": stages}).data["one_line_read"]
    except LLMError:
        common = max(stages, key=stages.get) if stages else "n/a"
        return f"Most candidates died at the {common.replace('_', ' ')} stage; no moment cleared every gate and the 8.5 bar."


def run_judging(cands: list[Candidate], ctx: JudgeContext, *, max_clips: int, console=None,
                full_panel: bool = False, on_estimate=None) -> JudgingResult:
    cfg = ctx.cfg
    dets = {c.id: deterministic_gates(c, ctx.transcript, ctx.signals, cfg) for c in cands}
    meas = {c.id: clip_measurements(c, ctx.transcript, ctx.signals) for c in cands}
    est = estimate_cost(cands, dets, ctx)
    if on_estimate is not None and not on_estimate(est):
        return JudgingResult(decisions=[], great=[], status="unavailable", notes=["judging cancelled at cost prompt"])

    order = sorted(cands, key=lambda c: -c.score)
    decisions: dict[str, Decision] = {}
    notes: list[str] = []

    def work(c: Candidate) -> Decision:
        d = judge_candidate(c, dets[c.id], meas[c.id], ctx, full_panel=full_panel)
        if console:
            mark = {"great": "[bold green]GREAT[/bold green]", "error": "[yellow]error[/yellow]"}.get(
                d.stage, "[red]rejected[/red]")
            console.print(f"  {c.id} {mark} [dim]{d.reasons[0] if d.reasons else ''}[/dim]")
        return d

    eligible = [c for c in order if all(g.passed for g in dets[c.id])]
    for c in order:
        if c not in eligible:
            decisions[c.id] = decide(c.id, dets[c.id], None, None, None, cfg)
    if eligible:
        # Judge one clip first so the shared prompt prefix is written to the cache once, not N times.
        first = work(eligible[0])
        decisions[first.candidate_id] = first
        if any("auth" in e for e in first.errors):
            notes.append("No Anthropic credentials: judging could not run. Set ANTHROPIC_API_KEY.")
            for c in eligible[1:]:
                decisions[c.id] = Decision(c.id, False, "error", ["judging unavailable (no credentials)"],
                                           deterministic=[g.to_json() for g in dets[c.id]])
        else:
            with ThreadPoolExecutor(max_workers=cfg.judging.max_concurrency) as ex:
                for d in ex.map(work, eligible[1:]):
                    decisions[d.candidate_id] = d

    ordered = [decisions[c.id] for c in cands]
    great = rank_great(ordered)[:max_clips]
    status = "complete"
    if any(d.stage == "error" for d in ordered):
        status = "unavailable" if all(d.stage in ("error", "deterministic_gates") for d in ordered) and \
            any("credentials" in r for d in ordered for r in d.reasons) else "incomplete"
    result = JudgingResult(decisions=ordered, great=great, status=status, notes=notes)
    by_id = {c.id: c for c in cands}
    for d in great:
        result.packaging[d.candidate_id] = package(by_id[d.candidate_id], d, ctx)
    if not great and status != "unavailable" and ordered:
        result.diagnosis = diagnose(ordered, ctx)
    return result
