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
from ..ui import emit
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
    duration: float
    title: str = ""  # for reports only. Never sent to the judges.


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
                        handbook=prompts.handbook(load_style_guide(), cfg.judging.text_hook), duration=duration,
                        title=title)


def _meta(c: Candidate, ctx: JudgeContext, **extra) -> dict:
    """Plain-data description of the request for the scripted (offline) backend."""
    return {"candidate_id": c.id, "clip_text": " ".join(w.text for w in c.words(ctx.transcript)),
            "duration": c.duration, **extra}


def _judge_once(c: Candidate, persona: str, packet: list[dict], ctx: JudgeContext,
                fixed_hook: str | None = None) -> JudgeOutcome:
    """fixed_hook (text-hook mode): the on-screen hook an earlier judge wrote; this judge judges with it."""
    j = ctx.cfg.judging
    system = prompts.system_blocks(ctx.handbook, persona, j.cache_ttl)
    content = packet + ([prompts.hook_block(fixed_hook, for_judge=True)] if fixed_hook is not None else [])
    try:
        res = ctx.llm.call(purpose="judge", model=j.model, system=system,
                           messages=[{"role": "user", "content": content}], output=JudgeVerdict, effort=j.effort,
                           max_tokens=j.max_output_tokens, meta=_meta(c, ctx, persona=persona))
        verdict = dict(res.data)
        if fixed_hook is not None:
            verdict["text_hook"] = fixed_hook  # the hook it was told to judge with is the one on record
        return JudgeOutcome(persona=persona, verdict=verdict, served_model=res.served_model,
                            fallback=res.fallback_used)
    except LLMError as e:
        return JudgeOutcome(persona=persona, error=str(e))


def posted_hook(outcomes_or_judges) -> str | None:
    """The single on-screen text hook a clip is judged and posted with: the first judge's (text-hook mode)."""
    for o in outcomes_or_judges:
        v = o.verdict if isinstance(o, JudgeOutcome) else (o or {}).get("verdict")
        if v:
            return " ".join(str(v.get("text_hook", "")).split())
    return None


def _panel_summary(outcomes: list[JudgeOutcome]) -> str:
    """The panel's case in words only. Scores are withheld so they don't anchor the devil or the arbiter."""
    parts = []
    for i, o in enumerate(outcomes, 1):
        v = o.verdict or {}
        parts.append(f"Judge {i}: {v.get('overall_justification', '')} "
                     f"(Their own biggest worry: {v.get('biggest_weakness', '')})")
    return "\n".join(parts)


def _devil_and_arbiter(c: Candidate, packet: list[dict], outcomes: list[JudgeOutcome], ctx: JudgeContext):
    """Returns (devil, ruling, error). Neither call may be answered by a fallback model."""
    j = ctx.cfg.judging
    panel_case = _panel_summary(outcomes)
    stream = prompts.transcript_block(ctx.duration, ctx.transcript, ctx.cfg, j.cache_ttl)
    hook = posted_hook(outcomes) if ctx.cfg.judging.text_hook else None
    if hook is not None:
        packet = packet + [prompts.hook_block(hook, for_judge=False)]
    try:
        res = ctx.llm.call(
            purpose="devils_advocate", model=j.model,
            system=prompts.system_blocks(ctx.handbook, "devils_advocate", j.cache_ttl),
            messages=[{"role": "user", "content": [stream] + packet + [
                {"type": "text", "text": f"THE PANEL'S CASE FOR THIS CLIP:\n<panel_reasons>\n"
                                         f"{prompts.data_text(panel_case)}\n</panel_reasons>\n\nMake the case it flops."}]}],
            output=DevilsCase, effort=j.effort, max_tokens=j.max_output_tokens,
            meta=_meta(c, ctx, role="devils_advocate"))
        if res.fallback_used:
            return None, None, f"devil's advocate answered by {res.served_model}, not {j.model}"
        devil = res.data
    except LLMError as e:
        return None, None, f"devil's advocate failed: {e}"
    try:
        res = ctx.llm.call(
            purpose="arbiter", model=j.model,
            system=prompts.system_blocks(ctx.handbook, "arbiter", j.cache_ttl),
            messages=[{"role": "user", "content": packet + [{"type": "text", "text": (
                f"THE PANEL'S REASONS FOR THIS CLIP:\n<panel_reasons>\n{prompts.data_text(panel_case)}\n</panel_reasons>\n\n"
                f"THE DEVIL'S ADVOCATE'S CASE AGAINST IT:\n<devils_case>\n"
                f"{prompts.data_text(json.dumps(devil, indent=1, ensure_ascii=False))}\n</devils_case>\n\n"
                f"{prompts.excerpts_text(ctx.transcript, prompts.cited_times(json.dumps(devil)), ctx.cfg)}\n\n"
                f"Rule.")}]}],
            output=ArbiterRuling, effort=j.effort, max_tokens=j.max_output_tokens,
            meta=_meta(c, ctx, role="arbiter", devil=devil))
        if res.fallback_used:
            return devil, None, f"arbiter answered by {res.served_model}, not {j.model}"
        ruling = res.data
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
    done: list[JudgeOutcome] = []

    def judge(persona: str) -> JudgeOutcome:
        # Text-hook mode: the first judge writes the clip's on-screen hook; every later judge, the devil's
        # advocate, the arbiter and the export use exactly that hook, so the posted hook is the judged one.
        fixed = posted_hook(done) if cfg.judging.text_hook else None
        o = _judge_once(c, persona, packet, ctx, fixed_hook=fixed)
        done.append(o)
        return o

    outcomes = collect_panel(cfg, judge, full_panel=full_panel)
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
    uncached = [c for c in eligible if ctx.llm.refresh or not _first_request_cached(c, j.judges[0], ctx)]
    prefix_tokens = estimate_tokens(ctx.handbook)
    stream_tokens = estimate_tokens(" ".join(w.text for w in ctx.transcript.words)) * 1.3  # + timestamps
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

    def devil_cost(n_finalists: float) -> float:
        # The devil's advocate also reads the full stream transcript (cached after the first call).
        if n_finalists <= 0 or not cfg.devils_advocate.enabled:
            return 0.0
        return (stream_tokens * write_rate + (n_finalists - 1) * stream_tokens * price.cache_read) / 1e6

    n = len(uncached)
    k = len(j.judges)
    expected_judges = sum(ASSUMED_PASS_RATE ** i for i in range(k)) if j.short_circuit else k
    expected_finalists = n * ASSUMED_PASS_RATE ** k
    expected_calls = round(n * expected_judges + 2 * expected_finalists * cfg.devils_advocate.enabled)
    worst_calls = n * k + 2 * n * cfg.devils_advocate.enabled
    expected = call_cost(expected_calls) + devil_cost(max(1.0, expected_finalists) if n else 0)
    worst = call_cost(worst_calls) + devil_cost(n)
    # On the subscription the dollar figures are API-equivalent sizes, not charges.
    return {"candidates": len(cands), "eligible": len(eligible), "uncached": n, "expected_usd": round(expected, 2),
            "worst_case_usd": round(worst, 2), "expected_calls": expected_calls, "worst_case_calls": worst_calls,
            "prefix_tokens": prefix_tokens, "model": j.model, "effort": j.effort, "billing": ctx.llm.billing}


def _first_request_cached(c: Candidate, persona: str, ctx: JudgeContext) -> bool:
    j = ctx.cfg.judging
    measurements = clip_measurements(c, ctx.transcript, ctx.signals, ctx.cfg.gates.music_prob)
    packet = prompts.clip_packet(c, ctx.transcript, ctx.signals, measurements, ctx.frames.get(c.id, []), ctx.cfg)
    system = prompts.system_blocks(ctx.handbook, persona, j.cache_ttl)
    from ..llm import strict_schema

    rh = ctx.llm.request_hash(model=j.model, system=system, messages=[{"role": "user", "content": packet}],
                              schema=strict_schema(JudgeVerdict), effort=j.effort)
    if ctx.llm.backend == "scripted":
        rh = ctx.llm.scripted_hash("judge", _meta(c, ctx, persona=persona), rh)
    return ctx.llm.cached_result("judge", rh) is not None


def needs_approval(est: dict, cfg: Config) -> bool:
    """Only money needs approval: API spend above the configured threshold. Subscription runs cost no
    API credits (and stop on their own if Claude Code starts using paid extra usage). The scripted test
    judge follows the API rule so the approval flow stays testable offline."""
    return (est["uncached"] > 0 and est.get("billing", "api") != "subscription"
            and est["expected_usd"] > cfg.judging.confirm_above_usd)


def confirm_cost(est: dict, cfg: Config, assume_yes: bool, console) -> bool:
    if assume_yes or not needs_approval(est, cfg):
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
    measurements = clip_measurements(c, ctx.transcript, ctx.signals, ctx.cfg.gates.music_prob)
    packet = prompts.clip_packet(c, ctx.transcript, ctx.signals, measurements, ctx.frames.get(c.id, []), ctx.cfg)
    hooks = [jd.get("verdict", {}) or {} for jd in decision.judges]
    notes = "\n".join(f"- opening: {h.get('hook_line', '')} | {h.get('overall_justification', '')}"
                      for h in hooks if h)
    # Text-hook mode: the on-screen hook was fixed during judging and every judge, the devil's advocate and
    # the arbiter judged the clip with it. It is posted verbatim, never rewritten here.
    fixed = posted_hook(decision.judges) if j.text_hook else None
    extra = []
    if fixed is not None:
        extra = [prompts.hook_block(fixed, for_judge=False),
                 {"type": "text", "text": "That on-screen text hook is final: write the other fields to match it."}]
    try:
        data = ctx.llm.call(
            purpose="packaging", model=j.model,
            system=prompts.system_blocks(ctx.handbook, "packaging", j.cache_ttl),
            messages=[{"role": "user", "content": packet + extra + [
                {"type": "text", "text": f"PANEL NOTES:\n{notes}\n\nWrite the packaging."}]}],
            output=Packaging, effort="medium", max_tokens=8000, meta=_meta(c, ctx, role="packaging"),
            allow_fallback=True).data
    except LLMError as e:
        first = (hooks[0] if hooks else {}) or {}
        data = {"hook_line": first.get("hook_line", ""), "why_viral": first.get("overall_justification", ""),
                "caption_title": " ".join(first.get("hook_line", "").split()[:8]), "posting_hook_text": "",
                "hashtags": ["#joebartolozzi"], "error": str(e)}
    if fixed is not None:
        data = {**data, "caption_title": fixed, "text_hook": fixed}
    return data


def diagnose(decisions: list[Decision], ctx: JudgeContext) -> str:
    j = ctx.cfg.judging
    stages: dict[str, int] = {}
    for d in decisions:
        stages[d.stage] = stages.get(d.stage, 0) + 1
    top = sorted(decisions, key=near_miss_key)[:8]
    summary = "\n".join(f"- {d.candidate_id} ({d.stage}): {d.reasons[0] if d.reasons else ''}" for d in top)
    text = (f"Candidates judged: {len(decisions)}; rejected at stage: {stages}\n"
            f"Closest candidates and why they died:\n{summary}")
    try:
        return ctx.llm.call(
            purpose="diagnosis", model=j.model,
            system=prompts.system_blocks(ctx.handbook, "diagnosis", j.cache_ttl),
            messages=[{"role": "user", "content": text}], output=Diagnosis, effort="low", max_tokens=4000,
            meta={"role": "diagnosis", "stages": stages}, allow_fallback=True).data["one_line_read"]
    except LLMError:
        common = max(stages, key=stages.get) if stages else "n/a"
        return f"Most candidates died at the {common.replace('_', ' ')} stage; no moment cleared every gate and the 8.5 bar."


def same_moment(a: Candidate, b: Candidate, cfg: Config) -> bool:
    """Two candidates are one moment if their peaks are close, or they mostly overlap AND each contains
    the other's peak. (A reaction-first window running through its aftermath can overlap the next
    moment's build-up a lot without being the same moment.)"""
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    return abs(a.peak - b.peak) <= cfg.candidates.peak_min_separation_s or (
        inter > 0.5 * min(a.end - a.start, b.end - b.start)
        and a.start <= b.peak <= a.end and b.start <= a.peak <= b.end)


def one_per_moment(ranked: list[Decision], by_id: dict[str, Candidate], cfg: Config) -> list[Decision]:
    """Each moment is judged in up to two versions (opening on the reaction, opening on the build-up).
    If both are GREAT, keep the higher-ranked one; the other is marked as a duplicate, never exported."""
    kept: list[Decision] = []
    for d in ranked:
        twin = next((k for k in kept if same_moment(by_id[d.candidate_id], by_id[k.candidate_id], cfg)), None)
        if twin is None:
            kept.append(d)
            continue
        d.great, d.stage = False, "duplicate"
        d.reasons = [f"GREAT, but the same moment as {twin.candidate_id}, which ranked higher "
                     "(only one version of a moment is kept)"]
    return kept


def run_judging(cands: list[Candidate], ctx: JudgeContext, *, max_clips: int, console=None,
                full_panel: bool = False, on_estimate=None, recall_complete: bool = True) -> JudgingResult:
    """recall_complete=False: candidate generation was cut short (the transcript proposer couldn't read
    part of the video), so "no GREAT clips" would not be a verdict about the whole video."""
    cfg = ctx.cfg
    dets = {c.id: deterministic_gates(c, ctx.transcript, ctx.signals, cfg) for c in cands}
    meas = {c.id: clip_measurements(c, ctx.transcript, ctx.signals, ctx.cfg.gates.music_prob) for c in cands}
    est = estimate_cost(cands, dets, ctx)
    if on_estimate is not None and not on_estimate(est):
        return JudgingResult(decisions=[], great=[], status="unavailable", notes=["judging cancelled at cost prompt"])

    order = sorted(cands, key=lambda c: -c.score)
    decisions: dict[str, Decision] = {}
    notes: list[str] = []

    def work(c: Candidate) -> Decision:
        d = judge_candidate(c, dets[c.id], meas[c.id], ctx, full_panel=full_panel)
        emit("candidate", id=c.id, stage=d.stage, great=d.great, reason=d.reasons[0] if d.reasons else "")
        if console:
            mark = {"great": "[bold green]GREAT[/bold green]", "error": "[yellow]error[/yellow]"}.get(
                d.stage, "[red]rejected[/red]")
            console.print(f"  {c.id} {mark} [dim]{d.reasons[0] if d.reasons else ''}[/dim]")
        return d

    eligible = [c for c in order if all(g.passed for g in dets[c.id])]
    emit("judging", total=len(cands), eligible=len(eligible))
    for c in order:
        if c not in eligible:
            decisions[c.id] = decide(c.id, dets[c.id], None, None, None, cfg)
    fatal_before = ctx.llm.fatal  # e.g. the proposer already found Claude Code logged out
    if eligible:
        # Judge one clip first so the shared prompt prefix is written to the cache once, not N times.
        first = work(eligible[0])
        decisions[first.candidate_id] = first
        # After a fatal error (no login, API billing refused, plan limit) live requests fail at once,
        # but candidates judged on an earlier run still come back from the cache.
        ex = ThreadPoolExecutor(max_workers=cfg.judging.max_concurrency)
        try:
            for d in ex.map(work, eligible[1:]):
                decisions[d.candidate_id] = d
        except BaseException:
            # Ctrl-C or a crash: don't let queued candidates keep starting model calls on the way out.
            ex.shutdown(wait=False, cancel_futures=True)
            from ..claude_code import stop_all_runs

            stop_all_runs()
            raise
        finally:
            ex.shutdown(wait=True)
    fatal = ctx.llm.fatal or fatal_before
    if fatal is not None:
        notes.append(f"Judging stopped: {str(fatal).split(': ', 1)[-1]}")

    ordered = [decisions[c.id] for c in cands]
    great = one_per_moment(rank_great(ordered), {c.id: c for c in cands}, cfg)[:max_clips]
    for d in ordered:
        if d.stage == "duplicate" and console:
            console.print(f"  {d.candidate_id} [dim]{d.reasons[0]}[/dim]")
    emit("judging_done", great=len(great), duplicates=[d.candidate_id for d in ordered if d.stage == "duplicate"])
    status = "complete"
    if any(d.stage == "error" for d in ordered):
        judged_any = any(d.stage not in ("error", "deterministic_gates") for d in ordered)
        status = "unavailable" if fatal is not None and not judged_any else "incomplete"
    elif fatal is not None:
        status = "incomplete"  # e.g. the transcript proposer was cut off: recall is not what it should be
    if not recall_complete and status == "complete" and not great:
        status = "incomplete"
        notes.append("The transcript proposer couldn't read part of the video, so some moments were never "
                     "considered. Re-run to finish (everything already done is cached).")
    if ordered and not ctx.signals.events_available and cfg.gates.require_music_check and status == "complete":
        # Every candidate failed the music gate because music couldn't be measured: that's not a
        # verdict about the video.
        status = "incomplete"
        notes.append("Music could not be measured (audio-event model unavailable), so every candidate failed "
                     "the music gate. Run `clip-scanner doctor`, fix it, and re-run.")
    result = JudgingResult(decisions=ordered, great=great, status=status, notes=notes)
    by_id = {c.id: c for c in cands}
    for d in great:
        result.packaging[d.candidate_id] = package(by_id[d.candidate_id], d, ctx)
    if not great and status == "complete" and ordered and ctx.llm.fatal is None:
        result.diagnosis = diagnose(ordered, ctx)
    if ctx.llm.fatal is not None and fatal is None:
        # Stopped during packaging or the diagnosis: the verdicts stand, but say why captions are basic.
        result.notes.append(f"Stopped after judging: {str(ctx.llm.fatal).split(': ', 1)[-1]} Clips judged GREAT "
                            "are still GREAT; their caption/hook text may be a plain fallback.")
    return result
