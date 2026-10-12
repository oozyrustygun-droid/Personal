"""Orchestrates the scan: ingest → transcribe → signals → candidates → judging → report → export."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .candidates import Candidate, candidates_stage
from .cache import Workspace
from .config import Config, load_config
from .cost import CostTracker
from .ingest import VideoMeta, ingest
from .keyframes import extract_keyframes
from .llm import LLM
from .signals import Signals, signals_stage
from .transcribe import Transcript, transcribe
from .ui import console, fmt_usd, stage


@dataclass
class Analysis:
    meta: VideoMeta
    ws: Workspace
    transcript: Transcript
    signals: Signals
    candidates: list[Candidate]
    frames: dict
    llm: LLM
    tracker: CostTracker
    candidates_fp: str


def make_llm(cfg: Config, ws: Workspace, tracker: CostTracker, judge_backend: str = "anthropic",
             scripted_judge: str | None = None, refresh: bool = False) -> LLM:
    if judge_backend == "scripted":
        from .scripted import load_script

        llm = LLM(cfg, ws.path("llm"), tracker, backend="scripted", script=load_script(scripted_judge))
    else:
        llm = LLM(cfg, ws.path("llm"), tracker)
    llm.refresh = refresh
    return llm


def analyze(source: str, cfg: Config, *, rerun: set[str] = frozenset(), judge_backend: str = "anthropic",
            scripted_judge: str | None = None, single_window: bool = False, quiet: bool = False) -> Analysis:
    """Stages 1-4 (+ keyframes). `single_window` treats the whole file as one candidate (calibration)."""
    out = console if not quiet else None
    with stage("Ingest") as st:
        meta, ws = ingest(source, cfg, console=out)
        st["note"] = f"{meta.title} · {meta.duration / 60:.1f} min · {meta.width}x{meta.height}"
    console.print(f"  [dim]workspace: {ws.root}[/dim]")
    with stage("Transcribe") as st:
        before = ws.stage_fingerprint("transcribe")
        transcript, t_fp = transcribe(ws, cfg, console=out, force="transcribe" in rerun)
        st["cached"] = before == t_fp and "transcribe" not in rerun
        st["note"] = f"{len(transcript.words)} words ({transcript.backend}/{transcript.model})"
    with stage("Signals") as st:
        before = ws.stage_fingerprint("signals")
        signals, s_fp = signals_stage(ws, transcript, t_fp, meta.duration, meta.path, cfg, console=out,
                                      force="signals" in rerun)
        st["cached"] = before == s_fp and "signals" not in rerun
        st["note"] = (f"{len(signals.scene_cuts)} scene cuts · audio events "
                      f"{'on' if signals.events_available else 'off'}")
    tracker = CostTracker(cfg)
    llm = make_llm(cfg, ws, tracker, judge_backend, scripted_judge, refresh="judging" in rerun)
    with stage("Candidates") as st:
        if single_window:
            from .calibrate import whole_clip_candidate

            cands, c_fp = whole_clip_candidate(transcript, signals, meta.duration, cfg, s_fp)
        else:
            before = ws.stage_fingerprint("candidates")
            cands, c_fp = candidates_stage(ws, transcript, signals, s_fp, meta.duration, cfg, llm, console=out,
                                           force="candidates" in rerun)
            st["cached"] = before == c_fp and "candidates" not in rerun
        st["note"] = f"{len(cands)} candidate window(s)"
    with stage("Keyframes"):
        frames = extract_keyframes(ws, meta.path, cands, signals, cfg, c_fp)
    return Analysis(meta, ws, transcript, signals, cands, frames, llm, tracker, c_fp)


def print_estimate(est: dict) -> None:
    if est["uncached"] == 0:
        console.print(f"  [green]all {est['eligible']} eligible candidate(s) already judged (cached): $0.00[/green]")
        return
    console.print(f"  {est['eligible']}/{est['candidates']} candidates passed the deterministic gates; "
                  f"{est['uncached']} need judging with {est['model']} (effort {est['effort']}).")
    console.print(f"  [bold]Cost estimate: ~{fmt_usd(est['expected_usd'])}[/bold] expected "
                  f"(worst case {fmt_usd(est['worst_case_usd'])} if every clip reaches the full panel "
                  f"and the devil's advocate).")


def run_scan_cli(args: argparse.Namespace) -> int:
    from .judging.runner import build_context, confirm_cost, run_judging
    from .report import print_report, write_reports

    cfg = load_config(args.config)
    max_clips = args.max_clips or cfg.general.max_clips
    a = analyze(args.source, cfg, rerun=set(args.rerun), judge_backend=args.judge_backend,
                scripted_judge=args.scripted_judge)
    ctx = build_context(cfg, a.llm, a.transcript, a.signals, a.frames, a.meta.title, a.meta.uploader,
                        a.meta.duration)

    def on_estimate(est: dict) -> bool:
        print_estimate(est)
        if args.dry_run:
            return False
        return confirm_cost(est, cfg, args.yes, console)

    with stage("Judging"):
        result = run_judging(a.candidates, ctx, max_clips=max_clips, console=console, on_estimate=on_estimate)
    if args.dry_run:
        console.print("[dim]--dry-run: stopped before judging.[/dim]")
        return 0
    paths = write_reports(a, result, cfg, explain_rejects=args.explain_rejects)
    exported = {}
    if result.great and not args.no_cut:
        from .export import export_clips

        with stage("Export"):
            exported = export_clips(a, result, cfg, paths["dir"], force="export" in set(args.rerun))
        paths = write_reports(a, result, cfg, explain_rejects=args.explain_rejects, exported=exported)
    print_report(a, result, cfg, paths, explain_rejects=args.explain_rejects, exported=exported)
    return 0 if result.status != "unavailable" else 3
