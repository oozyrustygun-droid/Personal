"""Orchestrates the scan: ingest → transcribe → signals → candidates → judging → report → export."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .candidates import Candidate, candidates_stage
from .cli_common import BACKEND_LABEL, resolve_backend
from .cache import Workspace, read_json
from .config import Config, load_config
from .cost import CostTracker
from .ingest import VideoMeta, ingest
from .keyframes import extract_keyframes
from .llm import LLM
from .signals import Signals, signals_stage
from .transcribe import Transcript, transcribe
from .ui import console, emit, fmt_usd, stage


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
    recall_complete: bool = True  # False if the transcript proposer skipped part of the video


def make_llm(cfg: Config, ws: Workspace, tracker: CostTracker, judge_backend: str | None = None,
             scripted_judge: str | None = None, refresh: bool = False) -> LLM:
    backend = resolve_backend(cfg, judge_backend)
    if backend == "scripted":
        from .scripted import load_script

        llm = LLM(cfg, ws.path("llm"), tracker, backend="scripted", script=load_script(scripted_judge))
    else:
        llm = LLM(cfg, ws.path("llm"), tracker, backend=backend)
    tracker.billing = llm.billing
    llm.refresh = refresh
    return llm


def front_stages(source: str, cfg: Config, *, rerun: set[str] = frozenset(), quiet: bool = False,
                 on_ingested=None) -> tuple[VideoMeta, Workspace, Transcript, Signals, str]:
    """Ingest, transcribe, signals (shared by both modes). Returns (meta, ws, transcript, signals, signals_fp).
    `on_ingested(meta)` runs as soon as the video is identified, before any slow stage."""
    out = console if not quiet else None
    with stage("Ingest") as st:
        meta, ws = ingest(source, cfg, console=out)
        st["note"] = f"{meta.title} · {meta.duration / 60:.1f} min · {meta.width}x{meta.height}"
    console.print(f"  [dim]workspace: {ws.root}[/dim]")
    if on_ingested is not None:
        on_ingested(meta)
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
    return meta, ws, transcript, signals, s_fp


def analyze(source: str, cfg: Config, *, rerun: set[str] = frozenset(), judge_backend: str | None = None,
            scripted_judge: str | None = None, single_window: bool = False, quiet: bool = False,
            on_ingested=None) -> Analysis:
    """Stages 1-4 (+ keyframes). `single_window` treats the whole file as one candidate (calibration)."""
    out = console if not quiet else None
    meta, ws, transcript, signals, s_fp = front_stages(source, cfg, rerun=rerun, quiet=quiet,
                                                       on_ingested=on_ingested)
    tracker = CostTracker(cfg)
    llm = make_llm(cfg, ws, tracker, judge_backend, scripted_judge, refresh="judging" in rerun)
    recall_complete = True
    with stage("Candidates") as st:
        if single_window:
            from .calibrate import whole_clip_candidate

            cands, c_fp = whole_clip_candidate(transcript, signals, meta.duration, cfg, s_fp)
        else:
            before = ws.stage_fingerprint("candidates")
            cands, c_fp = candidates_stage(ws, transcript, signals, s_fp, meta.duration, cfg, llm, console=out,
                                           force="candidates" in rerun)
            st["cached"] = before == c_fp and "candidates" not in rerun
            recall_complete = not read_json(ws.path("candidates.json")).get("proposer_incomplete", False)
        st["note"] = f"{len(cands)} candidate window(s)" + ("" if recall_complete else " (proposer incomplete)")
    with stage("Keyframes"):
        frames = extract_keyframes(ws, meta.path, cands, signals, cfg, c_fp)
    return Analysis(meta, ws, transcript, signals, cands, frames, llm, tracker, c_fp, recall_complete)


def print_estimate(est: dict) -> None:
    if est["uncached"] == 0:
        console.print(f"  [green]all {est['eligible']} eligible candidate(s) already judged (cached): nothing to "
                      f"spend[/green]")
        return
    via = " through Claude Code" if est.get("billing") == "subscription" else ""
    console.print(f"  {est['eligible']}/{est['candidates']} candidates passed the deterministic gates; "
                  f"{est['uncached']} need judging with {est['model']} (effort {est['effort']}){via}.")
    if est.get("billing") == "subscription":
        console.print(f"  [bold]Runs on your Claude plan: no API charges.[/bold] About {est['expected_calls']} "
                      f"model call(s) expected, {est['worst_case_calls']} at most; they count toward your plan's "
                      f"usage limits. (At API prices this would be ~{fmt_usd(est['expected_usd'])}, shown only "
                      f"as a size guide.)")
        return
    console.print(f"  [bold]Cost estimate: ~{fmt_usd(est['expected_usd'])}[/bold] expected "
                  f"(worst case {fmt_usd(est['worst_case_usd'])} if every clip reaches the full panel "
                  f"and the devil's advocate).")


def run_scan_cli(args: argparse.Namespace) -> int:
    from .judging.runner import confirm_cost

    return run_scan(args, lambda est, cfg: confirm_cost(est, cfg, args.yes, console))["code"]


def run_scan(args: argparse.Namespace, confirm) -> dict:
    """One full scan. `confirm(estimate, cfg) -> bool` decides whether to spend on judging (the terminal
    prompt in the CLI, an Approve button in the web UI). Returns {"code", "out_dir", "status"}."""
    from .export import (expected_files, export_clips, prune_stale_clips, restore_all, restore_exports,
                         stash_exports)
    from .judging.runner import build_context, run_judging
    from .report import out_dir_for, print_report, write_in_progress, write_reports

    cfg = load_config(args.config)
    if (getattr(args, "mode", None) or cfg.general.mode) == "top":
        from .top_run import run_top

        return run_top(args, cfg, confirm)
    max_clips = args.max_clips or cfg.general.max_clips
    backend = resolve_backend(cfg, args.judge_backend)
    scripted = backend == "scripted"
    state: dict = {}

    def on_ingested(meta) -> None:
        # From here on, an interrupted run leaves an IN PROGRESS marker, never the previous verdict.
        state["out_dir"] = out_dir = out_dir_for(meta, cfg, scripted=scripted)
        if not args.dry_run:
            stash_exports(out_dir)
            write_in_progress(meta, cfg, out_dir, scripted=scripted, backend=backend)

    a = analyze(args.source, cfg, rerun=set(args.rerun), judge_backend=args.judge_backend,
                scripted_judge=args.scripted_judge, on_ingested=on_ingested)
    out_dir = state["out_dir"]
    ctx = build_context(cfg, a.llm, a.transcript, a.signals, a.frames, a.meta.title, a.meta.uploader,
                        a.meta.duration)

    def on_estimate(est: dict) -> bool:
        print_estimate(est)
        emit("estimate", **est, threshold=cfg.judging.confirm_above_usd)
        if args.dry_run:
            return False
        return confirm(est, cfg)

    with stage("Judging"):
        console.print(f"  [dim]judging on {BACKEND_LABEL.get(backend, backend)}[/dim]")
        result = run_judging(a.candidates, ctx, max_clips=max_clips, console=console, on_estimate=on_estimate,
                             recall_complete=a.recall_complete)
    if args.dry_run:
        console.print("[dim]--dry-run: stopped before judging.[/dim]")
        return {"code": 0, "out_dir": out_dir, "status": "dry_run"}
    if result.status == "unavailable":
        # Nothing was judged (cancelled at the cost prompt, or no credentials): leave the previous
        # results exactly as they were. With nothing to restore, report why judging didn't run.
        if restore_all(out_dir):
            console.print("[yellow]Judging did not run; the previous report and clips were left in place.[/yellow]")
        else:
            write_reports(a, result, cfg, out_dir, explain_rejects=args.explain_rejects, scripted=scripted)
        for n in result.notes:
            console.print(f"[yellow]{n}[/yellow]")
        return {"code": 3, "out_dir": out_dir, "status": "unavailable", "notes": result.notes}
    keep = expected_files(out_dir, a, result, cfg)
    restore_exports(out_dir, keep)
    paths = write_reports(a, result, cfg, out_dir, explain_rejects=args.explain_rejects, scripted=scripted)
    exported = {}
    if result.great and not args.no_cut:
        with stage("Export"):
            exported = export_clips(a, result, cfg, out_dir, force="export" in set(args.rerun))
        paths = write_reports(a, result, cfg, out_dir, explain_rejects=args.explain_rejects, exported=exported,
                              scripted=scripted)
    # Keep exports the current verdict justifies (even under --no-cut); remove everything else.
    for p in prune_stale_clips(out_dir, keep):
        console.print(f"  [dim]removed stale export that is no longer GREAT: {p.name}[/dim]")
    print_report(a, result, cfg, paths, explain_rejects=args.explain_rejects, exported=exported)
    return {"code": 0, "out_dir": out_dir, "status": result.status, "notes": result.notes}
