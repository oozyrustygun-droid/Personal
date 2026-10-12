"""Orchestrates the scan: ingest → transcribe → signals → candidates → judging → report → export."""

from __future__ import annotations

import argparse

from .candidates import candidates_stage
from .config import load_config
from .cost import CostTracker
from .ingest import ingest
from .keyframes import extract_keyframes
from .llm import LLM
from .signals import signals_stage
from .transcribe import transcribe
from .ui import console, fmt_ts, stage


def run_scan_cli(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    rerun = set(args.rerun)
    with stage("Ingest") as st:
        meta, ws = ingest(args.source, cfg, console=console)
        st["note"] = f"{meta.title} · {meta.duration / 60:.1f} min · {meta.width}x{meta.height}"
    console.print(f"  [dim]workspace: {ws.root}[/dim]")
    with stage("Transcribe") as st:
        before = ws.stage_fingerprint("transcribe")
        transcript, t_fp = transcribe(ws, cfg, console=console, force="transcribe" in rerun)
        st["cached"] = before == t_fp and "transcribe" not in rerun
        st["note"] = f"{len(transcript.words)} words ({transcript.backend}/{transcript.model})"
    with stage("Signals") as st:
        before = ws.stage_fingerprint("signals")
        signals, s_fp = signals_stage(ws, transcript, t_fp, meta.duration, meta.path, cfg, console=console,
                                      force="signals" in rerun)
        st["cached"] = before == s_fp and "signals" not in rerun
        st["note"] = f"{len(signals.scene_cuts)} scene cuts · audio events {'on' if signals.events_available else 'off'}"
    tracker = CostTracker(cfg)
    llm = make_llm(cfg, ws, tracker, args)
    with stage("Candidates") as st:
        before = ws.stage_fingerprint("candidates")
        cands, c_fp = candidates_stage(ws, transcript, signals, s_fp, meta.duration, cfg, llm, console=console,
                                       force="candidates" in rerun)
        st["cached"] = before == c_fp and "candidates" not in rerun
        st["note"] = f"{len(cands)} candidate windows"
    with stage("Keyframes"):
        frames = extract_keyframes(ws, meta.path, cands, signals, cfg, c_fp)
    for c in cands:
        console.print(f"  [dim]{c.id} {fmt_ts(c.start)}–{fmt_ts(c.end)} ({c.duration:.1f}s, {c.jump_cuts} jc) "
                      f"score={c.score:.2f} {c.origin}: {c.reason}[/dim]")
    return 0


def make_llm(cfg, ws, tracker, args) -> LLM:
    if getattr(args, "judge_backend", "anthropic") == "scripted":
        from .scripted import load_script

        return LLM(cfg, ws.path("llm"), tracker, backend="scripted", script=load_script(args.scripted_judge))
    return LLM(cfg, ws.path("llm"), tracker)
