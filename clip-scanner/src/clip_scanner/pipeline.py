"""Orchestrates the scan: ingest → transcribe → signals → candidates → judging → report → export."""

from __future__ import annotations

import argparse

from .config import load_config
from .ingest import ingest
from .signals import signals_stage
from .transcribe import transcribe
from .ui import console, stage


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
    return 0
