"""Orchestrates the scan: ingest → transcribe → signals → candidates → judging → report → export."""

from __future__ import annotations

import argparse

from .config import load_config
from .ingest import ingest
from .ui import console, stage


def run_scan_cli(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    with stage("Ingest") as st:
        meta, ws = ingest(args.source, cfg, console=console)
        st["note"] = f"{meta.title} · {meta.duration / 60:.1f} min · {meta.width}x{meta.height}"
    console.print(f"[dim]workspace: {ws.root}[/dim]")
    return 0
