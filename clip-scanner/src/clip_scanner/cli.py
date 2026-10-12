"""`clip-scanner` entry point.

    clip-scanner <url-or-file> [--max-clips 5] [--no-cut] [--explain-rejects]
    clip-scanner calibrate [--manifest calibration/manifest.toml] [--fetch] [--write-config]
    clip-scanner doctor
    clip-scanner ui            # local web front end (paste/drop links, review clips)
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .ui import console


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to config.toml (default: ./config.toml, then the repo's)")
    p.add_argument("-y", "--yes", action="store_true", help="skip the cost confirmation prompt")
    p.add_argument("--judge-backend", choices=["anthropic", "scripted"], default="anthropic",
                   help=argparse.SUPPRESS)
    p.add_argument("--scripted-judge", help=argparse.SUPPRESS)


def scan_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clip-scanner",
        description="Find only the GREAT short-form clips in a long-form video. "
                    "Zero clips is a valid, expected answer.",
        epilog="Other commands: `clip-scanner ui` (web front end), `clip-scanner calibrate --help`, "
               "`clip-scanner doctor`.",
    )
    p.add_argument("source", help="YouTube/TikTok URL or a local video file")
    p.add_argument("--max-clips", type=_positive_int, default=None,
                   help="cap on GREAT clips returned (default from config: 5); never fills with non-GREAT")
    p.add_argument("--no-cut", action="store_true", help="report only; do not export vertical MP4s")
    p.add_argument("--explain-rejects", action="store_true", help="list every rejected candidate with its reason")
    p.add_argument("--dry-run", action="store_true", help="stop after candidate generation and show the cost estimate")
    p.add_argument("--rerun", nargs="+", default=[], metavar="STAGE",
                   choices=["ingest", "transcribe", "signals", "candidates", "judging", "export"],
                   help="force these stages to run again")
    p.add_argument("--version", action="version", version=f"clip-scanner {__version__}")
    _common(p)
    return p


def calibrate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clip-scanner calibrate",
        description="Run the full judging panel on known hits and flops, report precision, "
                    "and suggest thresholds that reject every known flop.",
    )
    p.add_argument("--manifest", help="manifest TOML (default from config: calibration/manifest.toml)")
    p.add_argument("--fetch", action="store_true", help="download manifest entries that only have a URL")
    p.add_argument("--write-config", action="store_true",
                   help="write the suggested thresholds into config.toml (needs >=10 hits and >=10 flops)")
    p.add_argument("--allow-loosen", action="store_true",
                   help="let --write-config lower a threshold (by default it only ever tightens)")
    p.add_argument("--only", nargs="+", metavar="ID", help="only these manifest ids")
    _common(p)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "calibrate":
            args = calibrate_parser().parse_args(argv[1:])
            from .calibrate import run_calibrate

            return run_calibrate(args)
        if argv and argv[0] == "doctor":
            from .doctor import run_doctor

            return run_doctor(argv[1:])
        if argv and argv[0] == "ui":
            from .server import run_ui

            return run_ui(argv[1:])
        args = scan_parser().parse_args(argv)
        from .pipeline import run_scan_cli

        return run_scan_cli(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Finished stages are cached; re-run to resume.[/yellow]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
