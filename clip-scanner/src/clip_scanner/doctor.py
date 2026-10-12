"""`clip-scanner doctor`: check that the local toolchain is ready."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from .cli_common import BACKEND_LABEL, resolve_backend
from .config import load_config
from .ui import console


def _ok(flag: bool, label: str, detail: str = "") -> bool:
    mark = "[green]✓[/green]" if flag else "[red]✗[/red]"
    console.print(f"  {mark} {label}" + (f" [dim]{detail}[/dim]" if detail else ""))
    return flag


def _check_claude_code(cfg) -> bool:
    from . import claude_code as cc

    exe = cc.find_claude(cfg.judging.claude_code.path if cfg else "")
    if not _ok(bool(exe), "Claude Code CLI (`claude`)", exe or cc.INSTALL_HINT):
        return False
    _ok(True, "Claude Code version", cc.version(exe) or "unknown")
    try:
        st = cc.check_subscription(exe, cfg.judging.claude_code.allow_api_billing if cfg else False)
    except cc.ClaudeCodeError as e:
        return _ok(False, "Claude Code logged in to your Claude subscription", str(e))
    return _ok(True, "Claude Code logged in to your Claude subscription",
               f"{st.get('authMethod')}; judging uses your plan's usage, not API credits")


def run_doctor(argv: list[str]) -> int:
    console.print("[bold]clip-scanner doctor[/bold]")
    good = True
    good &= _ok(sys.version_info[:2] == (3, 12), "Python 3.12", platform.python_version())
    ff = shutil.which("ffmpeg")
    good &= _ok(bool(ff), "ffmpeg on PATH", ff or "brew install ffmpeg")
    if ff:
        filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
        good &= _ok(" ass " in filters, "ffmpeg built with libass (burned-in captions)")
        _ok(" sendcmd " in filters, "ffmpeg sendcmd filter (face-tracked crop)")
    good &= _ok(bool(shutil.which("ffprobe")), "ffprobe on PATH")

    try:
        cfg = load_config()
    except Exception as e:  # noqa: BLE001 - report it, keep checking the rest
        good &= _ok(False, "config.toml", str(e)[:200])
        cfg = None
    backend = resolve_backend(cfg, None) if cfg else "claude-code"
    console.print(f"  [dim]judging backend: {BACKEND_LABEL.get(backend, backend)}[/dim]")
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if backend == "claude-code":
        good &= _check_claude_code(cfg)
        if has_key:
            console.print("  [dim]ANTHROPIC_API_KEY is set in this shell; clip-scanner hides it from Claude Code so "
                          "judging uses your subscription, not API credits.[/dim]")
    else:
        ant = shutil.which("ant")
        good &= _ok(has_key or bool(ant), "Anthropic API credentials (billed per token)",
                    "ANTHROPIC_API_KEY set" if has_key else ("`ant auth status` to check profile" if ant else
                                                              "export ANTHROPIC_API_KEY=... (needed for judging)"))

    backends = []
    if importlib.util.find_spec("mlx_whisper"):
        backends.append("mlx-whisper")
    if importlib.util.find_spec("faster_whisper"):
        backends.append("faster-whisper")
    if importlib.util.find_spec("pocketsphinx"):
        backends.append("pocketsphinx (test only)")
    good &= _ok(bool(backends), "transcription backend", ", ".join(backends) or "uv sync")
    _ok(bool(importlib.util.find_spec("pyannote")), "speaker diarization (optional)",
        "uv sync --extra diarize + HF_TOKEN" if not importlib.util.find_spec("pyannote") else "pyannote.audio")
    good &= _ok(bool(importlib.util.find_spec("mediapipe")), "mediapipe (face tracking, YAMNet audio events)")
    good &= _ok(bool(importlib.util.find_spec("scenedetect")), "PySceneDetect")
    good &= _ok(bool(importlib.util.find_spec("yt_dlp")), "yt-dlp")
    console.print("[green]ready[/green]" if good else "[red]fix the items marked ✗[/red]")
    return 0 if good else 1
