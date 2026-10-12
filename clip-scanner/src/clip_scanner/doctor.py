"""`clip-scanner doctor`: check that the local toolchain is ready."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys

from .ui import console


def _ok(flag: bool, label: str, detail: str = "") -> bool:
    mark = "[green]✓[/green]" if flag else "[red]✗[/red]"
    console.print(f"  {mark} {label}" + (f" [dim]{detail}[/dim]" if detail else ""))
    return flag


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

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    ant = shutil.which("ant")
    _ok(has_key or bool(ant), "Anthropic credentials",
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
