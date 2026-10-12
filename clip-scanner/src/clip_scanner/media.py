"""Thin wrappers around ffmpeg / ffprobe."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


class MediaError(RuntimeError):
    pass


# Homebrew's regular `ffmpeg` formula no longer includes libass (burned-in captions). `ffmpeg-full` does,
# but it is keg-only, so it isn't on PATH. When it is installed we put it first on PATH, so every
# program that shells out to ffmpeg (our exports, yt-dlp, mlx-whisper) uses the same full build.
FFMPEG_FULL_DIRS = ("/opt/homebrew/opt/ffmpeg-full/bin", "/usr/local/opt/ffmpeg-full/bin",
                    "/home/linuxbrew/.linuxbrew/opt/ffmpeg-full/bin")
INSTALL_HINT = ("brew install ffmpeg-full" if platform.system() == "Darwin"
                else "install ffmpeg (with libass) using your package manager")


def prefer_full_ffmpeg(dirs: tuple[str, ...] | None = None) -> str | None:
    """Put a keg-only Homebrew ffmpeg-full first on PATH. Returns its directory, if found."""
    if dirs is None:
        prefix = os.environ.get("HOMEBREW_PREFIX")
        dirs = ((f"{prefix}/opt/ffmpeg-full/bin",) if prefix else ()) + FFMPEG_FULL_DIRS
    for d in dirs:
        exe = Path(d) / "ffmpeg"
        if exe.is_file() and os.access(exe, os.X_OK):
            parts = os.environ.get("PATH", "").split(os.pathsep)
            if d not in parts:
                os.environ["PATH"] = os.pathsep.join([d, *[p for p in parts if p]])
            return d
    return None


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"`{name}` not found on PATH ({INSTALL_HINT}).")
    return path


@lru_cache(maxsize=8)
def _filters(binary: str) -> str:
    try:
        return subprocess.run([binary, "-hide_banner", "-filters"], capture_output=True, text=True,
                              timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def has_filter(name: str) -> bool:
    """Whether the ffmpeg on PATH has this filter (e.g. "ass" needs libass, "sendcmd")."""
    binary = shutil.which("ffmpeg")
    return bool(binary) and any(line.split()[1:2] == [name] for line in _filters(binary).splitlines())


def captions_burnable(cfg) -> bool:
    """Captions are on and this ffmpeg can burn them in."""
    return cfg.captions.enabled and has_filter("ass")


def run(cmd: list[str], *, capture: bool = True, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=capture)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[-3000:] if capture and proc.stderr else ""
        raise MediaError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{err}")
    return proc


@dataclass
class ProbeInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    has_video: bool
    rotation: int = 0

    @property
    def display_size(self) -> tuple[int, int]:
        if self.rotation in (90, 270, -90, -270):
            return self.height, self.width
        return self.width, self.height


def probe(path: Path) -> ProbeInfo:
    require_binary("ffprobe")
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or (v or a or {}).get("duration") or 0.0)
    fps = 0.0
    rotation = 0
    width = height = 0
    if v:
        width, height = int(v.get("width", 0)), int(v.get("height", 0))
        rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
        num, _, den = rate.partition("/")
        try:
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        except ValueError:
            fps = 0.0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rotation = int(sd["rotation"])
        rotation = int(v.get("tags", {}).get("rotate", rotation))
    return ProbeInfo(duration=duration, width=width, height=height, fps=fps or 30.0,
                     has_audio=a is not None, has_video=v is not None, rotation=rotation)


@lru_cache(maxsize=1)
def hwaccel_args(mode: str = "auto") -> tuple[str, ...]:
    """Hardware decode flags. videotoolbox on macOS makes 1080p decode several times faster."""
    if mode != "auto" or platform.system() != "Darwin":
        return ()
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True).stdout
    except OSError:
        return ()
    return ("-hwaccel", "videotoolbox") if "videotoolbox" in out else ()


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000) -> None:
    require_binary("ffmpeg")
    tmp = dst.with_suffix(".tmp.wav")
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le", str(tmp)])
    tmp.replace(dst)


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM mono WAV written by extract_audio as float32 in [-1, 1]."""
    import wave

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise MediaError(f"{path} is not 16-bit mono PCM")
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sr


def make_proxy(src: Path, dst: Path, height: int, fps: int, hwaccel: str = "auto") -> None:
    """Low-res, low-fps video-only proxy for fast scene detection."""
    tmp = dst.with_suffix(".tmp.mp4")
    run(["ffmpeg", "-y", "-v", "error", *hwaccel_args(hwaccel), "-i", str(src), "-an",
         "-vf", f"fps={fps},scale=-2:{height}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
         "-pix_fmt", "yuv420p", str(tmp)])
    tmp.replace(dst)


def extract_frame(src: Path, t: float, dst: Path, width: int, vf: str | None = None) -> None:
    """Grab one frame at time t (fast input seek) as JPEG: scaled to `width`, or through a custom filter."""
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(src), "-frames:v", "1",
         "-vf", vf or f"scale={width}:-2", "-q:v", "4", str(dst)])


def read_frames(src: Path, start: float, end: float, fps: float, width: int) -> tuple[np.ndarray, list[float]]:
    """Decode [start, end) at `fps` into an (N, H, W, 3) uint8 RGB array plus timestamps."""
    info = probe(src)
    w0, h0 = info.display_size
    height = int(round(h0 * width / max(w0, 1) / 2) * 2)
    proc = run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{max(end - start, 0.05):.3f}",
                "-i", str(src), "-vf", f"fps={fps},scale={width}:{height}", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-"])
    frame_bytes = width * height * 3
    n = len(proc.stdout) // frame_bytes
    frames = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(n, height, width, 3)
    times = [start + i / fps for i in range(n)]
    return frames, times
