"""Build tests/fixtures/sample.mp4: a ~40 s, Joe-shaped mini stream for the end-to-end test.

Structure (deliberately mixed):
  0-9 s    low-energy filler ("so anyway chat ...")       -> should NOT be a clip on its own
  ~2.5 s   dead air
  12-25 s  setup -> loud payoff ("WHAT IS THAT ... never eating that again") + laughter
  ~1 s     pause
  26-40 s  low-energy wrap-up
A real face (NASA astronaut photo, public domain, shipped with scikit-image) pans across a
16:9 frame so the face-tracked vertical crop has something to follow. The background colour
changes twice to give PySceneDetect two cuts.

Needs a TTS engine: `say` (macOS) or `espeak-ng` (Linux). Usage:
    uv run python scripts/make_sample_video.py [--face tests/fixtures/face.png]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
SR = 22050

LINES = [
    # (start_s, text, gain, pitch, speed)
    (0.4, "So anyway chat, I was looking at this snack I bought today.", 0.45, 45, 160),
    (4.6, "Honestly it is just okay. Nothing special.", 0.45, 45, 160),
    (12.0, "Wait. Hold on. Chat, look at this.", 0.7, 55, 175),
    (15.0, "What is that? What is that thing?", 0.9, 70, 185),
    (18.0, "Oh my god! That is the worst thing I have ever tasted!", 1.0, 80, 190),
    (21.6, "I am never eating that again!", 1.0, 85, 190),
    (23.6, "ha ha ha ha ha ha ha", 1.0, 95, 230),
    (26.5, "Okay. Let's move on to the next one.", 0.45, 45, 160),
    (30.0, "This one is fine I guess. Type in chat what you think.", 0.45, 45, 160),
    (35.0, "Yeah. It is fine. Moving on.", 0.4, 45, 150),
]
DURATION = 40.0
SCENE_CUTS = (11.5, 26.0)


def tts(text: str, pitch: int, speed: int, out: Path) -> None:
    if shutil.which("espeak-ng"):
        subprocess.run(["espeak-ng", "-v", "en-us", "-p", str(pitch), "-s", str(speed), "-w", str(out), text],
                       check=True)
    elif shutil.which("say"):
        aiff = out.with_suffix(".aiff")
        subprocess.run(["say", "-r", str(speed + 20), "-o", str(aiff), text], check=True)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ar", str(SR), "-ac", "1", str(out)],
                       check=True)
    else:
        raise SystemExit("need `espeak-ng` (Linux) or `say` (macOS) to synthesise speech")


def read_wav(path: Path) -> np.ndarray:
    tmp = path.with_suffix(".22k.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-ar", str(SR), "-ac", "1",
                    "-c:a", "pcm_s16le", str(tmp)], check=True)
    with wave.open(str(tmp), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def build_audio(tmpdir: Path) -> Path:
    track = np.zeros(int(DURATION * SR), dtype=np.float32)
    rng = np.random.default_rng(7)
    track += rng.normal(0, 0.003, track.shape).astype(np.float32)  # faint room tone
    for i, (start, text, gain, pitch, speed) in enumerate(LINES):
        raw = tmpdir / f"line{i}.wav"
        tts(text, pitch, speed, raw)
        clip = read_wav(raw)
        clip = clip / max(np.abs(clip).max(), 1e-6) * gain
        s = int(start * SR)
        e = min(len(track), s + len(clip))
        track[s:e] += clip[: e - s]
    track = np.clip(track, -1, 1)
    out = tmpdir / "audio.wav"
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((track * 32767).astype("<i2").tobytes())
    return out


def ensure_face(face: Path) -> Path:
    if face.exists():
        return face
    try:
        from skimage import data  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit("pass --face PATH or `uv pip install scikit-image pillow`") from e
    img = data.astronaut()[0:300, 120:330]  # head and shoulders
    face.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(face)
    return face


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face", type=Path, default=FIX / "face.png")
    ap.add_argument("--out", type=Path, default=FIX / "sample.mp4")
    args = ap.parse_args()
    face = ensure_face(args.face)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        audio = build_audio(tmp)
        c1, c2 = SCENE_CUTS
        colors = [("0x30343f", c1), ("0x5a1e2b", c2 - c1), ("0x1e4a3a", DURATION - c2)]
        inputs: list[str] = []
        for col, dur in colors:
            inputs += ["-f", "lavfi", "-i", f"color=c={col}:s=1280x720:r=30:d={dur:.3f}"]
        inputs += ["-loop", "1", "-i", str(face), "-i", str(audio)]
        # Face pans left -> right, faster during the payoff (12-25 s).
        x_expr = "if(lt(t,12),120+20*t,if(lt(t,25),360+38*(t-12),854+8*(t-25)))"
        fc = (
            "[0:v][1:v][2:v]concat=n=3:v=1:a=0[bg];"
            "[3:v]scale=300:-2[face];"
            f"[bg][face]overlay=x='{x_expr}':y=(H-h)/2:shortest=1,format=yuv420p[v]"
        )
        subprocess.run(["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", fc, "-map", "[v]",
                        "-map", "4:a", "-t", f"{DURATION}", "-c:v", "libx264", "-preset", "slow", "-crf", "30",
                        "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(args.out)], check=True)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
