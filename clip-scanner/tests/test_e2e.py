"""End-to-end: the real CLI on tests/fixtures/sample.mp4 (40 s, TTS speech, a real face, two cuts).

Transcription uses faster-whisper tiny.en when its weights are available (first run downloads
about 75 MB), otherwise pocketsphinx. Override with CLIP_SCANNER_TEST_ASR=faster-whisper|mlx|pocketsphinx.

Judging uses the offline scripted backend by default, which exercises every stage except the model's
taste. Set CLIP_SCANNER_LIVE=1 to run the real Opus panel instead, on the backend in config.toml
(claude-code: your Claude subscription; add CLIP_SCANNER_LIVE_BACKEND=api to spend API credits);
live mode only asserts that a valid report comes out.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from clip_scanner.boundaries import straddles
from clip_scanner.cli import main
from clip_scanner.transcribe import Transcript

from conftest import pick_asr

pytestmark = pytest.mark.e2e
FIX = Path(__file__).parent / "fixtures" / "sample.mp4"
LIVE = os.environ.get("CLIP_SCANNER_LIVE") == "1"
LIVE_BACKEND = os.environ.get("CLIP_SCANNER_LIVE_BACKEND")  # None: config.toml's judging.backend
# Scripted stand-in for taste: the payoff lines make a clip great, but a clip that opens on the slow
# "so anyway chat" wind-up fails the hook gate, as the real panel is instructed to do.
GREAT_RULES = {"great_phrases": ["oh my god", "never eating", "worst thing"], "great_score": 9.2,
               "default_score": 6.2, "hook_fail_openers": ["so anyway"]}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    backend, model = pick_asr()
    cfg = tmp / "config.toml"
    cfg.write_text(
        f'[general]\ncache_dir = "{tmp / "cache"}"\noutput_dir = "{tmp / "out"}"\n'
        f'[transcribe]\nbackend = "{backend}"\nmodel = "{model}"\n'
        '[judging]\nconfirm_above_usd = 1000.0\ntext_hook = true\n')
    great = tmp / "great.json"
    great.write_text(json.dumps(GREAT_RULES))
    dull = tmp / "dull.json"
    dull.write_text(json.dumps({"great_phrases": [], "default_score": 6.2}))
    return {"tmp": tmp, "cfg": cfg, "great": great, "dull": dull}


def run(env, rules: Path, *extra: str) -> dict:
    if LIVE:
        judge = ["--judge-backend", LIVE_BACKEND] if LIVE_BACKEND else []
    else:
        judge = ["--judge-backend", "scripted", "--scripted-judge", str(rules)]
    code = main([str(FIX), "--config", str(env["cfg"]), "--yes", *judge, *extra])
    assert code == 0
    reports = list((env["tmp"] / "out").glob("*/report.json"))
    assert len(reports) == 1
    return json.loads(reports[0].read_text())


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def first(env):
    return run(env, env["great"], "--explain-rejects")


def test_report_and_great_clip(first, env):
    assert first["candidates"] >= 1
    if LIVE:
        assert first["verdict"] in ("NO GREAT CLIPS",) or first["verdict"].endswith(("GREAT CLIP", "GREAT CLIPS"))
        assert first["cost"]["api_equivalent_usd"] > 0
        assert (first["cost"]["total_usd"] > 0) == (first["cost"]["billing"] == "api")
        return
    assert first["great_clips"], first["near_misses"]
    # Offline runs are unmistakably labelled and kept out of the real output folder.
    assert first["judge_backend"] == "scripted" and first["verdict"].startswith("[SCRIPTED TEST JUDGE")
    assert next((env["tmp"] / "out").glob("*/report.json")).parent.name.endswith("-SCRIPTED-TEST")
    top = first["great_clips"][0]
    # The window must contain the planted payoff and start no earlier than its setup.
    assert top["start"] <= 18.0 and top["end"] >= 22.5, top
    assert 10.0 <= top["start"], top
    assert 12.0 <= top["duration"] <= 60.0
    assert all(g["passed"] for g in top["gates"])
    assert 0 < len(top["caption_title"].split()) <= 8
    assert top["hook_line"] and top["posting_hook_text"]
    md = next((env["tmp"] / "out").glob("*/report.md")).read_text()
    assert "GREAT CLIP" in md
    assert ("Every rejected candidate" in md) == bool(first["rejected"])  # --explain-rejects section


def test_boundaries_never_cut_words(first, env):
    ws = next((env["tmp"] / "cache").glob("local-sample-*"))
    tr = Transcript.from_json(json.loads((ws / "transcript.json").read_text()))
    for c in first["great_clips"] + first["rejected"]:
        assert not straddles(tr.words, c["start"]), c
        assert not straddles(tr.words, c["end"]), c


def test_export_is_vertical_with_audio_and_tracks_face(first, env):
    if LIVE and not first["great_clips"]:
        pytest.skip("live judges found no GREAT clip to export")
    path = Path(first["great_clips"][0]["file"])
    assert path.exists()
    info = ffprobe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    dur = float(info["format"]["duration"])
    assert abs(dur - first["great_clips"][0]["duration"]) <= 0.45  # + pre-roll at most
    sidecar = json.loads(path.with_suffix(".json").read_text())
    top = first["great_clips"][0]
    # The burned-in hook is the hook the first judge wrote and every later judge was given, not a rewrite.
    assert top["text_hook"].startswith("Joe: ") and top["text_hook_burned"]
    assert sidecar["text_hook"] == top["caption_title"] == top["text_hook"]

    raw = Path(first["great_clips"][0]["raw_file"])  # untouched landscape cut with handles, for editing
    rv = next(s for s in ffprobe(raw)["streams"] if s["codec_type"] == "video")
    assert (rv["width"], rv["height"]) == (1280, 720)
    t_in, t_out = first["great_clips"][0]["raw_clip_in_out"]
    assert 0 < t_in < t_out <= float(ffprobe(raw)["format"]["duration"]) + 0.05

    from clip_scanner.config import load_config
    from clip_scanner.facetrack import detect_faces
    from clip_scanner.media import read_frames

    frames, _ = read_frames(path, 0.5, dur - 0.5, 1.0, 540)
    faces = [f for f in detect_faces(frames, load_config(str(env["cfg"]))) if f]
    assert len(faces) >= len(frames) * 0.6
    assert all(0.2 <= cx <= 0.8 for cx, _ in faces), faces  # the panning face stays in frame


def test_rerun_is_cached_and_no_great_path_prunes_exports(first, env):
    ws = next((env["tmp"] / "cache").glob("local-sample-*"))
    before = json.loads((ws / "stages.json").read_text())
    if LIVE:
        pytest.skip("the no-great path is exercised with the scripted judge only")
    second = run(env, env["dull"])
    after = json.loads((ws / "stages.json").read_text())
    for st in ("ingest", "transcribe", "signals", "candidates"):
        assert before[st]["completed_at"] == after[st]["completed_at"], st  # resumed, not recomputed
    assert second["verdict"] == "[SCRIPTED TEST JUDGE, NOT REAL JUDGING] NO GREAT CLIPS"
    assert not second["great_clips"]
    assert 1 <= len(second["near_misses"]) <= 3
    assert all(n["reasons"] for n in second["near_misses"])
    assert second["diagnosis"]
    out = next((env["tmp"] / "out").glob("*/"))
    assert not list((out / "clips").glob("*.mp4"))  # yesterday's GREAT clip is gone
