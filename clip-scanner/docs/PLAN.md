# clip-scanner: build plan

Goal: scan one long-form Joe Bartolozzi video and return only GREAT short-form clips.
Saying "no great clips" is a valid result. A false positive (posting a mediocre clip)
costs more than a false negative (missing a good one), so every ambiguous path in the
design resolves to **reject**.

## Pipeline

| # | Stage | Cost | Output (cached under `<cache_dir>/<video_id>/`) |
|---|-------|------|-----------------------------------------------|
| 1 | **Ingest**: yt-dlp (≤1080p) or a local file; ffprobe metadata; 16 kHz mono WAV | network | `meta.json`, `source.*`, `audio.wav` |
| 2 | **Transcribe**: mlx-whisper (Apple Silicon) / faster-whisper, large-v3, word timestamps; optional pyannote diarization | CPU/GPU | `transcript.json` |
| 3 | **Signals**: 100 ms loudness, YAMNet audio events (laughter, crowd, scream, music, speech), speech rate, PySceneDetect cuts on a 360p proxy, silence gaps, and a combined per-second interest curve | CPU | `signals.npz`, `signals.json` |
| 4 | **Candidates** (recall): heuristic peaks plus an optional cheap Haiku transcript proposer, window variants snapped to sentence/beat edges, dead-air tightening (jump cuts), NMS, 15–40 windows | cheap | `candidates.json`, `keyframes/` |
| 5 | **Judging** (precision): deterministic gates, then 3 blind persona judges (Opus, structured JSON, cached prefix), a unanimity rule, then a devil's advocate and an independent arbiter | expensive | `judgments/*.json` |
| 6 | **Output**: terminal, `report.md`, `report.json`; NO GREAT CLIPS verdict with near-misses; `--explain-rejects`; 1080×1920 export with face-tracked crop, 0.3 s pre-roll and word-by-word burned captions | CPU | `<output_dir>/<video>/` |

Every stage writes a fingerprint (stage version + relevant config + upstream fingerprint).
Re-runs skip finished stages. Changing only thresholds re-runs only the verdict
aggregation: raw judge outputs are cached by request hash, so tightening the bar costs $0.

## Precision design (where a good-but-not-great clip gets killed)

1. **Deterministic pre-gates** (free, run before any API call): length 12–60 s; no dead
   air > 1.5 s (no words, no reaction audio, no loudness spike); first speech or a
   high-signal event within the hook window; music bed not dominating the clip.
2. **LLM gates**, judged by every judge: hook in the first 1.5–2 s, standalone, clear payoff
   before the end, no dead air, monetization/TOS safe. **One judge failing one gate kills the clip.**
3. **Scores**: each judge returns 8 rubric scores (0–10, with justification) plus a holistic
   overall. A judge's *effective* score is `min(overall, weighted rubric mean + allowance)`, so a judge cannot
   praise a clip in the holistic number while its own rubric says "average". Hook and
   payoff also have per-criterion floors.
4. **Unanimity**: GREAT requires *all* judges to clear the bar. A missing, malformed,
   out-of-range or refused verdict counts as a rejection (fail closed).
5. **Short-circuit**: under unanimity the first rejecting judge decides, so the remaining
   judges are skipped. This saves about 2/3 of the spend without changing any outcome.
   Calibration always runs the full panel.
6. **Devil's advocate + arbiter**: for each finalist, an adversarial agent writes the strongest
   flop case, and a fresh arbiter decides whether it is convincing. If it is convincing, or
   the arbiter is not confident the clip survives, the clip is dropped.
7. **Absolute bar**: candidates are judged one at a time, never side by side, and the
   prompts anchor the scale to absolute outcomes. `--max-clips` only truncates the GREAT
   list and never fills it. Zero GREAT clips gives a "NO GREAT CLIPS" report.

## Calibration

`calibration/manifest.toml` lists known hits and flops (local file or URL, plus view
counts). `clip-scanner calibrate` runs the full panel on each clip and reports
precision, recall and the flop pass-through rate. It then sweeps thresholds over the cached
raw scores and suggests (or writes with `--write-config`) the loosest bar that still rejects every known flop.

## Engineering

Python 3.12, uv, one `clip-scanner` entry point, `config.toml` for every threshold.
Claude calls use structured outputs (`output_config.format`), prompt caching on the
shared rubric and style-guide prefix and on the per-video transcript prefix, and adaptive thinking
at configurable effort. Server-side refusal fallbacks are on by default.
The tool prints a cost estimate before judging, asks for confirmation above a configurable
amount, and prints the actual cost afterwards.

Tests: boundary snapping, gates, panel consensus, cost, captions, plus an end-to-end
run on a short sample video. That run uses a scripted judge offline, or real Opus when an API key is
set and `CLIP_SCANNER_LIVE=1`.

## Phases (one commit each)

0. Plan + Joe style guide (research)
1. Scaffold: pyproject/uv, config, cache/stages, ingest, CLI skeleton
2. Transcription + signals
3. Candidates, boundary snapping, keyframes (+ tests)
4. Judging: gates, panel, devil's advocate, cost (+ tests)
5. Report + export (face crop, captions)
6. Calibrate command, e2e test, README
7. Adversarial review fixes
