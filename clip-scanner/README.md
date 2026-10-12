# clip-scanner

Scans one long-form Joe Bartolozzi video (YouTube or TikTok URL, or a local file) and returns **only the
GREAT short-form clips**. Those are the moments worth editing and posting to a clipping account on TikTok,
YouTube Shorts and Reels.

It is built to be ruthless. A mediocre clip you waste time posting costs more than a good one it
misses, so every uncertain path ends in **reject**. **"NO GREAT CLIPS" is a normal, expected
answer.** You still get the three closest near-misses and exactly what killed each one.

What a run prints, in shape (numbers depend on the video):

```
▸ Ingest  ▸ Transcribe  ▸ Signals  ▸ Candidates  ▸ Keyframes
▸ Judging
  N/M candidates passed the deterministic gates; K need judging with claude-opus-5-5 (effort high).
  Cost estimate: ~$X expected (worst case $Y …)       ← asks before spending above $5
  c07 rejected  jaded_scroller: gate 'standalone' failed …
  c12 GREAT
╭── clip-scanner · <video title> ──╮
│ 1 GREAT CLIP   (or: NO GREAT CLIPS + one-line read) │
╰─────────────────────────────────────────────────────╯
Actual API cost this run: $Z
```

## Setup (M-series Mac)

```bash
brew install ffmpeg uv
cd ~/clip-scanner
uv sync                                  # Python 3.12 + all deps (mlx-whisper on Apple Silicon)
export ANTHROPIC_API_KEY=sk-ant-...      # or: ant auth login
uv run clip-scanner doctor               # checks ffmpeg/libass, credentials, backends
```

Linux also needs `apt install ffmpeg libegl1 libgles2` (MediaPipe's native library links EGL).
Optional speaker diarization: `uv sync --extra diarize` plus a Hugging Face token in `HF_TOKEN` that
has the pyannote models accepted. Set `[diarization] enabled = true`.

## Usage

```bash
uv run clip-scanner <youtube/tiktok URL or file> [--max-clips 5] [--no-cut] [--explain-rejects]

uv run clip-scanner "https://www.youtube.com/watch?v=…"            # full scan + exports
uv run clip-scanner ~/Downloads/stream.mp4 --no-cut                  # report only
uv run clip-scanner URL --explain-rejects                            # list every rejected candidate + reason
uv run clip-scanner URL --dry-run                                    # stop at the cost estimate
uv run clip-scanner URL --rerun judging                              # ignore cached verdicts
uv run clip-scanner calibrate --fetch                                # measure + tune the judges (see below)
```

Every stage is cached per video under `~/.cache/clip-scanner/<video-id>/`. Re-running the same
video skips download, transcription, signals and any judge call it has already paid for. Changing a
threshold in `config.toml` re-applies the rule to the cached verdicts for $0.

### Output

`./clip-scanner-output/<title>-<id>/`

- `report.md` / `report.json`. For each GREAT clip: rank, start–end timestamps, duration, overall score
  (the *lowest* judge's effective score), the one-line hook, why it will go viral, a caption/title of
  8 words or fewer, posting hook text, hashtags, every judge's score, the devil's advocate's case and the arbiter's
  ruling. With no GREAT clips: the **NO GREAT CLIPS** verdict, the top 3 near-misses with the exact gate or
  score that killed each, and a one-line read on why the video lacks clip material.
- `clips/01-<caption>-cNN.mp4`: 1080×1920, face-tracked crop, 0.3 s pre-roll, word-by-word captions
  burned in (bold, outlined, centered in the lower third).
- `clips/01-<caption>-cNN-raw.mp4`: the **untouched landscape footage** with 3 s of handles on both
  sides, for your own edit. The report says where the clip starts and ends inside it.

Exports from an earlier run that are no longer GREAT are deleted, so `clips/` always matches the latest verdict.

## How it decides

| Stage | What happens |
|---|---|
| 1. Ingest | yt-dlp, best stream at or below 1080p (or a local file). |
| 2. Transcribe | mlx-whisper large-v3 on Apple Silicon (faster-whisper elsewhere), word timestamps, optional diarization. |
| 3. Signals | 100 ms loudness, YAMNet laughter/scream/crowd/music/speech, speech rate, Joe catchphrases and reaction words, PySceneDetect cuts, silence gaps. |
| 4. Candidates (recall) | Peaks of the combined interest curve, plus a cheap Haiku pass over the transcript for deadpan or verbal moments. Several window lengths per moment, snapped so a clip **starts on a sentence or scene edge** and **ends after the payoff and its reaction, never mid-word**. Dead air over 1.5 s is jump-cut out. This yields 15–40 windows, each with keyframes. |
| 5a. Hard gates | Measured first, for free: length 12–60 s, something audible within the first second, no dead air over 1.5 s, no music bed, has speech. |
| 5b. Panel | Three blind Opus judges: a **jaded scroller**, a **Joe superfan** and a **viral-shorts editor**. Each sees the clip transcript, NOT-IN-CLIP context before and after, the measured timeline and keyframes. Each returns structured JSON: five gates with evidence, eight 0–10 rubric scores with justifications, a holistic overall, and a GREAT/NOT_GREAT vote. |
| 5c. Unanimity | GREAT only if **every** judge passes **every** gate, votes GREAT, and has an effective score ≥ 8.5. The effective score is `min(overall, rubric mean + 1)`, so a judge can't say "9" while its own rubric says "7". Hook and payoff must each be ≥ 8, and the rubric mean ≥ 7.5. A missing, refused, malformed or out-of-range verdict is a rejection. |
| 5d. Devil's advocate | A fourth agent argues each finalist will flop. A fresh arbiter rules on that case. If the case is convincing, or survival confidence is below 7/10, the clip is dropped. |
| 6. Output | Absolute bar: clips are judged one at a time against fixed anchors, never ranked against each other. `--max-clips` only truncates the GREAT list and never fills it. |

The judges' notion of "Joe" comes from [`docs/joe-style-guide.md`](docs/joe-style-guide.md), which
covers his format, recurring bits, catchphrases, what performs, what flops and monetization red flags. It is
included verbatim in every judge's system prompt. Edit it and the next run uses it.

Claude API details: structured outputs (`output_config.format`) for every call; adaptive thinking
at `effort = "high"`; the handbook, style guide and full transcript sit in cached system blocks
(`cache_control`), so the three personas, the devil's advocate and the arbiter all read the same cached prefix.
Server-side refusal fallbacks (`fallbacks: "default"`) are on by default (`[judging] server_fallbacks`).
A refused or failed call never counts as a pass.

## Tightening or loosening the bar

All in `config.toml`, no code changes needed:

| Knob | Default | Effect |
|---|---|---|
| `[judging] overall_min` | 8.5 | every judge's overall must reach this |
| `[judging] rubric_mean_min` | 7.5 | anti-inflation floor on each judge's weighted rubric |
| `[judging.criterion_floors]` | hook 8, payoff 8 | per-criterion minimums |
| `[judging] gate_policy` | `any` | `any` = one judge failing one gate kills; `majority` tolerates one dissent |
| `[judging] judges` | 3 personas | remove one for a cheaper 2-judge panel |
| `[devils_advocate] min_survival_confidence` | 7.0 | arbiter confidence needed to survive |
| `[gates] max_dead_air_s`, `min_len_s`, `max_len_s` | 1.5 / 12 / 60 | deterministic gates |
| `[candidates] max_candidates` | 40 | recall breadth (and cost) |
| `[judging] effort` / `model` | high / claude-opus-5-5 | quality vs. cost |

## Cost

The tool prints an estimate before judging and asks for confirmation above `confirm_above_usd`
($5 by default; `--yes` skips the prompt). It prints the actual spend afterwards. Opus 5.5 costs $4 per million
input tokens and $20 per million output. Thinking output dominates, so `effort` is the main lever.
Short-circuiting means most candidates cost one judge call: under unanimity the first rejection
settles it (property-tested to give identical outcomes). For a 1-hour video with 40 candidates, the
built-in estimator predicts about $6 expected at `effort = "high"` ($21 worst case, if every clip went to
the full panel and the devil's advocate), and about $4.50 expected at `"medium"`. These are pre-run
estimates from assumed token counts, not measured bills. The real figure is printed after each run, and
re-runs of judged candidates are free.

## Calibration

```bash
uv run clip-scanner calibrate --fetch            # judge known hits and flops, report precision
uv run clip-scanner calibrate --write-config     # apply the recommended thresholds
```

Add clips with known view counts to [`calibration/manifest.toml`](calibration/manifest.toml) (local
files or URLs). Each is judged as posted by the full panel plus the devil's advocate. You get
precision, recall, the share of known flops that would pass, and the score gap between your weakest
hit and your strongest flop. The command then sweeps the thresholds over the cached verdicts and
recommends the smallest change that rejects every known flop. See [`calibration/README.md`](calibration/README.md).

**Your own flops are the most valuable data.** The tool exists to reject clips like them.

## Tests

```bash
uv run pytest                     # unit tests + end-to-end run on tests/fixtures/sample.mp4
CLIP_SCANNER_LIVE=1 uv run pytest tests/test_e2e.py   # same, with the real Opus panel (few cents)
```

The unit tests cover boundary snapping (including randomized never-mid-word / never-before-payoff
invariants), the deterministic gates, the consensus rule (every way a merely-good clip could slip
through, plus short-circuit equivalence), the devil's-advocate rule, calibration sweeps, captions,
crop smoothing, and the real Anthropic request and response path against a local mock server.
The end-to-end test runs the whole CLI on a 40-second synthetic stream: TTS speech, a stand-in face
(a public-domain NASA portrait) panning across the frame, and two scene cuts. It checks the report,
gates, boundaries, the 1080×1920 export (face stays centered, audio present), the raw cut,
resumability, and the NO GREAT CLIPS path. It uses faster-whisper tiny.en (75 MB download on the first
run) and an offline scripted judge, unless `CLIP_SCANNER_LIVE=1`.
`scripts/make_sample_video.py` regenerates the fixture.

## Limitations

- Judges can't hear audio or see motion. They get the transcript, a measured audio/visual timeline
  and 7 keyframes per clip. Physical comedy with no sound or words is under-detected.
- During gameplay the face crop follows Joe's webcam overlay. There's no split-screen
  (facecam over gameplay) layout yet; use the raw cut for those.
- YAMNet music detection is a heuristic. The judges re-check monetization safety, but you should
  still listen before posting.
- TikTok and members-only or age-gated YouTube may need `[ingest] cookies_from_browser = "chrome"`.

## Repo layout

This project currently lives in the `clip-scanner/` folder of the `Personal` repo. To turn it into
its own repo at `~/clip-scanner` with history:

```bash
git -C /path/to/Personal subtree split --prefix=clip-scanner -b clip-scanner-standalone
git clone -b clip-scanner-standalone /path/to/Personal ~/clip-scanner
```
