"""Keyframes per candidate so the judges can actually see the moment.

Frames are taken from the hook (first 2 s), the payoff/peak, the highest-signal moments
inside the clip, and the ending (loop point). Times are labelled in clip time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates import Candidate
from .cache import Workspace, fingerprint
from .config import Config
from .media import MediaError, extract_frame
from .signals import Signals

KEYFRAMES_VERSION = 1


@dataclass
class Keyframe:
    clip_t: float     # seconds from the start of the clip (pre-roll excluded)
    source_t: float
    label: str
    path: Path


def keyframe_times(c: Candidate, sig: Signals, n: int) -> list[tuple[float, str]]:
    dur = c.duration
    picks: list[tuple[float, str]] = [(0.15, "hook: first frame"), (min(1.0, dur), "hook: 1.0s"),
                                      (min(1.9, dur), "hook: 1.9s")]
    picks.append((c.to_content(c.peak), "peak/payoff"))
    picks += [(max(0.0, dur - 1.5), "ending"), (max(0.0, dur - 0.25), "last frame (loop point)")]
    # Fill the rest with the highest-interest moments not already covered.
    clip_ts = np.arange(0.0, dur, 0.5)
    vals = [(sig.max_in("interest", c.to_source(t), c.to_source(t) + 0.5), t) for t in clip_ts]
    for _, t in sorted(vals, reverse=True):
        if len(picks) >= n:
            break
        if all(abs(t - p) > 1.0 for p, _ in picks):
            picks.append((float(t), "high-signal moment"))
    picks = sorted({round(min(max(t, 0.0), max(dur - 0.05, 0.0)), 2): label for t, label in picks}.items())
    if len(picks) > n:  # keep hook + peak + ending, thin the middle
        must = [p for p in picks if not p[1].startswith("high-signal")]
        rest = [p for p in picks if p[1].startswith("high-signal")]
        picks = sorted((must + rest)[:n])
    return picks


def extract_keyframes(ws: Workspace, media: Path, cands: list[Candidate], sig: Signals, cfg: Config,
                      candidates_fp: str) -> dict[str, list[Keyframe]]:
    jc = cfg.judging
    fp = fingerprint("keyframes", KEYFRAMES_VERSION, candidates_fp, jc.keyframes_per_candidate, jc.keyframe_width)
    root = ws.path("keyframes", fp)
    jobs: list[tuple[float, Path]] = []
    result: dict[str, list[Keyframe]] = {}
    for c in cands:
        frames = []
        for clip_t, label in keyframe_times(c, sig, jc.keyframes_per_candidate):
            src_t = c.to_source(clip_t)
            path = root / c.id / f"{clip_t:06.2f}.jpg"
            frames.append(Keyframe(clip_t=clip_t, source_t=src_t, label=label, path=path))
            if not path.exists():
                jobs.append((src_t, path))
        result[c.id] = frames
    for _, p in jobs:
        p.parent.mkdir(parents=True, exist_ok=True)

    def grab(job: tuple[float, Path]) -> None:
        try:
            extract_frame(media, job[0], job[1], jc.keyframe_width)
        except MediaError:
            pass  # e.g. a seek past the last frame; the prompt simply omits a missing frame

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(grab, jobs))
    return {cid: [k for k in frames if k.path.exists()] for cid, frames in result.items()}
