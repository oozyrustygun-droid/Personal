"""Keyframes per candidate so the judges can actually see the moment, as it will be posted.

Frames show the 9:16 crop the export will use (same face tracking), not the full 16:9 frame. A clip
must not pass on visuals that the vertical crop cuts away. Frames come from the opening (the hook
window), the most active measured moments, and the ending (the loop point). Labels are neutral and
never claim that a frame shows a payoff.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import Workspace, fingerprint
from .candidates import Candidate
from .config import Config
from .media import MediaError, extract_frame, probe
from .signals import Signals

KEYFRAMES_VERSION = 3


@dataclass
class Keyframe:
    clip_t: float     # seconds from the first posted frame
    source_t: float
    label: str
    path: Path


def keyframe_times(c: Candidate, sig: Signals, n: int) -> list[tuple[float, str]]:
    dur = c.duration
    picks: list[tuple[float, str]] = [(0.15, "first frame"), (min(1.0, dur), "1.0s in"),
                                      (min(1.9, dur), "1.9s in"),
                                      (c.to_content(c.peak), "most active moment (measured)"),
                                      (max(0.0, dur - 1.5), "1.5s before the end"),
                                      (max(0.0, dur - 0.25), "last frame (loop point)")]
    clip_ts = np.arange(0.0, dur, 0.5)
    vals = [(sig.max_in("interest", c.to_source(t), c.to_source(t) + 0.5), t) for t in clip_ts]
    for _, t in sorted(vals, reverse=True):
        if len(picks) >= n:
            break
        if all(abs(t - p) > 1.0 for p, _ in picks):
            picks.append((float(t), "active moment (measured)"))
    picks = sorted({round(min(max(t, 0.0), max(dur - 0.05, 0.0)), 2): label for t, label in picks}.items())
    if len(picks) > n:  # keep the opening, the most active moment and the ending; thin the middle
        must = [p for p in picks if not p[1].startswith("active moment")]
        rest = [p for p in picks if p[1].startswith("active moment")]
        picks = sorted((must + rest)[:n])
    return picks


def extract_keyframes(ws: Workspace, media: Path, cands: list[Candidate], sig: Signals, cfg: Config,
                      candidates_fp: str) -> dict[str, list[Keyframe]]:
    from .export import crop_geometry
    from .facetrack import shared_crop_path, x_at

    jc = cfg.judging
    sw, sh = probe(media).display_size
    crop_w, crop_h, crop_y = crop_geometry(sw, sh, cfg.export.width, cfg.export.height)
    horizontal = crop_w < sw
    jobs: list[tuple[float, Path, str]] = []
    result: dict[str, list[Keyframe]] = {}
    for c in cands:
        # Keyed on the exact segments and crop settings: a different window can never reuse these frames.
        fp = fingerprint("keyframes", KEYFRAMES_VERSION, [list(s) for s in c.segments], jc.keyframes_per_candidate,
                         jc.keyframe_width, cfg.section_hash("export"), sw, sh)
        folder = ws.path("keyframes", f"{c.id}-{fp}")
        times = keyframe_times(c, sig, jc.keyframes_per_candidate)
        frames = [Keyframe(clip_t=t, source_t=c.to_source(t), label=label, path=folder / f"{t:06.2f}.jpg")
                  for t, label in times]
        result[c.id] = frames
        missing = [k for k in frames if not k.path.exists()]
        if not missing:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if horizontal:
            # Identical to the export: same cached path, same "last command at or before t" rule, same fallback.
            path = shared_crop_path(ws.root, media, c.segments, sw, sh, crop_w, cfg)
        for k in missing:
            if horizontal:
                x = x_at(path, k.clip_t, default=(sw - crop_w) // 2)
                vf = f"crop={crop_w}:{crop_h}:{x}:{crop_y},scale=-2:{jc.keyframe_width}"
            else:
                vf = f"crop={crop_w}:{crop_h}:0:{crop_y},scale=-2:{jc.keyframe_width}"
            jobs.append((k.source_t, k.path, vf))

    def grab(job: tuple[float, Path, str]) -> None:
        try:
            extract_frame(media, job[0], job[1], jc.keyframe_width, vf=job[2])
        except MediaError:
            pass  # e.g. a seek past the last frame; the prompt simply omits a missing frame

    with ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(grab, jobs))
    return {cid: [k for k in frames if k.path.exists()] for cid, frames in result.items()}
