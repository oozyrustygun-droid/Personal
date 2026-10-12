"""Face-tracked, center-weighted horizontal crop path for 16:9 → 9:16 exports.

Faces are detected (MediaPipe BlazeFace) on square tiles across each sampled frame. Tiling
matters: the detector expects a roughly square input, and a webcam face in a wide frame is
otherwise too small to find. The raw path is then:
  * gap-filled: hold the last face briefly, then ease back toward the frame center;
  * dead-zoned: the virtual camera moves only when the face leaves a central band;
  * smoothed: exponential smoothing, so the crop pans like an operator instead of jittering.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .config import Config
from .media import read_frames

HOLD_S = 1.5        # keep the last known face position this long when detection drops out
RETURN_S = 2.0      # then ease back to the center over this long


def detect_faces(frames: np.ndarray, cfg: Config) -> list[tuple[float, float] | None]:
    """Per frame: (center_x_norm, size_norm) of the most prominent face, or None."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    from .models import ensure_model

    opts = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model("blaze_face_short_range.tflite", cfg))),
        min_detection_confidence=0.45)
    out: list[tuple[float, float] | None] = []
    prev_x: float | None = None
    with vision.FaceDetector.create_from_options(opts) as det:
        for f in frames:
            H, W = f.shape[:2]
            tiles = sorted({0, max(0, (W - H) // 2), max(0, W - H)}) if W > H else [0]
            size = min(H, W)
            best, best_score = None, 0.0
            for x0 in tiles:
                tile = np.ascontiguousarray(f[:, x0: x0 + size])
                for d in det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile)).detections:
                    b = d.bounding_box
                    cx = (x0 + b.origin_x + b.width / 2) / W
                    area = (b.width * b.height) / (W * H)
                    score = d.categories[0].score * math.sqrt(area)
                    if prev_x is not None:  # prefer the face we were already following
                        score *= 1.0 + 0.5 * max(0.0, 1.0 - abs(cx - prev_x) / 0.2)
                    if score > best_score:
                        best, best_score = (cx, b.width / W), score
            out.append(best)
            if best:
                prev_x = best[0]
    return out


def smooth_path(times: list[float], xs: list[float | None], crop_frac: float, cfg: Config) -> list[float]:
    """Turn raw face centers (normalized 0..1, or None) into a steady crop-center path."""
    if not times:
        return []
    filled: list[float] = []
    last_x, last_t = None, None
    for t, x in zip(times, xs):
        if x is not None:
            last_x, last_t = x, t
            filled.append(x)
        elif last_x is None:
            filled.append(0.5)
        else:
            gap = t - last_t
            w = 0.0 if gap <= HOLD_S else min(1.0, (gap - HOLD_S) / RETURN_S)
            filled.append(last_x * (1 - w) + 0.5 * w)
    dz = cfg.export.face_deadzone * crop_frac
    tau = max(cfg.export.face_smoothing_s, 1e-3)
    cam = filled[0]
    path = []
    prev_t = times[0]
    for t, target in zip(times, filled):
        dt = max(t - prev_t, 0.0)
        prev_t = t
        if abs(target - cam) > dz:  # leave the dead zone: follow, keeping the face at its edge
            goal = target - math.copysign(dz, target - cam)
            alpha = 1 - math.exp(-dt / tau) if dt > 0 else 0.0
            cam = cam + alpha * (goal - cam)
        lo, hi = crop_frac / 2, 1 - crop_frac / 2
        cam = min(max(cam, lo), hi)
        path.append(cam)
    return path


def crop_path(src: Path, segments: list[tuple[float, float]], src_w: int, src_h: int, crop_w: int,
              cfg: Config) -> list[tuple[float, int]]:
    """[(output_time, crop_x_pixels)] for the concatenated segments."""
    crop_frac = crop_w / src_w
    fps = cfg.export.face_sample_fps
    out_times: list[float] = []
    raw: list[float | None] = []
    offset = 0.0
    for s, e in segments:
        if cfg.export.crop_mode == "face":
            frames, times = read_frames(src, s, e, fps, width=min(960, src_w))
            faces = detect_faces(frames, cfg)
        else:
            times = list(np.arange(s, e, 1 / fps))
            faces = [None] * len(times)
        for t, f in zip(times, faces):
            out_times.append(offset + (t - s))
            raw.append(f[0] if f else None)
        offset += e - s
    path = smooth_path(out_times, raw, crop_frac, cfg)
    return [(t, int(round(min(max(c * src_w - crop_w / 2, 0), src_w - crop_w)))) for t, c in zip(out_times, path)]
