"""Small on-device models (downloaded once into the cache)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from .config import Config

MODEL_URLS = {
    "yamnet.tflite": "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/"
                                     "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def ensure_model(name: str, cfg: Config) -> Path:
    path = cfg.cache_dir / "models" / name
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URLS[name], timeout=60) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.replace(path)
    return path
