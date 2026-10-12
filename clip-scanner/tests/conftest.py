import importlib.util
import os

import pytest


def pick_asr() -> tuple[str, str]:
    """faster-whisper tiny.en when its weights can be loaded, else pocketsphinx (offline). Override with
    CLIP_SCANNER_TEST_ASR=faster-whisper|mlx|pocketsphinx."""
    choice = os.environ.get("CLIP_SCANNER_TEST_ASR")
    if choice:
        return choice, os.environ.get("CLIP_SCANNER_TEST_ASR_MODEL", "tiny.en")
    if importlib.util.find_spec("faster_whisper"):
        try:
            from faster_whisper import WhisperModel

            WhisperModel("tiny.en", device="cpu", compute_type="int8")
            return "faster-whisper", "tiny.en"
        except Exception:
            pass
    if importlib.util.find_spec("pocketsphinx"):
        return "pocketsphinx", "en-us"
    pytest.skip("no usable ASR backend (Whisper weights unavailable and pocketsphinx not installed)")
