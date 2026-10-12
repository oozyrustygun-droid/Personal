"""Stage 2: speech to words with word-level timestamps (plus optional speaker labels).

Backends:
  mlx            mlx-whisper on Apple Silicon GPU. This is the default on M-series Macs, and
                 large-v3 runs roughly 10x realtime there.
  faster-whisper CTranslate2 build, the default elsewhere (CUDA if present, else CPU int8).
  pocketsphinx   Offline, test-only quality. Used when Whisper weights can't be downloaded.
"""

from __future__ import annotations

import importlib.util
import io
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cache import Workspace, fingerprint, read_json, write_json
from .config import Config

TRANSCRIBE_VERSION = 2

MLX_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
}


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float = 1.0
    speaker: str | None = None


@dataclass
class Transcript:
    words: list[Word]
    language: str = "en"
    backend: str = ""
    model: str = ""
    segments: list[dict] = field(default_factory=list)  # [{"start","end","text"}] as the ASR returned

    def to_json(self) -> dict:
        return {"language": self.language, "backend": self.backend, "model": self.model,
                "segments": self.segments, "words": [asdict(w) for w in self.words]}

    @classmethod
    def from_json(cls, d: dict) -> "Transcript":
        return cls(words=[Word(**w) for w in d["words"]], language=d.get("language", "en"),
                   backend=d.get("backend", ""), model=d.get("model", ""), segments=d.get("segments", []))

    def words_between(self, start: float, end: float) -> list[Word]:
        """Words whose midpoint falls inside [start, end)."""
        return [w for w in self.words if start <= (w.start + w.end) / 2 < end]

    def text_between(self, start: float, end: float) -> str:
        return " ".join(w.text for w in self.words_between(start, end))


def pick_backend(cfg: Config) -> str:
    b = cfg.transcribe.backend
    if b != "auto":
        return b
    if platform.system() == "Darwin" and platform.machine() == "arm64" and importlib.util.find_spec("mlx_whisper"):
        return "mlx"
    if importlib.util.find_spec("faster_whisper"):
        return "faster-whisper"
    if importlib.util.find_spec("pocketsphinx"):
        return "pocketsphinx"
    raise RuntimeError("no transcription backend installed (uv sync)")


def _clean_words(raw: list[Word]) -> list[Word]:
    """Normalise ASR output: strip whitespace, drop empties, enforce monotonic non-overlapping times."""
    out: list[Word] = []
    for w in raw:
        text = w.text.strip()
        if not text:
            continue
        start, end = float(w.start), float(w.end)
        if out and start < out[-1].end:
            # Overlap: split the difference so no instant belongs to two words.
            mid = (start + out[-1].end) / 2
            out[-1].end = max(out[-1].start, min(out[-1].end, mid))
            start = max(start, out[-1].end)
        end = max(end, start + 0.02)
        out.append(Word(start=round(start, 3), end=round(end, 3), text=text, prob=float(w.prob), speaker=w.speaker))
    return out


def _faster_whisper(audio: Path, cfg: Config, console) -> Transcript:
    from faster_whisper import WhisperModel

    tc = cfg.transcribe
    device = "auto"
    compute = tc.compute_type
    if compute == "auto":
        compute = "int8" if platform.system() == "Darwin" else "default"
    model = WhisperModel(tc.model, device=device, compute_type=compute,
                         cpu_threads=max(1, (os.cpu_count() or 4) - 1))
    segments, info = model.transcribe(
        str(audio), language=tc.language or None, beam_size=tc.beam_size, word_timestamps=True,
        vad_filter=tc.vad_filter, condition_on_previous_text=False,
    )
    words: list[Word] = []
    segs: list[dict] = []
    duration = getattr(info, "duration", 0) or 0
    last_report = 0.0
    for seg in segments:
        segs.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        for w in seg.words or []:
            words.append(Word(start=w.start, end=w.end, text=w.word, prob=w.probability))
        if console and duration and seg.end - last_report > 120:
            last_report = seg.end
            console.print(f"  [dim]… {seg.end / 60:.1f}/{duration / 60:.1f} min[/dim]")
    return Transcript(words=_clean_words(words), language=info.language, backend="faster-whisper",
                      model=tc.model, segments=segs)


def _mlx(audio: Path, cfg: Config, console) -> Transcript:
    import mlx_whisper  # type: ignore

    tc = cfg.transcribe
    repo = MLX_REPOS.get(tc.model, tc.model)
    result = mlx_whisper.transcribe(
        str(audio), path_or_hf_repo=repo, word_timestamps=True, language=tc.language or None,
        condition_on_previous_text=False, verbose=None,
    )
    words: list[Word] = []
    segs: list[dict] = []
    for seg in result.get("segments", []):
        segs.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
        for w in seg.get("words", []):
            words.append(Word(start=w["start"], end=w["end"], text=w["word"], prob=w.get("probability", 1.0)))
    return Transcript(words=_clean_words(words), language=result.get("language", tc.language or "en"),
                      backend="mlx", model=tc.model, segments=segs)


_PS_VARIANT = re.compile(r"\(\d+\)$")


def _pocketsphinx(audio: Path, cfg: Config, console) -> Transcript:
    from pocketsphinx import Decoder, Segmenter

    from .media import load_wav_mono

    samples, sr = load_wav_mono(audio)
    pcm = (samples * 32767).astype("<i2").tobytes()
    dec = Decoder(samprate=sr, loglevel="FATAL")
    frate = dec.config["frate"]
    words: list[Word] = []
    segs: list[dict] = []
    for speech in Segmenter(sample_rate=sr).segment(io.BytesIO(pcm)):
        dec.start_utt()
        dec.process_raw(speech.pcm, full_utt=True)
        dec.end_utt()
        seg_words = []
        for s in dec.seg():
            token = _PS_VARIANT.sub("", s.word)
            if token.startswith(("<", "[", "+")):
                continue
            seg_words.append(Word(start=speech.start_time + s.start_frame / frate,
                                  end=speech.start_time + (s.end_frame + 1) / frate, text=token))
        if seg_words:
            # Pocketsphinx emits no punctuation. Close each VAD segment as a sentence.
            seg_words[-1].text += "."
            segs.append({"start": seg_words[0].start, "end": seg_words[-1].end,
                         "text": " ".join(w.text for w in seg_words)})
            words.extend(seg_words)
    return Transcript(words=_clean_words(words), language="en", backend="pocketsphinx", model="en-us", segments=segs)


def _diarize(audio: Path, transcript: Transcript, cfg: Config, console) -> None:
    """Attach speaker labels with pyannote (optional; silently skipped when unavailable)."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not importlib.util.find_spec("pyannote") or not token:
        if console:
            console.print("  [yellow]diarization skipped (needs `uv sync --extra diarize` and HF_TOKEN)[/yellow]")
        return
    from pyannote.audio import Pipeline  # type: ignore

    pipeline = Pipeline.from_pretrained(cfg.diarization.model, token=token)
    diar = pipeline(str(audio))
    turns = [(t.start, t.end, spk) for t, _, spk in diar.itertracks(yield_label=True)]
    for w in transcript.words:
        best, best_ov = None, 0.0
        for s, e, spk in turns:
            ov = min(e, w.end) - max(s, w.start)
            if ov > best_ov:
                best, best_ov = spk, ov
        w.speaker = best


def transcribe(ws: Workspace, cfg: Config, *, console=None, force: bool = False) -> tuple[Transcript, str]:
    backend = pick_backend(cfg)
    fp = fingerprint("transcribe", TRANSCRIBE_VERSION, backend, cfg.section_hash("transcribe", "diarization"),
                     ws.stage_fingerprint("ingest"))
    out = ws.path("transcript.json")
    if not force and ws.is_done("transcribe", fp, [out]):
        return Transcript.from_json(read_json(out)), fp
    audio = ws.path("audio.wav")
    if console:
        console.print(f"  [dim]backend={backend} model={cfg.transcribe.model}[/dim]")
    if backend == "mlx":
        t = _mlx(audio, cfg, console)
    elif backend == "faster-whisper":
        t = _faster_whisper(audio, cfg, console)
    elif backend == "pocketsphinx":
        t = _pocketsphinx(audio, cfg, console)
    else:  # pragma: no cover
        raise ValueError(backend)
    if cfg.diarization.enabled:
        _diarize(audio, t, cfg, console)
    write_json(out, t.to_json())
    ws.mark_done("transcribe", fp, words=len(t.words))
    return t, fp
