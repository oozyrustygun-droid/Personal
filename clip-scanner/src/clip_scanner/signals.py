"""Stage 3: per-moment features that point at clip-worthy regions.

Everything is sampled on a 100 ms grid ("hops"):
  rms_db        loudness (dBFS)
  loud_spike    loudness above the local 30 s median (0..1.5)
  laughter, scream, crowd, gasp, music, speech, chewing   YAMNet audio-event scores (0..1)
  speech_rate   words per second (2 s window)
  reaction      reaction words, Joe catchphrases, exclamations in the transcript (0..1)
  visual_change PySceneDetect content score (robust z, 0..4)
  word_active   1 while a word is being spoken
  interest      the weighted, smoothed combination used to find candidate peaks
Plus scene-cut timestamps and silence gaps.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cache import Workspace, fingerprint, read_json, write_json
from .config import Config
from .media import load_wav_mono, make_proxy
from .transcribe import Transcript

SIGNALS_VERSION = 5
HOP = 0.1

EVENT_GROUPS: dict[str, list[str]] = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "scream": ["Screaming", "Shout", "Yell", "Bellow", "Whoop", "Battle cry", "Children shouting"],
    "crowd": ["Cheering", "Applause", "Crowd"],
    "gasp": ["Gasp"],
    "music": ["Music"],
    "speech": ["Speech"],
    "chewing": ["Chewing, mastication", "Biting"],
}

REACTION_PATTERNS = [
    # Joe catchphrases and recurring bits (see docs/joe-style-guide.md)
    r"type in chat", r"\bl chatter", r"admit you were joking", r"type in shots", r"what('?s| is) that",
    # generic big-reaction language
    r"oh my god", r"\bomg\b", r"what the", r"\bwtf\b", r"no way", r"\bholy\b", r"\bbro\b", r"\bdude\b",
    r"\bnah\b", r"\bstop\b", r"\bjesus\b", r"disgusting", r"\binsane\b", r"\bcrazy\b", r"\bworst\b",
    r"\bnever\b", r"\bhell\b", r"\bfuck", r"\bshit\b", r"\bdamn\b", r"\bha(ha)+\b", r"\blmao\b", r"\bscream",
]
_REACTION_RE = re.compile("|".join(f"(?:{p})" for p in REACTION_PATTERNS))

INTEREST_WEIGHTS = {
    "loud_spike": 1.0, "laughter": 1.6, "scream": 1.2, "crowd": 0.8, "gasp": 0.5,
    "reaction": 0.7, "speech_rate_z": 0.25, "visual_z": 0.35, "cut_near": 0.3,
}


@dataclass
class Signals:
    duration: float
    series: dict[str, np.ndarray]
    scene_cuts: np.ndarray
    silences: np.ndarray  # (N, 2) start/end seconds
    events_available: bool = True
    notes: list[str] = field(default_factory=list)
    hop: float = HOP

    # ---- access -------------------------------------------------------------------------
    def idx(self, t: float) -> int:
        n = len(self.series["rms_db"])
        return int(min(max(round(t / self.hop), 0), n))

    def window(self, name: str, start: float, end: float) -> np.ndarray:
        return self.series[name][self.idx(start): max(self.idx(end), self.idx(start) + 1)]

    def max_in(self, name: str, start: float, end: float) -> float:
        w = self.window(name, start, end)
        return float(w.max()) if w.size else 0.0

    def mean_in(self, name: str, start: float, end: float) -> float:
        w = self.window(name, start, end)
        return float(w.mean()) if w.size else 0.0

    def frac_above(self, name: str, thr: float, start: float, end: float) -> float:
        w = self.window(name, start, end)
        return float((w >= thr).mean()) if w.size else 0.0

    def per_second(self, name: str, how: str = "max") -> np.ndarray:
        x = self.series[name]
        per = int(round(1 / self.hop))
        n = int(np.ceil(len(x) / per))
        padded = np.pad(x, (0, n * per - len(x)), constant_values=x[-1] if len(x) else 0)
        grid = padded.reshape(n, per)
        return grid.max(axis=1) if how == "max" else grid.mean(axis=1)

    def cuts_between(self, start: float, end: float) -> list[float]:
        return [float(c) for c in self.scene_cuts if start <= c <= end]

    def reaction_mask(self) -> np.ndarray:
        """Hops where something is happening even without words (laugh, scream, crowd, spike)."""
        s = self.series
        return ((s["laughter"] >= 0.3) | (s["scream"] >= 0.3) | (s["crowd"] >= 0.3) | (s["gasp"] >= 0.3)
                | (s["loud_spike"] >= 0.5))

    def dead_mask(self) -> np.ndarray:
        """Hops of dead air: nobody talking and no reaction audio."""
        return (self.series["word_active"] < 0.5) & ~self.reaction_mask()

    # ---- persistence --------------------------------------------------------------------
    def save(self, ws: Workspace) -> None:
        np.savez_compressed(ws.path("signals.npz"), scene_cuts=self.scene_cuts, silences=self.silences,
                            **{f"s_{k}": v for k, v in self.series.items()})
        write_json(ws.path("signals.json"), {"duration": self.duration, "hop": self.hop,
                                             "events_available": self.events_available, "notes": self.notes,
                                             "scene_cuts": len(self.scene_cuts), "silences": len(self.silences)})

    @classmethod
    def load(cls, ws: Workspace) -> "Signals":
        meta = read_json(ws.path("signals.json"))
        data = np.load(ws.path("signals.npz"))
        series = {k[2:]: data[k] for k in data.files if k.startswith("s_")}
        return cls(duration=meta["duration"], series=series, scene_cuts=data["scene_cuts"],
                   silences=data["silences"], events_available=meta["events_available"], notes=meta["notes"],
                   hop=meta["hop"])


# ---- feature extraction --------------------------------------------------------------------

def loudness_db(audio: np.ndarray, sr: int, hop: float = HOP) -> np.ndarray:
    n = int(round(sr * hop))
    frames = len(audio) // n
    if frames == 0:
        return np.full(1, -90.0, dtype=np.float32)
    x = audio[: frames * n].reshape(frames, n).astype(np.float64)
    rms = np.sqrt((x ** 2).mean(axis=1) + 1e-12)
    return (20 * np.log10(rms + 1e-9)).astype(np.float32)


def speech_baseline(rms_db: np.ndarray, silence_db: float, window_s: int = 61, hop: float = HOP) -> np.ndarray:
    """Local typical loudness of non-silent audio (rolling median over ~1 min, silence ignored).

    Silence must be excluded, otherwise quiet gaps drag the baseline down and normal talking
    reads as a "spike".
    """
    per = int(round(1 / hop))
    n = len(rms_db)
    m = max(1, n // per)
    per_s = rms_db[: m * per].reshape(m, per).mean(axis=1) if n >= per else rms_db.copy()
    per_s = per_s.astype(np.float64)
    per_s[per_s < silence_db] = np.nan
    fallback = float(np.nanmedian(per_s)) if np.isfinite(per_s).any() else silence_db
    half = window_s // 2
    padded = np.pad(per_s, (half, half), mode="constant", constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(padded, window_s)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(view, axis=1)[: len(per_s)]
    med = np.where(np.isfinite(med), med, fallback)
    base = np.repeat(med, per)
    return np.pad(base, (0, max(0, n - len(base))), mode="edge")[:n].astype(np.float32)


def robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826 + 1e-6
    return np.clip((x - med) / mad, 0, 4).astype(np.float32)


def smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or x.size == 0:
        return x
    k = np.ones(width) / width
    return np.convolve(x, k, mode="same").astype(np.float32)


def silence_runs(rms_db: np.ndarray, thr_db: float, min_len_s: float = 0.3, hop: float = HOP) -> np.ndarray:
    quiet = rms_db < thr_db
    runs: list[tuple[float, float]] = []
    start = None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop >= min_len_s:
                runs.append((start * hop, i * hop))
            start = None
    return np.array(runs, dtype=np.float32).reshape(-1, 2)


def audio_events(audio: np.ndarray, sr: int, n_hops: int, cfg: Config, hop: float = HOP) -> dict[str, np.ndarray]:
    """YAMNet scores per event group, held over each ~0.975 s window onto the hop grid."""
    from mediapipe.tasks.python import audio as mp_audio
    from mediapipe.tasks.python.components.containers import AudioData
    from mediapipe.tasks.python.core.base_options import BaseOptions

    from .models import ensure_model

    allow = sorted({c for names in EVENT_GROUPS.values() for c in names})
    opts = mp_audio.AudioClassifierOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model("yamnet.tflite", cfg))),
        max_results=len(allow), category_allowlist=allow)
    group_of = {c: g for g, names in EVENT_GROUPS.items() for c in names}
    out = {g: np.zeros(n_hops, dtype=np.float32) for g in EVENT_GROUPS}
    chunk = int(120 * sr)
    with mp_audio.AudioClassifier.create_from_options(opts) as clf:
        for c0 in range(0, len(audio), chunk):
            piece = audio[c0: c0 + chunk]
            if len(piece) < int(0.5 * sr):
                continue
            results = clf.classify(AudioData.create_from_array(piece.astype(np.float32), sr))
            for r in results:
                t0 = c0 / sr + r.timestamp_ms / 1000.0
                h0, h1 = int(t0 / hop), min(n_hops, int((t0 + 0.975) / hop) + 1)
                if h0 >= n_hops:
                    continue
                for cat in r.classifications[0].categories:
                    g = group_of.get(cat.category_name)
                    if g:
                        seg = out[g][h0:h1]
                        np.maximum(seg, cat.score, out=seg)
    return out


def reaction_series(transcript: Transcript, n_hops: int, hop: float = HOP) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(reaction score, words-per-second, word-active mask) on the hop grid."""
    reaction = np.zeros(n_hops, dtype=np.float32)
    active = np.zeros(n_hops, dtype=np.float32)
    rate = np.zeros(n_hops, dtype=np.float32)
    words = transcript.words
    if not words:
        return reaction, rate, active

    # Map regex matches over the lowercase running text back onto word indices.
    pieces, offsets, pos = [], [], 0
    for w in words:
        offsets.append(pos)
        pieces.append(w.text.lower())
        pos += len(w.text) + 1
    text = " ".join(pieces)
    starts = np.array(offsets)

    def mark(i: int, val: float) -> None:
        w = words[i]
        h0, h1 = int(w.start / hop), min(n_hops, int(w.end / hop) + 1)
        seg = reaction[h0:h1]
        np.maximum(seg, val, out=seg)

    for m in _REACTION_RE.finditer(text):
        i0 = int(np.searchsorted(starts, m.start(), side="right") - 1)
        i1 = int(np.searchsorted(starts, m.end() - 1, side="right") - 1)
        for i in range(max(i0, 0), min(i1, len(words) - 1) + 1):
            mark(i, 1.0)
    for i, w in enumerate(words):
        if w.text.endswith("!"):
            mark(i, 0.8)
        elif w.text.endswith("?"):
            mark(i, 0.35)
        elif len(w.text) > 2 and w.text.isupper():
            mark(i, 0.5)
        # Exact word spans (a hop counts if its centre is inside the word), no padding: padding would
        # make real silence measure shorter than it is and let dead air slip past the gate.
        h0 = max(0, math.ceil(w.start / hop - 0.5))
        h1 = min(n_hops, math.floor(w.end / hop - 0.5) + 1)
        if h1 <= h0:  # word shorter than a hop: mark the hop holding its midpoint
            h0 = min(n_hops - 1, int(((w.start + w.end) / 2) / hop))
            h1 = h0 + 1
        active[h0:h1] = 1.0

    mids = np.array([(w.start + w.end) / 2 for w in words])
    centers = (np.arange(n_hops) + 0.5) * hop
    lo = np.searchsorted(mids, centers - 1.0)
    hi = np.searchsorted(mids, centers + 1.0)
    rate[:] = (hi - lo) / 2.0
    return reaction, rate, active


def scene_features(proxy: Path, n_hops: int, cfg: Config, hop: float = HOP) -> tuple[np.ndarray, np.ndarray]:
    from scenedetect import ContentDetector, SceneManager, StatsManager, open_video

    video = open_video(str(proxy))
    sm = SceneManager(stats_manager=StatsManager())
    sm.add_detector(ContentDetector(threshold=cfg.signals.scene_threshold))
    n_frames = sm.detect_scenes(video, show_progress=False)
    fps = float(video.frame_rate)
    cuts = np.array([s.seconds for s, _ in sm.get_scene_list()[1:]], dtype=np.float32)
    vals = np.zeros(n_hops, dtype=np.float32)
    counts = np.zeros(n_hops, dtype=np.float32)
    stats = sm.stats_manager
    for f in range(n_frames):
        m = stats.get_metrics(f, ["content_val"])
        if m and m[0] is not None:
            h = int((f / fps) / hop)
            if h < n_hops:
                vals[h] += float(m[0])
                counts[h] += 1
    vals = np.divide(vals, counts, out=np.zeros_like(vals), where=counts > 0)
    return cuts, vals


def compute_signals(ws: Workspace, transcript: Transcript, duration: float, media: Path, cfg: Config,
                    *, console=None) -> Signals:
    audio, sr = load_wav_mono(ws.path("audio.wav"))
    rms = loudness_db(audio, sr)
    n = len(rms)
    notes: list[str] = []

    base = speech_baseline(rms, cfg.signals.silence_db)
    loud_spike = np.clip((rms - base - 4.0) / 6.0, 0, 1.5).astype(np.float32)
    loud_spike[rms < cfg.signals.silence_db] = 0

    events_ok = cfg.signals.audio_events == "yamnet"
    events = {g: np.zeros(n, dtype=np.float32) for g in EVENT_GROUPS}
    if events_ok:
        try:
            events = audio_events(audio, sr, n, cfg)
        except Exception as e:  # model download / native lib problems should not kill the scan
            events_ok = False
            notes.append(f"audio events unavailable ({type(e).__name__}: {e}); laughter/music detection off")
            if console:
                console.print(f"  [yellow]{notes[-1]}[/yellow]")

    reaction, rate, active = reaction_series(transcript, n)

    proxy = ws.path("proxy.mp4")
    if not proxy.exists():
        make_proxy(media, proxy, cfg.signals.proxy_height, cfg.signals.proxy_fps, cfg.signals.hwaccel)
    cuts, visual = scene_features(proxy, n, cfg)

    cut_near = np.zeros(n, dtype=np.float32)
    for c in cuts:
        h = int(c / HOP)
        cut_near[max(0, h - 10): h + 10] = 1.0

    series = {
        "rms_db": rms, "loud_spike": loud_spike, **events, "reaction": reaction, "speech_rate": rate,
        "speech_rate_z": robust_z(rate), "word_active": active, "visual_change": visual,
        "visual_z": robust_z(visual), "cut_near": cut_near,
    }
    raw = sum(INTEREST_WEIGHTS[k] * series[k] for k in INTEREST_WEIGHTS)
    series["interest"] = smooth(raw.astype(np.float32), 10)
    return Signals(duration=duration, series=series, scene_cuts=cuts,
                   silences=silence_runs(rms, cfg.signals.silence_db), events_available=events_ok, notes=notes)


def signals_stage(ws: Workspace, transcript: Transcript, transcript_fp: str, duration: float, media: Path,
                  cfg: Config, *, console=None, force: bool = False) -> tuple[Signals, str]:
    fp = fingerprint("signals", SIGNALS_VERSION, cfg.section_hash("signals"), transcript_fp)
    if not force and ws.is_done("signals", fp, [ws.path("signals.npz"), ws.path("signals.json")]):
        return Signals.load(ws), fp
    sig = compute_signals(ws, transcript, duration, media, cfg, console=console)
    sig.save(ws)
    ws.mark_done("signals", fp)
    return sig, fp
