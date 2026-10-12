"""Typed configuration loaded from config.toml.

Defaults here mirror the repository's config.toml; the TOML file overrides them. Unknown
keys raise an error so a typo can never silently loosen the bar.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_GUIDE_PATH = REPO_ROOT / "docs" / "joe-style-guide.md"

RUBRIC_CRITERIA = (
    "hook_strength",
    "payoff",
    "emotional_intensity",
    "shareability",
    "comment_bait",
    "rewatch_loop",
    "joe_signature",
    "visual_strength",
)

GATE_NAMES = (
    "hook_in_first_2s",
    "standalone",
    "clear_payoff",
    "no_dead_air",
    "monetization_safe",
)


DEFAULT_FLOORS = {"hook_strength": 8.0, "payoff": 8.0}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _finite(obj, *names: str) -> None:
    """NaN/inf thresholds would make every comparison False and silently disable a check."""
    for n in names:
        v = getattr(obj, n)
        if not math.isfinite(v):
            raise ValueError(f"{type(obj).__name__}.{n} must be a finite number, got {v!r}")


class GeneralCfg(_Section):
    cache_dir: str = "~/.cache/clip-scanner"
    output_dir: str = "./clip-scanner-output"
    max_clips: int = Field(5, ge=1)


class IngestCfg(_Section):
    format: str = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b"
    cookies_from_browser: str = ""


class TranscribeCfg(_Section):
    backend: Literal["auto", "mlx", "faster-whisper", "pocketsphinx"] = "auto"
    model: str = "large-v3"
    language: str = "en"
    beam_size: int = 5
    vad_filter: bool = True
    compute_type: str = "auto"


class DiarizationCfg(_Section):
    enabled: bool = False
    model: str = "pyannote/speaker-diarization-3.1"


class SignalsCfg(_Section):
    silence_db: float = -42.0
    audio_events: Literal["yamnet", "none"] = "yamnet"
    scene_threshold: float = 27.0
    proxy_height: int = 360
    proxy_fps: int = 15
    hwaccel: Literal["auto", "none"] = "auto"


class CandidatesCfg(_Section):
    min_candidates: int = 15
    max_candidates: int = 40
    min_len_s: float = 12.0
    max_len_s: float = 60.0
    target_lens_s: list[float] = [14.0, 20.0, 28.0, 38.0, 52.0]
    peak_min_separation_s: float = 8.0
    llm_proposer: bool = True
    proposer_model: str = "claude-haiku-5-5"
    proposer_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    proposer_chunk_s: float = 600.0
    tighten_dead_air: bool = True
    tightened_gap_s: float = 0.35
    pause_break_s: float = 0.7
    snap_search_s: float = 6.0
    end_tail_s: float = 0.35
    nms_iou: float = 0.5


class GatesCfg(_Section):
    min_len_s: float = 12.0
    max_len_s: float = 60.0
    hook_window_s: float = 2.0
    max_dead_air_s: float = 1.5
    music_prob: float = 0.5
    music_bed_max_fraction: float = 0.5
    require_music_check: bool = True  # if the audio-event model is unavailable, the music gate fails

    @model_validator(mode="after")
    def _check(self) -> "GatesCfg":
        _finite(self, "min_len_s", "max_len_s", "hook_window_s", "max_dead_air_s", "music_prob",
                "music_bed_max_fraction")
        if not 0 < self.min_len_s < self.max_len_s:
            raise ValueError("gates: need 0 < min_len_s < max_len_s")
        for name in ("music_prob", "music_bed_max_fraction"):  # "50" meaning 50% would disable the gate
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"gates.{name} must be between 0 and 1 (e.g. 0.5, not 50)")
        if self.hook_window_s <= 0 or self.max_dead_air_s <= 0:
            raise ValueError("gates: hook_window_s and max_dead_air_s must be > 0")
        return self


class ClaudeCodeCfg(_Section):
    path: str = ""                    # blank: find `claude` on PATH
    timeout_s: float = Field(900.0, gt=0)
    allow_extra_usage: bool = False   # False: stop judging as soon as Claude Code draws on paid extra usage
    allow_api_billing: bool = False   # False: refuse to run if Claude Code would bill an API key


class JudgingCfg(_Section):
    # claude-code: every call runs through the Claude Code CLI on your Claude subscription (no API credits).
    # api: the Claude API with ANTHROPIC_API_KEY, billed per token.
    backend: Literal["claude-code", "api"] = "claude-code"
    claude_code: ClaudeCodeCfg = ClaudeCodeCfg()
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    judges: list[str] = ["jaded_scroller", "joe_superfan", "shorts_editor"]
    overall_min: float = 8.5
    rubric_mean_min: float = 7.5
    consistency_allowance: float = 1.0
    gate_policy: Literal["any", "majority"] = "any"
    short_circuit: bool = True
    max_concurrency: int = Field(4, ge=1)
    keyframes_per_candidate: int = Field(7, ge=1, le=16)
    keyframe_width: int = 768
    context_window_s: float = 20.0
    max_output_tokens: int = 16000
    confirm_above_usd: float = 5.0
    server_fallbacks: bool = True   # api backend only; packaging/diagnosis/proposer; never for verdicts
    cache_ttl: Literal["5m", "1h"] = "5m"
    criterion_floors: dict[str, float] = dict(DEFAULT_FLOORS)
    weights: dict[str, float] = {
        "hook_strength": 1.5,
        "payoff": 1.5,
        "emotional_intensity": 1.0,
        "shareability": 1.25,
        "comment_bait": 0.75,
        "rewatch_loop": 0.75,
        "joe_signature": 1.0,
        "visual_strength": 0.75,
    }

    @model_validator(mode="before")
    @classmethod
    def _merge_floors(cls, data):
        # Setting one floor in config.toml must not silently drop the other default floors.
        # To remove a floor on purpose, set it to 0.
        if isinstance(data, dict) and isinstance(data.get("criterion_floors"), dict):
            data = {**data, "criterion_floors": {**DEFAULT_FLOORS, **data["criterion_floors"]}}
        return data

    @model_validator(mode="after")
    def _check(self) -> "JudgingCfg":
        _finite(self, "overall_min", "rubric_mean_min", "consistency_allowance")
        for name in ("overall_min", "rubric_mean_min"):
            if not 0 <= getattr(self, name) <= 10:
                raise ValueError(f"judging.{name} must be within 0-10")
        if self.consistency_allowance < 0:
            raise ValueError("judging.consistency_allowance must be >= 0")
        for k, v in {**self.criterion_floors, **self.weights}.items():
            if not math.isfinite(v):
                raise ValueError(f"judging: {k} must be a finite number")
        if any(not 0 <= v <= 10 for v in self.criterion_floors.values()):
            raise ValueError("judging.criterion_floors must be within 0-10")
        unknown = (set(self.weights) | set(self.criterion_floors)) - set(RUBRIC_CRITERIA)
        if unknown:
            raise ValueError(f"unknown rubric criteria in config: {sorted(unknown)}")
        missing = set(RUBRIC_CRITERIA) - set(self.weights)
        if missing:
            raise ValueError(f"judging.weights is missing criteria: {sorted(missing)}")
        if any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("judging.weights must be non-negative and not all zero")
        if not self.judges:
            raise ValueError("judging.judges must name at least one judge")
        if len(set(self.judges)) != len(self.judges):
            raise ValueError("judging.judges must not repeat a persona")
        return self


class DevilsAdvocateCfg(_Section):
    enabled: bool = True
    min_survival_confidence: float = 7.0

    @model_validator(mode="after")
    def _check(self) -> "DevilsAdvocateCfg":
        _finite(self, "min_survival_confidence")
        if not 0 <= self.min_survival_confidence <= 10:
            raise ValueError("devils_advocate.min_survival_confidence must be within 0-10")
        return self


class ExportCfg(_Section):
    width: int = 1080
    height: int = 1920
    preroll_s: float = 0.3
    crop_mode: Literal["face", "center"] = "face"
    face_sample_fps: float = 6.0
    face_smoothing_s: float = 0.6
    face_deadzone: float = 0.08
    crf: int = 18
    preset: str = "medium"
    audio_bitrate: str = "192k"
    raw_copy: bool = True
    raw_handles_s: float = 3.0


class CaptionsCfg(_Section):
    enabled: bool = True
    font: str = "Arial Black"
    font_size: int = 92
    bold: bool = True
    uppercase: bool = True
    primary_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline: float = 7.0
    shadow: float = 2.0
    margin_v: int = 480
    words_per_caption: int = Field(1, ge=1, le=4)
    max_word_hold_s: float = 0.9


class CalibrationCfg(_Section):
    manifest: str = "calibration/manifest.toml"
    hit_min_views: int = 1_000_000
    flop_max_views: int = 100_000


class Price(_Section):
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


DEFAULT_PRICING = {
    "claude-opus-5-5": Price(input=4.0, output=20.0, cache_write_5m=5.0, cache_write_1h=8.0, cache_read=0.20),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_write_5m=0.125, cache_write_1h=0.20, cache_read=0.01),
}


class Config(_Section):
    general: GeneralCfg = GeneralCfg()
    ingest: IngestCfg = IngestCfg()
    transcribe: TranscribeCfg = TranscribeCfg()
    diarization: DiarizationCfg = DiarizationCfg()
    signals: SignalsCfg = SignalsCfg()
    candidates: CandidatesCfg = CandidatesCfg()
    gates: GatesCfg = GatesCfg()
    judging: JudgingCfg = JudgingCfg()
    devils_advocate: DevilsAdvocateCfg = DevilsAdvocateCfg()
    export: ExportCfg = ExportCfg()
    captions: CaptionsCfg = CaptionsCfg()
    calibration: CalibrationCfg = CalibrationCfg()
    pricing: dict[str, Price] = dict(DEFAULT_PRICING)

    # Not from TOML: where the config was loaded from (for messages / calibrate --write-config).
    source_path: str | None = None

    @property
    def cache_dir(self) -> Path:
        return Path(os.path.expanduser(self.general.cache_dir))

    @property
    def output_dir(self) -> Path:
        return Path(os.path.expanduser(self.general.output_dir))

    def price(self, model: str) -> Price:
        if model in self.pricing:
            return self.pricing[model]
        # Unknown model: price pessimistically at the most expensive configured rate.
        return max(self.pricing.values(), key=lambda p: p.output)

    def section_hash(self, *names: str) -> str:
        data = {n: getattr(self, n).model_dump() for n in names}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def find_config_path(explicit: str | None = None) -> Path | None:
    # An explicitly requested config that doesn't exist is an error, never a silent fallback to the
    # default bar (a typo in the path to a stricter config must not loosen anything).
    for named, label in ((explicit, "--config"), (os.environ.get("CLIP_SCANNER_CONFIG"), "$CLIP_SCANNER_CONFIG")):
        if named:
            p = Path(named).expanduser()
            if not p.is_file():
                raise FileNotFoundError(f"{label} file not found: {named}")
            return p
    for p in (Path.cwd() / "config.toml", REPO_ROOT / "config.toml"):
        if p.is_file():
            return p
    return None


def load_config(explicit: str | None = None, overrides: dict | None = None) -> Config:
    path = find_config_path(explicit)
    data: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    if overrides:
        data = deep_merge(data, overrides)
    if "pricing" in data:
        data["pricing"] = {**{k: v.model_dump() for k, v in DEFAULT_PRICING.items()}, **data["pricing"]}
    cfg = Config.model_validate(data)
    cfg.source_path = str(path) if path else None
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_style_guide(path: Path = STYLE_GUIDE_PATH) -> str:
    """The judge-facing part of the style guide (everything before the end-of-prompt marker)."""
    text = path.read_text(encoding="utf-8")
    marker = "<!-- end-of-prompt -->"
    return text.split(marker, 1)[0].strip()
