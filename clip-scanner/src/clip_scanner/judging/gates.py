"""Hard gates that can be measured without a model.

They run before any API spend, and a failure is final. The LLM judges check the same gates
semantically (hook quality, standalone context and so on); these only catch what the
numbers can prove.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..candidates import Candidate
from ..config import Config
from ..signals import Signals
from ..transcribe import Transcript


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str
    value: float | None = None

    def to_json(self) -> dict:
        return asdict(self)


def clip_measurements(c: Candidate, transcript: Transcript, sig: Signals) -> dict:
    """Facts about the clip as exported (clip time = seconds from clip start, pre-roll excluded)."""
    words = c.words(transcript)
    first_word = c.to_content(words[0].start) if words else None
    # First moment anything happens: a word, or reaction audio / loudness spike.
    react = sig.reaction_mask()
    first_react = None
    acc = 0.0
    for s, e in c.segments:
        i0, i1 = sig.idx(s), sig.idx(e)
        hits = [i for i in range(i0, min(i1, len(react))) if react[i]]
        if hits:
            first_react = acc + (hits[0] * sig.hop - s)
            break
        acc += e - s
    onset_candidates = [t for t in (first_word, first_react) if t is not None]
    onset = min(onset_candidates) if onset_candidates else None

    # Longest dead-air run inside the kept segments (jump cuts split runs by construction).
    dead = sig.dead_mask()
    longest = 0.0
    for s, e in c.segments:
        run = 0
        for i in range(sig.idx(s), min(sig.idx(e), len(dead))):
            run = run + 1 if dead[i] else 0
            longest = max(longest, run * sig.hop)

    music_frac = 0.0
    if sig.events_available:
        total = sum(e - s for s, e in c.segments)
        music_frac = sum(sig.frac_above("music", 0.5, s, e) * (e - s) for s, e in c.segments) / max(total, 1e-6)
    return {
        "duration": round(c.duration, 2),
        "first_word_at": None if first_word is None else round(first_word, 2),
        "first_activity_at": None if onset is None else round(onset, 2),
        "longest_dead_air": round(longest, 2),
        "music_fraction": round(music_frac, 3),
        "word_count": len(words),
        "jump_cuts": c.jump_cuts,
        "removed_gaps_s": round(sum(b - a for a, b in c.removed_gaps), 2),
    }


def deterministic_gates(c: Candidate, transcript: Transcript, sig: Signals, cfg: Config) -> list[GateResult]:
    g = cfg.gates
    m = clip_measurements(c, transcript, sig)
    out: list[GateResult] = []
    out.append(GateResult("length", g.min_len_s <= m["duration"] <= g.max_len_s,
                          f"{m['duration']:.1f}s (allowed {g.min_len_s:.0f}-{g.max_len_s:.0f}s)", m["duration"]))
    onset_limit = g.hook_window_s / 2
    onset = m["first_activity_at"]
    out.append(GateResult("hook_onset", onset is not None and onset <= onset_limit,
                          "nothing audible happens in the clip" if onset is None else
                          f"first speech/reaction at {onset:.1f}s (must start by {onset_limit:.1f}s "
                          f"so the hook can land within {g.hook_window_s:.1f}s)", onset))
    out.append(GateResult("dead_air", m["longest_dead_air"] <= g.max_dead_air_s,
                          f"longest dead air {m['longest_dead_air']:.1f}s (max {g.max_dead_air_s:.1f}s)",
                          m["longest_dead_air"]))
    if sig.events_available:
        out.append(GateResult("music_bed", m["music_fraction"] <= g.music_bed_max_fraction,
                              f"music detected in {m['music_fraction']:.0%} of the clip "
                              f"(max {g.music_bed_max_fraction:.0%})", m["music_fraction"]))
    else:
        out.append(GateResult("music_bed", True, "not measured (audio-event model unavailable); judges check it",
                              None))
    if m["word_count"] == 0:
        out.append(GateResult("has_speech", False, "no transcribed speech in the clip", 0.0))
    return out
