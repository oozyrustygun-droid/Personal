"""Boundary snapping: clips start on a sentence or beat edge and end after the payoff, never mid-word.

All functions are pure (words in, times out) so they can be unit-tested exhaustively.
Invariants enforced here and checked in tests:
  * no word straddles a clip boundary (start/end always fall in an inter-word gap or edge);
  * the end is never before the requested payoff time;
  * when a sentence edge is within reach, boundaries land on it; scene cuts in the gap
    are preferred because a visual cut makes the cleanest edit point.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Sequence

from .transcribe import Word

TERMINAL = (".", "!", "?", "…")
EPS = 1e-3


@dataclass(frozen=True)
class Sentence:
    start: float
    end: float
    i0: int  # first word index
    i1: int  # last word index (inclusive)


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"')]}»”’").endswith(TERMINAL)


def split_sentences(words: Sequence[Word], pause_break_s: float = 0.7) -> list[Sentence]:
    """Sentence/beat units: terminal punctuation, a long pause, or a speaker change closes a unit."""
    out: list[Sentence] = []
    if not words:
        return out
    i0 = 0
    for i, w in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        boundary = (
            nxt is None
            or _ends_sentence(w.text)
            or nxt.start - w.end >= pause_break_s
            or (w.speaker is not None and nxt.speaker is not None and w.speaker != nxt.speaker)
        )
        if boundary:
            out.append(Sentence(words[i0].start, w.end, i0, i))
            i0 = i + 1
    return out


def gap_before(words: Sequence[Word], i: int) -> float:
    return words[i].start - (words[i - 1].end if i > 0 else 0.0)


def gap_after(words: Sequence[Word], j: int, duration: float) -> float:
    return (words[j + 1].start if j + 1 < len(words) else duration) - words[j].end


def _cut_in(cuts: Sequence[float], lo: float, hi: float) -> float | None:
    k = bisect.bisect_left(cuts, lo - EPS)
    if k < len(cuts) and cuts[k] <= hi + EPS:
        return float(cuts[k])
    return None


def start_point(words: Sequence[Word], i: int, cuts: Sequence[float] = (), lead: float = 0.08,
                max_cut_lead: float = 0.5) -> float:
    """Clip start just before word i without touching word i-1 (prefers a scene cut in the gap)."""
    prev_end = words[i - 1].end if i > 0 else 0.0
    w = words[i]
    cut = _cut_in(cuts, max(prev_end, w.start - max_cut_lead), w.start)
    if cut is not None:
        return max(cut, prev_end)
    return max(prev_end, w.start - lead)


def end_point(words: Sequence[Word], j: int, duration: float, cuts: Sequence[float] = (), tail: float = 0.35,
              extend_to: float | None = None, max_cut_tail: float = 0.6) -> float:
    """Clip end just after word j without touching word j+1; may extend over a wordless reaction."""
    next_start = words[j + 1].start if j + 1 < len(words) else duration
    w = words[j]
    end = min(next_start, w.end + tail)
    if extend_to is not None and extend_to > end:
        end = min(next_start, extend_to)
    cut = _cut_in(cuts, w.end, min(next_start, w.end + max_cut_tail))
    if cut is not None and cut >= w.end and (extend_to is None or cut >= extend_to):
        end = cut
    return max(min(end, duration), w.end)


def snap_start(t: float, words: Sequence[Word], sentences: Sequence[Sentence], cuts: Sequence[float] = (),
               search_s: float = 6.0) -> tuple[float, int] | None:
    """Pick the first word of the clip near target t and return (start_time, word_index).

    Preference order: the nearest sentence start within ±search_s (earlier wins ties, since a
    little extra setup beats a missing setup); then a phrase start (a word preceded by a pause of
    at least 0.25 s); then the nearest word start.
    """
    if not words:
        return None
    starts = [s for s in sentences if abs(s.start - t) <= search_s]
    if starts:
        best = min(starts, key=lambda s: (abs(s.start - t), s.start))
        i = best.i0
    else:
        phrase = [k for k, w in enumerate(words) if abs(w.start - t) <= search_s and gap_before(words, k) >= 0.25]
        if phrase:
            i = min(phrase, key=lambda k: (abs(words[k].start - t), words[k].start))
        else:
            k = bisect.bisect_left([w.start for w in words], t)
            cands = [c for c in (k - 1, k) if 0 <= c < len(words)]
            i = min(cands, key=lambda c: abs(words[c].start - t))
    return start_point(words, i, cuts), i


def snap_end(t: float, words: Sequence[Word], sentences: Sequence[Sentence], duration: float,
             cuts: Sequence[float] = (), search_s: float = 6.0, tail: float = 0.35,
             extend_to: float | None = None) -> tuple[float, int] | None:
    """Pick the last word so the clip ends at the first sentence end at or after t (the payoff).

    The end is never earlier than t: a clip that stops before its payoff lands is worthless.
    If the containing sentence runs on past t + search_s, fall back to the first pause (≥0.25 s)
    after t, then to the end of the word that contains t.
    """
    if not words:
        return None
    starts = [w.start for w in words]
    p = bisect.bisect_right(starts, t + EPS) - 1  # last word starting at or before t
    if p < 0:
        anchor = 0
    elif words[p].end + EPS >= t:
        anchor = p  # t is inside word p: never cut through it
    else:
        # t is in a gap after word p (e.g. a wordless laugh). If p closes its sentence, end
        # there and stretch over the gap to cover t; otherwise finish p's sentence.
        sent = next((s for s in sentences if s.i0 <= p <= s.i1), None)
        if sent is None or sent.i1 == p:
            reach = max(extend_to or 0.0, t + 0.2)
            return end_point(words, p, duration, cuts, tail=tail, extend_to=reach), p
        anchor = p
    ends = [s for s in sentences if s.i1 >= anchor and s.end <= t + search_s]
    if ends:
        j = min(ends, key=lambda s: s.end).i1
    else:
        pauses = [m for m in range(anchor, len(words))
                  if words[m].end <= t + search_s and gap_after(words, m, duration) >= 0.25]
        j = pauses[0] if pauses else anchor
    # Whatever unit we chose, the clip must still reach t (a payoff in a mid-sentence pause).
    end = end_point(words, j, duration, cuts, tail=tail, extend_to=max(extend_to or 0.0, t))
    return end, j


def straddles(words: Sequence[Word], t: float, eps: float = EPS) -> bool:
    """True if time t falls strictly inside some word."""
    k = bisect.bisect_right([w.start for w in words], t) - 1
    return 0 <= k < len(words) and words[k].start + eps < t < words[k].end - eps


def tighten(start: float, end: float, dead_runs: Sequence[tuple[float, float]], max_dead: float,
            keep: float) -> list[tuple[float, float]]:
    """Remove dead-air runs longer than `max_dead` from [start, end] (jump cuts).

    Each removed run keeps `keep` seconds of air (half on either side) so the cut breathes.
    Dead air touching the clip edges is trimmed off instead of being jump-cut.
    """
    runs = sorted((max(s, start), min(e, end)) for s, e in dead_runs if e > start and s < end)
    # A clip never ends on silence: trailing dead air longer than the breathing room is trimmed whatever
    # its length (a 1.5 s silent ending is within the dead-air limit but reads as the clip dying).
    if runs and runs[-1][1] >= end - EPS and runs[-1][1] - runs[-1][0] > keep + 0.1 + EPS and runs[-1][0] > start + EPS:
        end = runs[-1][0] + keep / 2
        runs = runs[:-1]
    segments: list[tuple[float, float]] = []
    cur = start
    for s, e in runs:
        if e - s <= max_dead:
            continue
        if s <= start + EPS:  # leading dead air: move the start
            cur = max(cur, e - keep / 2)
            continue
        if e >= end - EPS:  # trailing dead air: move the end
            end = min(end, s + keep / 2)
            break
        cut_from, cut_to = s + keep / 2, e - keep / 2
        if cut_from > cur:
            segments.append((cur, cut_from))
        cur = max(cur, cut_to)
    if end > cur:
        segments.append((cur, end))
    return [(round(a, 3), round(b, 3)) for a, b in segments if b - a > 0.05]


def apply_preroll(segments: list[tuple[float, float]], words: Sequence[Word], cuts: Sequence[float],
                  preroll: float) -> list[tuple[float, float]]:
    """Prepend up to `preroll` seconds to the first segment. It is clamped so it never includes the tail
    of the previous word (the clip would open mid-word) or reaches back across a scene cut (it would
    open on a flash of the previous shot). Applied BEFORE judging, so the judged clip is the posted clip."""
    if not segments or preroll <= 0:
        return list(segments)
    s0, e0 = segments[0]
    prev_end = max((w.end for w in words if w.end <= s0 + EPS), default=0.0)
    last_cut = max((float(c) for c in cuts if c <= s0 + EPS), default=0.0)
    pre = max(0.0, min(preroll, s0 - prev_end - 0.02, s0 - last_cut, s0))
    return [(round(s0 - pre, 3), e0)] + list(segments[1:])
