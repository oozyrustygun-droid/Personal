"""Stage 4: recall. Propose 15-40 candidate windows, snapped to clean edit points.

Sources of candidates:
  * peaks of the interest curve (loudness spikes, laughter, screams, reaction words, cuts);
  * optional transcript proposals from a cheap model (deadpan/verbal moments).
For each source moment we build two kinds of window:
  * one that OPENS ON the moment (the phrase with the reaction lands inside the 2-second hook window,
    then the clip runs on through the aftermath), because a hook in the first two seconds is a hard
    gate and a reaction is usually the best hook a moment has;
  * build-up windows of several lengths that END just after the payoff (plus any wordless reaction
    tail), for moments whose setup is the hook.
Edges are snapped with `boundaries`, long dead air is jump-cut, variants are scored heuristically,
overlapping windows are de-duplicated (two openings of the same moment are different clips), and
the best moments are kept, each in its most hook-friendly version first.

This stage is deliberately generous. Precision is the judges' job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import bisect

import numpy as np

from .boundaries import (apply_preroll, gap_before, snap_end, snap_start, split_sentences, start_point, straddles,
                         tighten)
from .cache import Workspace, fingerprint, read_json, write_json
from .config import Config
from .signals import Signals
from .transcribe import Transcript, Word

CANDIDATES_VERSION = 12


@dataclass
class Candidate:
    id: str
    segments: list[tuple[float, float]]  # source-time spans kept in the clip (>1 means jump cuts)
    peak: float                          # source time of the payoff / peak
    origin: str                          # signal | llm | signal+llm
    score: float                         # heuristic recall score (NOT a quality verdict)
    reason: str
    llm_kind: str = ""
    llm_why: str = ""
    removed_gaps: list[tuple[float, float]] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.segments[0][0]

    @property
    def end(self) -> float:
        return self.segments[-1][1]

    @property
    def duration(self) -> float:
        return sum(e - s for s, e in self.segments)

    @property
    def jump_cuts(self) -> int:
        return len(self.segments) - 1

    def to_content(self, t: float) -> float:
        """Source time -> time inside the exported clip (before pre-roll)."""
        acc = 0.0
        for s, e in self.segments:
            if t < s:
                return acc
            if t <= e:
                return acc + (t - s)
            acc += e - s
        return acc

    def to_source(self, c: float) -> float:
        """Time inside the clip -> source time."""
        acc = 0.0
        for s, e in self.segments:
            if c <= acc + (e - s):
                return s + max(0.0, c - acc)
            acc += e - s
        return self.end

    def words(self, transcript: Transcript) -> list[Word]:
        return [w for w in transcript.words if any(s - 1e-3 <= w.start and w.end <= e + 1e-3 for s, e in self.segments)]

    def to_json(self) -> dict:
        d = asdict(self)
        d["segments"] = [list(x) for x in self.segments]
        d["removed_gaps"] = [list(x) for x in self.removed_gaps]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Candidate":
        d = dict(d)
        d["segments"] = [tuple(x) for x in d["segments"]]
        d["removed_gaps"] = [tuple(x) for x in d.get("removed_gaps", [])]
        return cls(**d)


# ---- helpers --------------------------------------------------------------------------------

def find_peaks(curve: np.ndarray, hop: float, min_sep_s: float, quantile: float = 0.7, limit: int = 120) -> list[int]:
    """Greedy non-maximum suppression over the smoothed interest curve."""
    if curve.size == 0:
        return []
    thr = float(np.quantile(curve, quantile))
    order = np.argsort(curve)[::-1]
    sep = int(min_sep_s / hop)
    taken: list[int] = []
    blocked = np.zeros(curve.size, dtype=bool)
    for i in order:
        if curve[i] <= thr or len(taken) >= limit:
            break
        if blocked[i]:
            continue
        taken.append(int(i))
        blocked[max(0, i - sep): i + sep + 1] = True
    return sorted(taken)


def activity_end(sig: Signals, t: float, max_s: float = 10.0, frac: float = 0.5, bridge_s: float = 1.5) -> float:
    """Where the high-interest region around a peak ends. The payoff usually sits at its end, not at
    the first maximum of a smoothed plateau. Dips shorter than `bridge_s` (a breath between
    "what is that thing?" and "OH MY GOD") don't end the moment."""
    x = sig.series["interest"]
    i = sig.idx(t)
    if i >= len(x):
        return t
    level = frac * float(x[i])
    last_above = i
    limit = min(len(x), i + int(max_s / sig.hop))
    k = i
    while k + 1 < limit:
        k += 1
        if x[k] >= level:
            last_above = k
        elif (k - last_above) * sig.hop > bridge_s:
            break
    return max(t, last_above * sig.hop)


def activity_start(sig: Signals, t: float, max_s: float = 10.0, frac: float = 0.5) -> float:
    """Where the high-interest region around a peak begins (the mirror of activity_end, no bridging)."""
    x = sig.series["interest"]
    i = sig.idx(t)
    if i >= len(x):
        return t
    level = frac * float(x[i])
    k = i
    floor = max(0, i - int(max_s / sig.hop))
    while k - 1 >= floor and x[k - 1] >= level:
        k -= 1
    return min(t, k * sig.hop)


def reaction_tail(sig: Signals, t: float, max_s: float = 6.0, lookahead_s: float = 1.0) -> float:
    """End of a wordless reaction (laugh, scream, crowd) running at t or starting within lookahead_s."""
    mask = sig.reaction_mask()
    i = sig.idx(t)
    k = i
    ahead = min(len(mask), i + int(lookahead_s / sig.hop))
    while k < ahead and not mask[k]:
        k += 1
    if k >= len(mask) or not mask[k]:
        return t
    j = k
    limit = min(len(mask), k + int(max_s / sig.hop))
    while j < limit and mask[j]:
        j += 1
    return j * sig.hop


def dead_runs(sig: Signals, start: float, end: float) -> list[tuple[float, float]]:
    dm = sig.dead_mask()
    i0, i1 = sig.idx(start), sig.idx(end)
    runs, s = [], None
    for i in range(i0, i1 + 1):
        d = i < i1 and i < len(dm) and dm[i]
        if d and s is None:
            s = i
        elif not d and s is not None:
            # a run still open at the last hop runs to the clip's end, not to the rounded hop
            runs.append((s * sig.hop, end if i == i1 else i * sig.hop))
            s = None
    return runs


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def heuristic_score(sig: Signals, segments: list[tuple[float, float]], peak: float, llm_strength: int = 0,
                    own_end: float | None = None) -> float:
    """own_end: score the peak and density only up to here, so a window that runs on into the next
    moment doesn't borrow that moment's score."""
    start, end = segments[0][0], segments[-1][1]
    dur = sum(e - s for s, e in segments)
    lim = end if own_end is None else max(start + 0.1, min(end, own_end))
    peak_val = sig.max_in("interest", start, lim)
    density = float(np.mean([sig.mean_in("interest", s, min(e, lim)) for s, e in segments if s < lim] or [0.0]))
    hook = sig.mean_in("interest", start, start + 2.0) + 0.5 * sig.max_in("reaction", start, start + 2.0)
    pos = (peak - start) / max(end - start, 1e-6)
    late = 1.0 if 0.4 <= pos <= 1.0 else 0.0
    length_pen = 0.03 * max(0.0, dur - 35.0) + 0.05 * max(0.0, 14.0 - dur)
    return round(1.0 * peak_val + 0.8 * density + 0.6 * hook + 0.3 * late + 0.25 * llm_strength - length_pen, 4)


# ---- generation -----------------------------------------------------------------------------

def hook_first_start(peak: float, words: list[Word], cuts: list[float], back_s: float,
                     sentences: list | None = None, onset: float | None = None) -> float:
    """A start that opens ON the moment: the clip starts where the reaction starts, and the peak lands
    within `back_s` of the first frame. `onset` is where the reaction begins (the interest rise), which
    can be well before the curve's maximum. In order of preference:
      1. speech that begins at the reaction: the latest phrase start (a word after a pause of >= 0.25 s)
         within ~0.6 s before to 0.3 s after the onset;
      2. the first word of the sentence the reaction is in, if it starts close enough;
      3. the first word starting at the reaction (mid-phrase);
      4. for a wordless reaction (a laugh, a scream), a scene cut just before it, else ~0.3 s before it,
         moved out of any word (after the word when the reaction starts after it, so the clip doesn't
         open on the last word of someone else's line).
    Never inside a word."""
    clamped = onset is not None and onset < peak - back_s  # a long build-up: search the whole window
    t0 = peak if onset is None else min(peak, max(onset, peak - back_s))
    starts = [w.start for w in words]
    lo_t = peak - back_s if clamped else max(peak - back_s, t0 - 0.6)
    near = list(range(bisect.bisect_left(starts, lo_t), bisect.bisect_right(starts, peak if clamped else min(peak, t0 + 0.3))))
    # The reaction starts in a pause after someone's line: that line's words are not part of the moment.
    j = bisect.bisect_right(starts, t0) - 1
    line_end = 0.0
    if j >= 0 and words[j].end <= t0 - 0.1 and (j + 1 == len(words) or words[j + 1].start - words[j].end >= 0.25):
        near = [k for k in near if k > j]
        line_end = words[j].end
    phrase = [k for k in near if gap_before(words, k) >= 0.25]
    if phrase:
        return start_point(words, phrase[0] if clamped else phrase[-1], cuts)
    if sentences is not None:
        own = next((s for s in sentences if s.end >= t0 - 0.1), None)
        if own is not None and peak - back_s <= own.start <= min(peak, t0 + 0.3) and own.start >= line_end:
            return start_point(words, own.i0, cuts)
    if near:
        return start_point(words, near[0], cuts)
    cut = next((float(c) for c in cuts if max(t0 - 0.6, line_end) <= c <= t0 and not straddles(words, float(c))),
               None)
    if cut is not None:  # open on the new shot rather than a flash of the previous one
        return cut
    t = max(0.0, t0 - min(0.3, back_s))
    k = bisect.bisect_right(starts, t) - 1
    if k >= 0 and words[k].end > t:  # t falls inside a word
        nxt = words[k + 1].start if k + 1 < len(words) else float("inf")
        if words[k].end + 0.02 <= min(t0, nxt):
            return words[k].end + 0.02  # just after it: the word belongs to the line before the reaction
        return start_point(words, k, cuts)
    return t


HOOK_SLACK_S = 0.25  # a scene cut can pull a start up to ~0.5 s before its first word


def opens_on_moment(start: float, peak: float, cfg: Config) -> bool:
    """The clip's peak lands inside the hook window. `start` is the clip's first frame (pre-roll included)."""
    return 0.0 <= peak - start <= cfg.gates.hook_window_s + HOOK_SLACK_S


def _hook_window(g: dict, peak: float, payoff: float, own_end: float, back: float, transcript: Transcript,
                 sentences, sig: Signals, duration: float, cfg: Config, cuts: list[float], seen: set):
    """The version of a moment that opens on it and runs on through the aftermath, or None.

    Lengths are tried shortest first (jump-cut dead air after a reaction can push the short one under
    the minimum). If the window would run into the next moment, it ends before it when that still
    leaves a full-length clip, else it takes the next moment's payoff too (two reactions in a row):
    never a clip that stops just as the next reaction lands."""
    cc = cfg.candidates
    words = transcript.words
    hs = hook_first_start(peak, words, cuts, back, sentences, onset=activity_start(sig, peak))
    if hs > peak:
        return None
    # Where each later moment begins (its reaction, or its own reaction-first opening, whichever is
    # first) and where its payoff ends. A clip may end before a moment or after it, never inside it.
    later = [(min(activity_start(sig, p), hook_first_start(p, words, cuts, back, sentences,
                                                           onset=activity_start(sig, p))) - 0.2, e)
             for p, e in (g.get("later") or []) if p > peak]
    nxt = later[0] if later else None
    for L in sorted(cc.target_lens_s) or [cc.min_len_s + 2.0]:
        t_end = max(payoff, hs + L)
        tries = [t_end]
        if nxt is not None and t_end + cc.snap_search_s >= nxt[0]:
            # End on the last sentence before the next moment if that makes a full clip; otherwise (or if
            # jump cuts shorten it too much) run on through the next moment's payoff.
            fits = [x.end for x in sentences if max(payoff, hs + cc.min_len_s) <= x.end <= nxt[0] - cc.end_tail_s]
            tries = fits[-1:] + [max(t_end, nxt[1])]
        for t in tries:
            built = build_window(hs, t, transcript, sentences, sig, duration, cfg, floor=g.get("floor", 0.0),
                                 fixed_start=True)
            if built is None:
                continue
            segments, removed = built
            end = segments[-1][1]
            if (segments[0][0], end) in seen or not segments[0][0] <= peak <= end:
                continue
            if any(lim < end < e for lim, e in later):
                continue  # would stop inside a later moment, before its payoff
            if not opens_on_moment(segments[0][0], peak, cfg):
                continue  # tightening moved the start: it no longer opens on the moment
            return heuristic_score(sig, segments, peak, g["strength"], own_end=own_end), segments, removed, False
    return None


def build_window(t_start: float, t_payoff: float, transcript: Transcript, sentences, sig: Signals,
                 duration: float, cfg: Config, floor: float = 0.0, fixed_start: bool = False,
                 ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    """Snap one (start target, payoff) pair into clean segments, or None if it can't be made valid.
    `floor`: earliest allowed start (the end of the previous moment), so one clip = one moment.
    `fixed_start`: t_start is already a clean edit point (see hook_first_start); don't move it."""
    cc = cfg.candidates
    words = transcript.words
    cuts = [float(c) for c in sig.scene_cuts]
    if fixed_start:
        if t_start < floor - 0.3:
            return None
        s = (t_start, None)
    else:
        s = snap_start(max(t_start, floor), words, sentences, cuts, cc.snap_search_s)
    if s is not None and s[0] < floor - 0.3:
        later = [w for w in sentences if w.start >= floor - 0.3 and w.start < t_payoff]
        s = (start_point(words, later[0].i0, cuts), later[0].i0) if later else None
    tail = reaction_tail(sig, t_payoff)
    e = snap_end(t_payoff, words, sentences, duration, cuts, cc.snap_search_s, cc.end_tail_s,
                 extend_to=tail if tail > t_payoff else None)
    if s is None or e is None:
        return None
    start, end = s[0], e[0]
    if end - start < cc.min_len_s * 0.6 or start >= t_payoff:
        return None
    runs = dead_runs(sig, start, end)
    if cc.tighten_dead_air:
        segments = tighten(start, end, runs, cfg.gates.max_dead_air_s, cc.tightened_gap_s)
    else:
        segments = [(start, end)]
    if not segments:
        return None
    segments = apply_preroll(segments, words, cuts, cfg.export.preroll_s)
    if straddles(words, segments[0][0]) or straddles(words, segments[-1][1]):
        return None
    removed = [(a[1], b[0]) for a, b in zip(segments, segments[1:])]
    dur = sum(b - a for a, b in segments)
    if not (cc.min_len_s <= dur <= cc.max_len_s):
        return None
    return segments, removed


def generate(transcript: Transcript, sig: Signals, duration: float, cfg: Config,
             proposals: list | None = None) -> list[Candidate]:
    cc = cfg.candidates
    sentences = split_sentences(transcript.words, cc.pause_break_s)
    if not transcript.words:
        return []

    groups: list[dict] = []
    peaks = [p * sig.hop for p in find_peaks(sig.series["interest"], sig.hop, cc.peak_min_separation_s)]
    ends = [activity_end(sig, p) for p in peaks]
    for k, p in enumerate(peaks):
        # Don't let a window reach back into the previous moment: its payoff would become this
        # clip's "setup", and the clip would contain two half-moments instead of one whole one.
        floor = ends[k - 1] + 0.2 if k > 0 and p - peaks[k - 1] < 60.0 else 0.0
        groups.append({"peak": p, "payoff": ends[k], "floor": min(floor, p - cc.min_len_s * 0.5),
                       "later": list(zip(peaks[k + 1:], ends[k + 1:])),
                       "origin": "signal", "strength": 0, "kind": "", "why": "", "starts": None})
    for m in proposals or []:
        # Same one-moment rule for transcript moments: not back past a reaction that ended before the
        # proposer's own start (a laugh inside the story's setup is part of the story).
        prev = [e for p, e in zip(peaks, ends) if e <= m.start_s + 0.5 and m.payoff_s - p < 60.0]
        floor = max(0.0, min(max(prev) + 0.2, m.payoff_s - cc.min_len_s * 0.5)) if prev else 0.0
        later = [(p, e) for p, e in zip(peaks, ends) if p > max(m.payoff_s, m.end_s) + 0.5]
        groups.append({"peak": m.payoff_s, "origin": "llm", "strength": m.strength, "kind": m.kind, "why": m.why,
                       "starts": [m.start_s], "end": m.end_s, "floor": floor, "later": later})

    cuts = [float(c) for c in sig.scene_cuts]
    back = max(0.5, cfg.gates.hook_window_s - cfg.export.preroll_s - 0.1)
    pool: list[tuple[Candidate, int, bool]] = []  # (candidate, moment index, primary version of its moment)
    for gi, g in enumerate(groups):
        peak = g["peak"]
        floor = g.get("floor", 0.0)
        payoff = max(peak, g.get("end", peak) - 0.5) if g["origin"] == "llm" else g["payoff"]
        own_end = max(payoff, reaction_tail(sig, payoff)) + 0.5
        variants = []  # (score, segments, removed, from the proposer's own start)
        seen = set()
        # Build-up variants are measured back from the payoff (so "14 s" really is a ~14 s clip).
        targets = [(t, True) for t in (g["starts"] or [])] + [(payoff - L, False) for L in cc.target_lens_s]
        for ts, proposed in targets:
            built = build_window(max(0.0, ts), payoff, transcript, sentences, sig, duration, cfg, floor=floor)
            if built is None:
                continue
            segments, removed = built
            key = (segments[0][0], segments[-1][1])
            if key in seen or not segments[0][0] <= peak <= segments[-1][1]:
                continue  # (tightening can trim a window off its own peak)
            seen.add(key)
            variants.append((heuristic_score(sig, segments, peak, g["strength"]), segments, removed, proposed))
        hook = _hook_window(g, peak, payoff, own_end, back, transcript, sentences, sig, duration, cfg, cuts, seen)
        variants.sort(key=lambda v: -v[0])
        on_moment = [v for v in variants if opens_on_moment(v[1][0][0], peak, cfg)]
        build_up = [v for v in variants if not opens_on_moment(v[1][0][0], peak, cfg)]
        if g["origin"] == "llm":  # the proposer's own window first: it chose where the setup starts
            build_up.sort(key=lambda v: (not v[3], -v[0]))
        hook = hook or (on_moment[0] if on_moment else None)
        if hook is None:
            picked = variants[:2]
        elif g["origin"] == "llm":
            # A transcript moment (a story, a punchline) usually needs its setup: lead with the build-up.
            picked = build_up[:1] + [hook] if build_up else [hook]
        else:
            # A reaction is the hook: lead with the version that opens on it.
            picked = [hook] + build_up[:1]
        for rank, (score, segments, removed, _) in enumerate(picked):
            reason = _describe(sig, segments, peak) if g["origin"] == "signal" else f"transcript: {g['why']}"
            if opens_on_moment(segments[0][0], peak, cfg):
                reason += " (opens on the moment)"
            pool.append((Candidate(id="", segments=segments, peak=round(peak, 2), origin=g["origin"], score=score,
                                   reason=reason, llm_kind=g["kind"], llm_why=g["why"], removed_gaps=removed),
                         gi, rank == 0))

    # De-duplicate: keep the higher-scoring of overlapping windows, merging provenance. Two versions of
    # the same moment that open differently (on the reaction vs. on the build-up) are different clips.
    pool.sort(key=lambda t: -t[0].score)
    kept: list[tuple[Candidate, int, bool]] = []

    def same_clip(a: Candidate, a_gi: int, b: Candidate, b_gi: int) -> bool:
        if iou((a.start, a.end), (b.start, b.end)) <= cc.nms_iou:
            return False
        a_on, b_on = opens_on_moment(a.start, a.peak, cfg), opens_on_moment(b.start, b.peak, cfg)
        if a_gi == b_gi and a_on != b_on:
            return False  # one moment opening on its reaction vs. on its build-up
        # A window that opens on its reaction is a different clip from one that opens seconds earlier.
        return not ((a_on or b_on) and abs(a.start - b.start) > cfg.gates.hook_window_s)

    for c, gi, primary in pool:
        k = next((n for n, (d, d_gi, _) in enumerate(kept) if same_clip(c, gi, d, d_gi)), None)
        if k is None:
            kept.append((c, gi, primary))
            continue
        d, d_gi, d_primary = kept[k]
        if primary and not d_primary:
            kept[k] = (d, d_gi, True)  # it now stands in for a moment's lead version
        if abs(d.peak - c.peak) <= cc.peak_min_separation_s and d.origin != c.origin and "+" not in d.origin:
            d.origin = "signal+llm"
            if c.llm_why and not d.llm_why:
                d.llm_kind, d.llm_why = c.llm_kind, c.llm_why

    # Keep the best moments, each in its primary version, then add the other opening of those moments
    # while there is room (alt_openings).
    target_n = int(np.clip(round(duration / 90.0), cc.min_candidates, cc.max_candidates))
    by_moment: dict[int, list[tuple[Candidate, bool]]] = {}
    for c, gi, primary in kept:  # score order
        by_moment.setdefault(gi, []).append((c, primary))
    moments = sorted(by_moment, key=lambda gi: -max(c.score for c, _ in by_moment[gi]))[:target_n]
    chosen: list[Candidate] = []
    extras: list[Candidate] = []
    for gi in moments:
        versions = sorted(by_moment[gi], key=lambda v: (not v[1], -v[0].score))
        chosen.append(versions[0][0])
        extras += [c for c, _ in versions[1:]]
    if cc.alt_openings:
        for c in sorted(extras, key=lambda c: -c.score):
            if len(chosen) >= cc.max_candidates:
                break
            chosen.append(c)
    chosen = sorted(chosen, key=lambda c: c.start)
    for i, c in enumerate(chosen, 1):
        c.id = f"c{i:02d}"
    return chosen


def _describe(sig: Signals, segments, peak: float) -> str:
    s, e = segments[0][0], segments[-1][1]
    bits = []
    for name, label in (("laughter", "laughter"), ("scream", "scream/shout"), ("crowd", "crowd"),
                        ("reaction", "reaction words"), ("loud_spike", "loudness spike")):
        v = sig.max_in(name, s, e)
        if v >= (0.5 if name != "loud_spike" else 0.8):
            bits.append(label)
    if sig.cuts_between(s, e):
        bits.append(f"{len(sig.cuts_between(s, e))} cut(s)")
    return "signal peak: " + (", ".join(bits) if bits else "interest peak")


def candidates_fp(cfg: Config, signals_fp: str, llm=None) -> str:
    use_llm = cfg.candidates.llm_proposer and llm is not None
    return fingerprint("candidates", CANDIDATES_VERSION, cfg.section_hash("candidates", "gates"), signals_fp,
                       use_llm and cfg.candidates.proposer_model, use_llm and llm.backend, cfg.export.preroll_s)


def cached_proposals(ws: Workspace, signals_fp: str, cfg: Config, llm=None) -> list | None:
    """The transcript proposer's moments from a finished candidates stage on the same transcript and
    signals (free to reuse), or None."""
    from .proposer import Moment

    out = ws.path("candidates.json")
    if not ws.is_done("candidates", candidates_fp(cfg, signals_fp, llm), [out]):
        return None
    try:
        return [Moment.model_validate(d) for d in read_json(out).get("proposals", [])]
    except (OSError, ValueError):
        return None


def candidates_stage(ws: Workspace, transcript: Transcript, sig: Signals, signals_fp: str, duration: float,
                     cfg: Config, llm=None, *, console=None, force: bool = False) -> tuple[list[Candidate], str]:
    use_llm = cfg.candidates.llm_proposer and llm is not None
    fp = candidates_fp(cfg, signals_fp, llm)
    out = ws.path("candidates.json")
    if not force and ws.is_done("candidates", fp, [out]):
        cached = read_json(out)
        # A proposer pass that failed (no credentials, rate limit) is retried instead of silently
        # losing recall; its successful chunks come back from the response cache for free.
        if not (use_llm and cached.get("proposer_incomplete")):
            return [Candidate.from_json(d) for d in cached["candidates"]], fp
    proposals, complete = [], True
    if use_llm:
        from .proposer import propose

        proposals, complete = propose(llm, transcript, duration, cfg, console=console)
        if console:
            note = "" if complete else " (some chunks failed; will retry next run)"
            console.print(f"  [dim]transcript proposer flagged {len(proposals)} moments{note}[/dim]")
    cands = generate(transcript, sig, duration, cfg, proposals)
    write_json(out, {"candidates": [c.to_json() for c in cands], "proposer_incomplete": not complete,
                     "proposals": [p.model_dump() for p in proposals]})
    ws.mark_done("candidates", fp, count=len(cands))
    return cands, fp
