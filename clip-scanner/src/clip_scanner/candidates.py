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

CANDIDATES_VERSION = 8


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
            runs.append((s * sig.hop, i * sig.hop))
            s = None
    return runs


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def heuristic_score(sig: Signals, segments: list[tuple[float, float]], peak: float, llm_strength: int = 0) -> float:
    start, end = segments[0][0], segments[-1][1]
    dur = sum(e - s for s, e in segments)
    peak_val = sig.max_in("interest", start, end)
    density = float(np.mean([sig.mean_in("interest", s, e) for s, e in segments]))
    hook = sig.mean_in("interest", start, start + 2.0) + 0.5 * sig.max_in("reaction", start, start + 2.0)
    pos = (peak - start) / max(end - start, 1e-6)
    late = 1.0 if 0.4 <= pos <= 1.0 else 0.0
    length_pen = 0.03 * max(0.0, dur - 35.0) + 0.05 * max(0.0, 14.0 - dur)
    return round(1.0 * peak_val + 0.8 * density + 0.6 * hook + 0.3 * late + 0.25 * llm_strength - length_pen, 4)


# ---- generation -----------------------------------------------------------------------------

def hook_first_start(peak: float, words: list[Word], cuts: list[float], back_s: float) -> float:
    """A start that opens ON the moment, so the peak lands within `back_s` of the first frame.

    Prefers the earliest phrase start (a word after a pause of >= 0.25 s) in [peak - back_s, peak], so
    the line carrying the reaction is heard from its first word; else the earliest word start in that
    range (on the moment, mid-phrase); else, for a wordless reaction, a point just before the peak that
    doesn't cut into a word. Never inside a word."""
    starts = [w.start for w in words]
    lo, hi = bisect.bisect_left(starts, peak - back_s), bisect.bisect_right(starts, peak)
    in_range = list(range(lo, hi))
    phrase = [k for k in in_range if gap_before(words, k) >= 0.25]
    if phrase or in_range:
        return start_point(words, (phrase or in_range)[0], cuts)
    t = max(0.0, peak - min(0.6, back_s))
    k = bisect.bisect_right(starts, t) - 1
    if k >= 0 and words[k].end > t:  # t falls inside a word: start before that word instead
        return start_point(words, k, cuts)
    return t


def opens_on_moment(start: float, peak: float, cfg: Config) -> bool:
    """The clip's peak lands inside the hook window (allowing for the pre-roll)."""
    return peak - start <= cfg.gates.hook_window_s + cfg.export.preroll_s


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
                       "origin": "signal", "strength": 0, "kind": "", "why": "", "starts": None})
    for m in proposals or []:
        groups.append({"peak": m.payoff_s, "origin": "llm", "strength": m.strength, "kind": m.kind, "why": m.why,
                       "starts": [m.start_s], "end": m.end_s})

    cuts = [float(c) for c in sig.scene_cuts]
    back = max(0.5, cfg.gates.hook_window_s - cfg.export.preroll_s - 0.1)
    pool: list[tuple[Candidate, int, bool]] = []  # (candidate, moment index, primary version of its moment)
    for gi, g in enumerate(groups):
        peak = g["peak"]
        variants = []
        payoff = max(peak, g.get("end", peak) - 0.5) if g["origin"] == "llm" else g["payoff"]
        # Build-up variants are measured back from the payoff (so "14 s" really is a ~14 s clip).
        targets = list(g["starts"] or [])
        targets += [payoff - L for L in cc.target_lens_s]
        seen = set()
        for ts in targets:
            ts = max(0.0, ts)
            built = build_window(ts, payoff, transcript, sentences, sig, duration, cfg, floor=g.get("floor", 0.0))
            if built is None:
                continue
            segments, removed = built
            key = (segments[0][0], segments[-1][1])
            if key in seen:
                continue
            seen.add(key)
            score = heuristic_score(sig, segments, peak, g["strength"])
            variants.append((score, segments, removed))
        # The version that opens on the moment and runs on through the aftermath.
        hook = None
        hs = hook_first_start(peak, transcript.words, cuts, back)
        built = build_window(hs, max(payoff, hs + cc.target_lens_s[0]), transcript, sentences, sig, duration, cfg,
                             floor=g.get("floor", 0.0), fixed_start=True)
        if built is not None and (built[0][0][0], built[0][-1][1]) not in seen:
            hook = (heuristic_score(sig, built[0], peak, g["strength"]), built[0], built[1])
        variants.sort(key=lambda v: -v[0])
        on_moment = [v for v in variants if opens_on_moment(v[1][0][0], peak, cfg)]
        build_up = [v for v in variants if not opens_on_moment(v[1][0][0], peak, cfg)]
        hook = hook or (on_moment[0] if on_moment else None)
        if hook is None:
            picked = variants[:2]
        elif g["origin"] == "llm":
            # A transcript moment (a story, a punchline) usually needs its setup: lead with the build-up.
            picked = build_up[:1] + [hook] if build_up else [hook]
        else:
            # A reaction is the hook: lead with the version that opens on it.
            picked = [hook] + build_up[:1]
        for rank, (score, segments, removed) in enumerate(picked):
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
    for c, gi, primary in pool:
        c_on = opens_on_moment(c.start, c.peak, cfg)
        dup = next((k for k in kept if iou((c.start, c.end), (k[0].start, k[0].end)) > cc.nms_iou
                    and not (k[1] == gi and opens_on_moment(k[0].start, k[0].peak, cfg) != c_on)), None)
        if dup is None:
            kept.append((c, gi, primary))
            continue
        d = dup[0]
        if d.origin != c.origin and "+" not in d.origin:
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


def candidates_stage(ws: Workspace, transcript: Transcript, sig: Signals, signals_fp: str, duration: float,
                     cfg: Config, llm=None, *, console=None, force: bool = False) -> tuple[list[Candidate], str]:
    use_llm = cfg.candidates.llm_proposer and llm is not None
    fp = fingerprint("candidates", CANDIDATES_VERSION, cfg.section_hash("candidates", "gates"), signals_fp,
                     use_llm and cfg.candidates.proposer_model, use_llm and llm.backend, cfg.export.preroll_s)
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
