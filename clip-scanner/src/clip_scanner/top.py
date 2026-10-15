"""Top-moments mode: always the N moments viewers care most about, each as a raw 1-3 minute landscape clip.

Where people rewind is the best evidence of what they found clip-worthy, so YouTube's "Most replayed"
graph (saved at ingest when the video has one) leads. The scanner's own signals (laughter, screams,
loudness, reaction words) and the transcript proposer's flags fill in, and carry the ranking alone when
there is no graph. The best-scoring, non-overlapping regions then go to Claude in one call, which reads
their transcripts and picks the best N, best first, with a label, a reason, the peak second and a
suggested text hook. If that call can't run, the measured order is used.

Nothing here judges whether a moment is GREAT: the editor gets N raw segments and makes the short.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from .boundaries import end_point, split_sentences, start_point
from .cache import Workspace, read_json
from .config import Config, load_style_guide
from .judging.prompts import data_text
from .llm import LLM, LLMError
from .proposer import transcript_lines
from .signals import Signals
from .transcribe import Transcript, Word
from .ui import fmt_ts

TOP_VERSION = 1
CORE_FRAC = 0.6        # a region's core: the seconds around its peak scoring at least this share of the peak
FLAG_SIGMA_S = 12.0    # how far a transcript flag's bump reaches
SNAP_S = 6.0           # how far a clip edge may move to land on a sentence edge
ADJUST_S = 30.0        # how far Claude may move a region's edges
SHOULDER_DIP = 0.8     # a peak next to a picked hill is its own moment only if the score dips below this between


@dataclass
class Region:
    start: float
    end: float
    peak: float                      # second the combined score peaks
    score: float                     # combined score at the peak (0..1)
    lo: float                        # Claude may move the edges within [lo, hi]
    hi: float
    replay_peak: float | None = None  # YouTube "Most replayed" value at its highest in the clip (0..1)
    replay_mean: float | None = None
    sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)   # transcript proposer notes inside the clip

    @property
    def duration(self) -> float:
        return self.end - self.start


# ---- the curve --------------------------------------------------------------------------------------

def load_heatmap(ws: Workspace) -> list[dict]:
    path = ws.path("heatmap.json")
    if not path.exists():
        return []
    try:
        segs = read_json(path).get("segments") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [s for s in segs if isinstance(s, dict)]


def _heat_points(segments: list[dict]) -> list[tuple[float, float, float]]:
    """(centre, width, value) of each valid replay-graph segment, in time order."""
    pts = []
    for s in segments:
        try:
            a, b, v = float(s["start_time"]), float(s["end_time"]), float(s["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b) and math.isfinite(v) and b > a:
            pts.append(((a + b) / 2, b - a, max(0.0, v)))
    return sorted(pts)


def edge_windows(segments: list[dict], duration: float, cfg: Config) -> tuple[float, float]:
    """(head_s, tail_s): the stretches at the start (people pressing play, skipping the intro) and at the end
    (the end card) that are weighed down. Scaled to short videos. On long ones YouTube's graph segments are
    duration/100 wide, so the head also runs to the bottom of the graph's opening decline (within the first
    10 % of the video): a play-press spike spread over several segments can't outlast it."""
    head = min(cfg.top.skip_intro_s, 0.1 * duration)
    pts = _heat_points(segments)
    if pts:
        k = 0
        while k + 1 < len(pts) and pts[k + 1][2] < pts[k][2] and pts[k + 1][0] <= 0.1 * duration:
            k += 1
        head = max(head, 1.5 * pts[0][1], pts[k][0])
    return min(head, 0.25 * duration), min(15.0, 0.05 * duration)


def heat_series(segments: list[dict], duration: float, head_s: float = 0.0,
                tail_s: float = 0.0) -> np.ndarray | None:
    """YouTube's replay graph resampled to one value per second, or None if it is missing or partial.
    1.0 is the most replayed second outside the first head_s and last tail_s seconds; a play-press or end-card
    spike there is capped at 1.0 instead of setting the scale for the whole video."""
    n = max(1, int(math.ceil(duration)))
    pts = _heat_points(segments)
    if len(pts) < 10:
        return None
    pts.sort()
    covered = sum(w for _, w, _ in pts)
    if covered < 0.8 * duration:
        return None
    top = max(v for _, _, v in pts)
    if top <= 0:
        return None
    t = np.arange(n) + 0.5
    vals = np.interp(t, [p[0] for p in pts], [p[2] / top for p in pts]).astype(np.float64)
    body = vals[int(head_s): max(int(head_s), n - int(tail_s))]
    scale = float(body.max()) if body.size and body.max() > 0 else 1.0
    return np.clip(vals / scale, 0.0, 1.0)


def _moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(x) == 0:
        return x.astype(np.float64)
    k = np.ones(width) / width
    pad = width // 2
    return np.convolve(np.pad(x.astype(np.float64), (pad, width - 1 - pad), mode="edge"), k, mode="valid")


def score_curve(sig: Signals, duration: float, heat: np.ndarray | None, proposals: list, cfg: Config,
                head_s: float | None = None, tail_s: float | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Combined per-second score (0..1) and its parts. head_s / tail_s: see edge_windows."""
    n = max(1, int(math.ceil(duration)))
    interest = sig.per_second("interest", "mean")[:n]
    interest = np.pad(interest, (0, n - len(interest))) if len(interest) < n else interest
    smooth = _moving_mean(interest, 7)
    ref = float(np.percentile(smooth, 99)) if smooth.size else 0.0
    react = np.clip(smooth / ref, 0, 1) if ref > 1e-9 else np.zeros(n)
    if ref > 1e-9:  # the top 1 % saturate at 1.0: keep them ordered by strength, not by time
        react = (1 - 1e-3) * react + 1e-3 * smooth / float(smooth.max())
    flags = np.zeros(n)
    t = np.arange(n) + 0.5
    for m in proposals:
        try:
            centre, strength = float(m.payoff_s), float(m.strength)
        except (AttributeError, TypeError, ValueError):
            continue
        flags = np.maximum(flags, (strength / 5.0) * np.exp(-0.5 * ((t - centre) / FLAG_SIGMA_S) ** 2))
    parts = {"reaction": react, "flags": flags}
    if heat is not None:
        base = float(np.median(heat))
        replay = np.clip((heat - base) / max(1e-9, 1.0 - base), 0, 1)
        parts["replay"] = replay
        score = 0.6 * replay + 0.25 * react + 0.15 * flags
    else:
        score = 0.65 * react + 0.35 * flags
    # The replay graph spikes at the start (people pressing play, skipping the intro) and the end card
    # never makes a clip: weigh both down without ruling them out.
    if head_s is None or tail_s is None:
        head_s, tail_s = edge_windows([], duration, cfg)
    intro = min(n, int(head_s))
    if intro > 0:
        score[:intro] *= np.linspace(0.3, 1.0, intro)
    outro = min(n, int(tail_s))
    if outro > 0:
        score[n - outro:] *= 0.5
    return score, parts


def curve(segments: list[dict], sig: Signals, duration: float, proposals: list,
          cfg: Config) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray | None]:
    """(score, parts, heat) for a video: the replay graph (if any) and the signals on one scale."""
    head, tail = edge_windows(segments, duration, cfg)
    heat = heat_series(segments, duration, head, tail)
    score, parts = score_curve(sig, duration, heat, proposals, cfg, head, tail)
    return score, parts, heat


# ---- clip edges -------------------------------------------------------------------------------------

def snap_edges(s: float, e: float, lo: float, hi: float, words: list[Word], sentences,
               max_len: float | None = None) -> tuple[float, float]:
    """Move s and e (within [lo, hi] and SNAP_S) onto sentence edges; never inside a word, never longer than
    max_len. An edge that already sits in the silence next to that sentence stays put, so a wordless reaction
    isn't trimmed off."""
    if not words:
        return s, e
    starts = [w.start for w in words]
    near = [x for x in sentences if max(lo, s - SNAP_S) <= x.start <= min(hi, s + SNAP_S)]
    if near:
        best = min(near, key=lambda x: (abs(x.start - s), x.start))
        snapped = start_point(words, best.i0)
        in_gap = s < snapped and (best.i0 == 0 or words[best.i0 - 1].end <= s)
        s = s if in_gap else max(lo, snapped)
    else:
        k = bisect.bisect_right(starts, s) - 1
        if 0 <= k < len(words) and words[k].start < s < words[k].end:  # inside a word: start before it
            s = words[k].start if words[k].start >= lo else words[k].end
    if max_len is not None:
        hi = min(hi, s + max_len)
        e = min(e, hi)
    ends = [x for x in sentences if max(lo, e - SNAP_S) <= x.end <= min(hi, e + SNAP_S)]
    if ends:
        best = min(ends, key=lambda x: (abs(x.end - e), -x.end))
        e = min(hi, end_point(words, best.i1, hi, extend_to=e))
    else:
        k = bisect.bisect_right(starts, e) - 1
        if 0 <= k < len(words) and words[k].start < e < words[k].end:
            e = words[k].end if words[k].end <= hi else words[k].start
    return s, e


def fit_length(s: float, e: float, focus: float, lo: float, hi: float, cfg: Config) -> tuple[float, float]:
    """Clamp to [lo, hi] and to min_len_s..max_len_s, keeping `focus` inside with build-up before it."""
    tc = cfg.top
    s, e = max(lo, s), min(hi, e)
    if e - s > tc.max_len_s:
        s = max(s, min(focus - 0.4 * tc.max_len_s, e - tc.max_len_s))
        e = s + tc.max_len_s
    if e - s < tc.min_len_s:
        need = tc.min_len_s - (e - s)
        s2 = max(lo, s - 0.6 * need)
        e = min(hi, e + need - (s - s2))
        s = max(lo, min(s2, e - tc.min_len_s))
    return s, e


def _shoulder(score: np.ndarray, near: np.ndarray, peaks: list[int], p: int, a: int, b: int) -> bool:
    """The core a..b around peak p runs into a picked hill (or a slope already traced from one) that is higher
    there, and the score never dips below SHOULDER_DIP of p's own between that pick's peak and p: it is the
    hill's slope, not a moment of its own. A separate moment whose crest hides in a pick's gap zone has its dip
    between the two peaks, so it is kept."""
    v = float(score[p])
    floor = SHOULDER_DIP * v
    if a > 0 and near[a - 1] and score[a - 1] > v:
        left = [q for q in peaks if q < p]
        q = max(left) if left else a
        if float(score[min(q, a): p + 1].min()) >= floor:
            return True
    if b < len(score) - 1 and near[b + 1] and score[b + 1] > v:
        right = [q for q in peaks if q > p]
        q = min(right) if right else b
        if float(score[p: max(q, b) + 1].min()) >= floor:
            return True
    return False


def _greedy(score: np.ndarray, duration: float, words: list[Word], sentences, cfg: Config, limit: int,
            need: int, strict: bool) -> list[tuple[float, float, float, float, int]]:
    """Pick (start, end, focus, score, peak index) windows, best first. Strict: only distinct moments with
    room for a full-length clip; if that yields fewer than `need`, continue with the rest allowed."""
    tc = cfg.top
    n = len(score)
    taken_mask = np.zeros(n, dtype=bool)   # picked clips plus min_gap_s around them
    skipped = np.zeros(n, dtype=bool)      # peaks passed over in this pass
    slope = np.zeros(n, dtype=bool)        # cores skipped as a pick's slope: the slope continues past them
    out: list[tuple[float, float, float, float, int]] = []
    while len(out) < limit:
        masked = taken_mask | skipped
        avail = np.where(masked, -np.inf, score)
        p = int(np.argmax(avail))
        if not np.isfinite(avail[p]):
            if strict and len(out) < need:
                strict = False  # too few distinct moments: take the near-misses too
                skipped[:] = slope[:] = False
                continue
            break
        v = float(score[p])
        a = b = p
        while a > 0 and not masked[a - 1] and score[a - 1] >= CORE_FRAC * v and p - a < tc.max_len_s / 2:
            a -= 1
        while b < n - 1 and not masked[b + 1] and score[b + 1] >= CORE_FRAC * v and b - p < tc.max_len_s / 2:
            b += 1
        if strict and _shoulder(score, taken_mask | slope, [x[4] for x in out], p, a, b):
            skipped[a: b + 1] = slope[a: b + 1] = True
            continue
        free_lo = max([x[1] for x in out if x[1] <= p] + [0.0])
        free_hi = min([x[0] for x in out if x[0] >= p + 1] + [duration])
        focus = min(p + 0.5, duration)
        s, e = fit_length(a - tc.lead_s, b + 1 + tc.tail_s, focus, free_lo, free_hi, cfg)
        room = e - s  # what fits between earlier picks (snapping below trims up to SNAP_S per edge)
        s, e = snap_edges(s, e, free_lo, free_hi, words, sentences, tc.max_len_s)
        floor = min(tc.min_len_s, duration)
        if (strict and room < 0.9 * floor) or e - s < 0.5 * floor or not s <= focus <= e:
            skipped[a: b + 1] = True  # squeezed between earlier picks: no room for a clip of its own
            continue
        out.append((s, e, focus, v, p))
        taken_mask[max(0, int(s - tc.min_gap_s)): min(n, int(math.ceil(e + tc.min_gap_s)))] = True
    return out


def find_regions(score: np.ndarray, parts: dict[str, np.ndarray], heat: np.ndarray | None, duration: float,
                 transcript: Transcript, proposals: list, cfg: Config, limit: int, need: int = 0) -> list[Region]:
    """The best-scoring non-overlapping clip windows, best first. Distinct moments with room for a full-length
    clip come first; if there are fewer than `need`, slopes of a picked hill and clips squeezed between picks
    (down to half of min_len_s) are allowed too."""
    tc = cfg.top
    sentences = split_sentences(transcript.words, cfg.candidates.pause_break_s)
    if duration <= tc.min_len_s:
        s, e = 0.0, duration
        peak = int(np.argmax(score)) if score.size else 0
        return [_describe(Region(s, e, min(float(peak) + 0.5, duration), float(score[peak]) if score.size else 0.0,
                                 s, e), parts, heat, proposals)]
    picks = _greedy(score, duration, transcript.words, sentences, cfg, limit, need, strict=True)
    if len(picks) < need:
        # Strict picks can take the room a skipped slope needed: start over with everything allowed.
        alt = _greedy(score, duration, transcript.words, sentences, cfg, limit, need, strict=False)
        if len(alt) > len(picks):
            picks = alt
    out = [_describe(Region(s, e, focus, v, s, e), parts, heat, proposals) for s, e, focus, v, _ in picks]
    # How far Claude may move each edge: ADJUST_S, never into another region.
    for r in out:
        r.lo = max([x.end for x in out if x.end <= r.start] + [0.0, r.start - ADJUST_S])
        r.hi = min([x.start for x in out if x.start >= r.end] + [duration, r.end + ADJUST_S])
    return out


def _describe(r: Region, parts: dict[str, np.ndarray], heat: np.ndarray | None, proposals: list) -> Region:
    i0, i1 = int(r.start), max(int(r.start) + 1, int(math.ceil(r.end)))
    if heat is not None:
        r.replay_peak = round(float(heat[i0:i1].max()), 3)
        r.replay_mean = round(float(heat[i0:i1].mean()), 3)
        if float(parts["replay"][i0:i1].max()) >= 0.5:
            r.sources.append("most replayed")
    if float(parts["reaction"][i0:i1].max()) >= 0.6:
        r.sources.append("audio reaction")
    for m in proposals:
        if r.start <= getattr(m, "payoff_s", -1) <= r.end:
            r.flags.append(f"{fmt_ts(m.payoff_s)} {m.kind} (strength {m.strength}): {m.why}")
    if r.flags:
        r.sources.append("transcript")
    return r


# ---- ranking ----------------------------------------------------------------------------------------

class TopPick(BaseModel):
    region: int = Field(ge=1)
    title: str
    why: str
    focus_s: float
    start_s: float
    end_s: float
    text_hook: str


class TopRanking(BaseModel):
    picks: list[TopPick]


RANK_SYSTEM = """You choose moments from a long Joe Bartolozzi video for a clipping account (TikTok / YouTube \
Shorts / Reels). The editor cuts each short from a raw 1-3 minute landscape segment you choose. You pick \
WHICH moments; the editor does the rest.

You get candidate regions found by measurement:
- YouTube's "Most replayed" graph, when the video has one: where real viewers rewound and rewatched. It is \
the strongest evidence of what people found worth watching again. 1.00 is the most replayed part of the \
video outside its first and last moments (a spike there, from people pressing play or skipping to the end, \
is capped at 1.00).
- Audio reactions (laughter, screams, loudness spikes) and transcript flags.
- The transcript of each region (speech-to-text: it can mishear names and words).

Pick the best {count} regions, best first. Rank by how well the moment works as a short for a viewer who \
has never seen Joe: an explosive reaction, a punchline, an absurd line, a confrontation, a blunt verdict, a \
story payoff, a jumpscare, a laugh fit. Favour the regions viewers replayed most, unless the replays are \
explained by something that doesn't clip: a sponsor read or ad, the intro, a giveaway or "link in the \
description", the end card, or a purely visual moment with nothing to hear. Never pick a sponsor read.

For each pick:
- region: its number.
- title: a label for the editor, at most 8 words.
- why: one or two sentences: what happens and why it holds a stranger.
- focus_s: the second the moment peaks (absolute video time, inside the region).
- start_s, end_s: the raw segment (absolute video times). Keep the setup a viewer needs and the reaction \
after it, 60-180 seconds long, inside that region's allowed range. Use the region's own edges if they \
already work.
- text_hook: an on-screen hook for the top of the vertical short, at most 9 words, true to the clip, no \
spoiler of the payoff, no emojis.
Return exactly {count} picks if at least {count} regions are given, otherwise one per region.

Style guide for this creator:
"""


def ranking_message(regions: list[Region], transcript: Transcript, title: str, duration: float, cfg: Config,
                    has_heat: bool, count: int) -> str:
    parts = [f"Video: {data_text(title)} ({fmt_ts(duration)} long).",
             "Most replayed data: " + ("available." if has_heat else "not available for this video (rank on "
                                       "the transcript and reactions)."), ""]
    for k, r in enumerate(regions, 1):
        heat = (f"most replayed: peak {r.replay_peak:.2f}, region average {r.replay_mean:.2f}"
                if r.replay_peak is not None else "most replayed: no data")
        lines = transcript_lines(transcript, r.start, r.end, cfg.candidates.pause_break_s) or "(no speech)"
        flags = "".join(f"\ntranscript flag: {data_text(f)}" for f in r.flags)
        parts.append(
            f'<region n="{k}" start_s="{r.start:.1f}" end_s="{r.end:.1f}" allowed_start_s="{r.lo:.1f}" '
            f'allowed_end_s="{r.hi:.1f}" measured_peak_s="{r.peak:.1f}">\n{heat}\n'
            f"measured: {', '.join(r.sources) or 'interest peak'}{flags}\n"
            f"<transcript>\n{data_text(lines)}\n</transcript>\n</region>")
    parts.append(f"\nPick the best {min(count, len(regions))} and return them as JSON, best first.")
    return "\n".join(parts)


def ranking_request(regions: list[Region], transcript: Transcript, title: str, duration: float, cfg: Config,
                    has_heat: bool, count: int) -> dict:
    """The one ranking call's arguments for LLM.call (also used to tell whether it is already cached)."""
    system = [{"type": "text", "text": RANK_SYSTEM.replace("{count}", str(min(count, len(regions))))
               + load_style_guide(), "cache_control": {"type": "ephemeral"}}]
    msg = ranking_message(regions, transcript, title, duration, cfg, has_heat, count)
    meta = {"regions": [[r.start, r.end, r.peak] for r in regions], "count": count,
            "texts": [transcript.text_between(r.start, r.end) for r in regions]}
    return {"purpose": "top_rank", "model": cfg.judging.model, "system": system,
            "messages": [{"role": "user", "content": msg}], "output": TopRanking, "effort": cfg.top.effort,
            "meta": meta}


@dataclass
class Moment:
    rank: int
    start: float
    end: float
    focus: float
    title: str
    why: str
    text_hook: str
    region: Region

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TopResult:
    moments: list[Moment]
    regions: list[Region]
    method: str                  # "claude" | "signals"
    has_heat: bool
    notes: list[str] = field(default_factory=list)
    served_model: str = ""
    recall_incomplete: bool = False  # the transcript proposer missed part of the video (no replay data)
    cancelled: bool = False          # the ranking didn't run and nothing was exported
    error: str = ""                  # why Claude couldn't rank (method "signals" after a failed call)
    error_final: bool = False        # that failure repeats on every re-run until the request changes


def _fallback_title(r: Region) -> str:
    return f"Moment at {fmt_ts(r.peak).split('.')[0]}"


def _fallback_why(r: Region) -> str:
    bits = []
    if r.replay_peak is not None:
        bits.append(f"most replayed peak {r.replay_peak:.2f}")
    bits += [s for s in r.sources if s != "most replayed"]
    head = "Ranked by measured signals: " + (", ".join(bits) if bits else "interest peak") + "."
    return head + (f" Transcript flag: {r.flags[0]}" if r.flags else "")


def apply_picks(picks: list[dict], regions: list[Region], transcript: Transcript, cfg: Config,
                count: int) -> list[Moment]:
    """Turn Claude's picks into non-overlapping moments; invalid edges fall back to the region's own."""
    sentences = split_sentences(transcript.words, cfg.candidates.pause_break_s)
    out: list[Moment] = []
    used: set[int] = set()
    for p in picks:
        k = int(p.get("region", 0)) - 1
        if not 0 <= k < len(regions) or k in used:
            continue
        r = regions[k]
        focus = float(p.get("focus_s", r.peak))
        if not (math.isfinite(focus) and r.lo <= focus <= r.hi):
            focus = r.peak
        s, e = float(p.get("start_s", r.start)), float(p.get("end_s", r.end))
        lo = max([m.end for m in out if m.end <= focus] + [r.lo])
        hi = min([m.start for m in out if m.start >= focus] + [r.hi])
        ok = math.isfinite(s) and math.isfinite(e) and e > s and lo < hi
        if ok:
            s, e = fit_length(s, e, focus, lo, hi, cfg)
            s, e = snap_edges(s, e, lo, hi, transcript.words, sentences, cfg.top.max_len_s)
            ok = s <= focus <= e and e - s >= min(cfg.top.min_len_s, r.duration) * 0.9
        if not ok or any(s < m.end and m.start < e for m in out):
            s, e = r.start, r.end  # its own window never overlaps another region's
            if not s <= focus <= e:
                focus = r.peak
        used.add(k)
        out.append(Moment(len(out) + 1, s, e, focus, _clean(p.get("title")) or _fallback_title(r),
                          _clean(p.get("why")) or _fallback_why(r), _clean(p.get("text_hook")), r))
        if len(out) >= count:
            break
    for k, r in enumerate(regions):  # Claude returned fewer than asked: fill in measured order
        if len(out) >= count:
            break
        if k not in used:
            out.append(Moment(len(out) + 1, r.start, r.end, r.peak, _fallback_title(r), _fallback_why(r), "", r))
    return out


def _clean(text) -> str:
    return " ".join(str(text or "").split())[:300]


def rank(llm: LLM | None, regions: list[Region], transcript: Transcript, title: str, duration: float,
         cfg: Config, has_heat: bool, count: int, *, console=None) -> TopResult:
    notes: list[str] = []
    if len(regions) < count:
        notes.append(f"Only {len(regions)} distinct moment(s) of {cfg.top.min_len_s:.0f}-{cfg.top.max_len_s:.0f} s "
                     f"fit in this video, so you get {len(regions)} instead of {count}.")
    if not has_heat:
        notes.append("This video has no YouTube \"Most replayed\" data (too new, too few views, or not a YouTube "
                     "link), so the moments were found from the audio, the transcript and Claude's read only.")
    if llm is not None and cfg.top.use_claude and regions:
        try:
            res = llm.call(**ranking_request(regions, transcript, title, duration, cfg, has_heat, count),
                           max_tokens=8000)
            moments = apply_picks(res.data["picks"], regions, transcript, cfg, count)
            if res.fallback_used:
                notes.append(f"Claude Code answered with {res.served_model} instead of {cfg.judging.model} "
                             f"for the ranking.")
            return TopResult(moments, regions, "claude", has_heat, notes, res.served_model)
        except LLMError as e:
            error = str(e)
            # Refusals and unusable answers are cached on the API backend, so asking again gives the same.
            final = e.kind in ("refusal", "schema") and (e.cached or llm.backend != "claude-code")
            if console:
                console.print(f"  [yellow]Claude couldn't rank the moments ({error}).[/yellow]")
            moments = [Moment(k + 1, r.start, r.end, r.peak, _fallback_title(r), _fallback_why(r), "", r)
                       for k, r in enumerate(regions[:count])]
            notes.append(f"Claude couldn't rank the moments ({error}); they are in measured order instead.")
            if final:
                notes.append("Asking again returns the same answer; to make Claude rank afresh, re-scan from the "
                             "command line with --rerun judging.")
            return TopResult(moments, regions, "signals", has_heat, notes, error=error, error_final=final)
    moments = [Moment(k + 1, r.start, r.end, r.peak, _fallback_title(r), _fallback_why(r), "", r)
               for k, r in enumerate(regions[:count])]
    return TopResult(moments, regions, "signals", has_heat, notes)
