"""Prompt construction for judges, devil's advocate, arbiter, packaging and diagnosis.

Cache layout. Every role shares the same two cached system blocks:
  [1] the judging handbook + creator style guide (identical across videos)   cache_control
  [2] the source video's full transcript (identical across calls for a video) cache_control
  [3] the role / persona instructions (small, uncached)
The per-clip packet (clip transcript, context, measured timeline, keyframes) goes in the user turn.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from ..boundaries import split_sentences
from ..candidates import Candidate
from ..config import Config
from ..signals import Signals
from ..transcribe import Transcript, Word
from ..ui import fmt_ts

HANDBOOK = """You are part of the quality-control panel for a short-form clipping account that posts \
Joe Bartolozzi moments to TikTok, YouTube Shorts and Instagram Reels.

The account's one rule: post only GREAT clips. Posting a mediocre clip costs more than missing a \
great one. It burns the audience's trust and teaches the algorithm to show the account less. Most \
windows cut from a long stream are NOT great, and a whole stream with zero great clips is normal. \
Every clip is judged alone, against an absolute bar, never against the other moments in the same \
video. "The best part of a weak stream" is not a reason to pass anything.

## What you receive for a clip
- The clip transcript in clip time (0.0s = the first frame the viewer sees). A [JUMP CUT] line marks \
where silence was removed.
- Context before and after the clip, marked NOT IN CLIP. Viewers never see it. Use it only to detect \
setup the clip silently depends on (then it is not standalone) or a payoff that happens after the clip \
ends (then it has no clear payoff).
- A tool-measured audio/visual timeline: loudness spikes, laughter, screams, crowd, music, scene cuts, \
dead air, jump cuts.
- Keyframes labelled with clip time.
You cannot hear audio or see motion. The measured timeline is your ears. Do not assume anything the \
transcript, timeline and frames do not support, and resolve missing or ambiguous evidence against the clip.

## Hard gates (one failure rejects the clip, whatever its scores)
1. hook_in_first_2s: within the first 1.5-2 seconds there is a line, reaction or visual that stops a \
scrolling viewer. Fail on any slow wind-up: greetings, "so anyway", "okay chat", quietly reading chat, \
silence, plain gameplay, or setup whose interest only arrives later.
2. standalone: a viewer with zero context (never saw the stream, may not know Joe) understands what \
is happening and why it is funny or shocking. Fail if it depends on earlier events, unexplained names, \
running bits, or a stimulus the viewer can't see or hear in the clip.
3. clear_payoff: the payoff (punchline, reveal, absurd moment, strong verdict, emotional peak) is in \
the clip and fully lands before the clip ends. Fail if it ends on setup, cuts the payoff or reaction \
short, trails off, or the real payoff happens after the end (check the NOT IN CLIP context after).
4. no_dead_air: no stretch over about 1.5 s where nothing happens (no speech, no reaction, no meaningful \
visual action). Use the measured timeline.
5. monetization_safe: nothing that gets a clip demonetized, claimed, muted or struck. That includes \
audible copyrighted music under the moment (check the music measurement), another creator's \
video/TV/song carrying the moment, slurs (even read from chat), sexual content, graphic drug use, \
dangerous acts, doxxing, harassment of a private person, and gambling promotion.

## Rubric (each 0-10)
- hook_strength: how hard the first 2 seconds stop the scroll.
- payoff: how hard the payoff hits and how cleanly it lands.
- emotional_intensity: strength of the laugh, shock, disgust or awe.
- shareability: would a typical viewer send this to a friend or group chat?
- comment_bait: does it make people comment (quote it, argue, pick a side, tag someone)?
- rewatch_loop: rewatch value; does the ending flow back into the start?
- joe_signature: unmistakably the Joe his audience loves (see the style guide), not a generic streamer moment.
- visual_strength: in a 9:16 crop, is Joe's face or reaction visible, is the stimulus visible, is the action readable?

## Absolute scale (for every score and for overall)
10 = account-defining, meme-template moment. 9 = instantly quotable, near-certain breakout. \
8.5 = THE BAR: you would bet it clearly beats the account's median Short. 7-8 = solid, fine for fans, \
NOT great. 5-6 = typical stream moment. 4 or below = filler or unusable.
Be calibrated. A judge who rates most candidates 8+ is useless. Do not raise a score because it is \
the best moment in this video, because Joe is generally funny, or because it "could work with \
better editing". Judge the clip as it would be posted: these exact boundaries, burned-in captions, \
and no other edits.

# Creator style guide
"""

PERSONAS: dict[str, tuple[str, str]] = {
    "jaded_scroller": (
        "The jaded scroller",
        "You are 22 and scroll TikTok and Shorts about three hours a day. You have seen every streamer-reaction "
        "format a thousand times and you swipe within a second unless something grabs you. You know of "
        "Joe vaguely but you are not a fan. You send a clip to the group chat only when it genuinely made "
        "you laugh out loud or say 'what' out loud. Judge from the feed: would this stop your thumb, hold you "
        "to the end, and make you send it or comment? Loud is not the same as funny. You have seen a "
        "thousand streamers scream.",
    ),
    "joe_superfan": (
        "The Joe superfan",
        "You have watched Joe since the green-screen TikTok days and you know every bit: 'type in chat', "
        "'L chatter', the jumpscare screams, the food verdicts, the chat confrontations. Because you love "
        "Joe, you protect the account's reputation. Posting mid Joe clips trains fans and the algorithm "
        "to scroll past. You rate GREAT only the moments the fandom would clip, quote and meme. Penalize "
        "generic moments any streamer could produce, and anything that only lands if you were in the stream.",
    ),
    "shorts_editor": (
        "The viral-shorts editor",
        "You are a professional short-form editor who has cut thousands of streamer clips for accounts with "
        "millions of followers, and you read retention graphs every day. You judge mechanics: does the first "
        "frame and line hook, does retention hold with no sag, does the payoff land inside the clip, can it "
        "loop, will burned-in captions read well, is the audio clean and safe from claims, does a vertical "
        "crop keep the face and stimulus readable, and is the length right for the payoff? You know which "
        "clips the algorithm pushes and which die at 400 views.",
    ),
}

JUDGE_OUTPUT = """
## Your task
Judge the clip in the user message from your persona's perspective, following every rule above. You \
judge independently and will never see the other judges' opinions.
Fill every JSON field:
- first_two_seconds: literally what the viewer sees and hears in seconds 0-2.
- gates: for each, the evidence (quote with clip time, or a frame), then passed.
- scores: justification (1-3 sentences), then score.
- biggest_weakness: the single most likely reason this clip underperforms.
- overall_justification, then overall: your holistic 0-10. It must be consistent with your rubric \
scores and must not exceed what they support.
- hook_line: the opening line or beat, verbatim when it is speech.
- verdict: GREAT only if every gate passes, overall >= 8.5, and you would stake the account's \
reputation on it. Otherwise NOT_GREAT."""

DEVIL_ROLE = """
# Your role: devil's advocate
The panel has rated the clip in the user message GREAT. Your only job is to make the strongest \
honest case that it will FLOP, meaning it will underperform the account's median Short. Hunt for real weaknesses: \
a first second that doesn't stop the scroll, context it silently depends on, a payoff weaker than the \
panel thinks, energy sag, a format seen a thousand times, audio or claim risk, a broken loop, a face that \
gets lost in a vertical crop, the wrong length. Use only facts supported by the transcript, timeline and \
frames; never invent problems. Rate each reason's severity honestly (minor, major, fatal). A weak case \
stated honestly is more useful than a strong case made up."""

ARBITER_ROLE = """
# Your role: independent arbiter
You rule on one clip. You get the clip itself, the panel's case FOR it, and the devil's advocate's \
case AGAINST it. Decide whether the case against is convincing: does it identify a real, material \
flaw that a typical viewer would actually experience, and that drops the clip below the 8.5 bar? \
Check every claim against the clip; dismiss claims the evidence does not support. Remember the account's \
rule: a mediocre clip that gets posted is worse than a great clip that is skipped. If you are unsure the clip \
still clears the bar, it does not survive: set argument_convincing true or give a low survival_confidence.
survival_confidence is your 0-10 confidence that the clip is still GREAT after hearing both sides."""

PACKAGING_ROLE = """
# Your role: packaging
The clip in the user message passed the panel and survived the devil's advocate. Write its packaging:
- hook_line: one line describing the opening beat that stops the scroll.
- why_viral: one or two sentences on why it will travel (specific, not generic hype).
- caption_title: the on-screen caption / title, AT MOST 8 words, in the account's voice (punchy, \
no hashtags, no emojis unless they add meaning).
- posting_hook_text: the first line of the post description, written to pull the viewer in.
- hashtags: 3-6 relevant hashtags (include #joebartolozzi)."""

DIAGNOSIS_ROLE = """
# Your role: post-mortem
No clip in this video met the GREAT bar. From the summary of candidates and why each was rejected, \
write ONE sentence on why this video lacks great clip material (for example "a low-energy chat-reading \
stream where every peak depends on earlier context"). Be specific to this video."""


def handbook(style_guide: str) -> str:
    return HANDBOOK + style_guide


def video_block(title: str, uploader: str, duration: float, transcript: Transcript, cfg: Config) -> str:
    lines = [f"# Source video (reference only; clip viewers never see this)",
             f"Title: {title}", f"Uploader: {uploader or 'unknown'}", f"Duration: {fmt_ts(duration)}",
             "Full transcript with source timestamps:", ""]
    words = transcript.words
    for s in split_sentences(words, cfg.candidates.pause_break_s):
        spk = words[s.i0].speaker
        text = " ".join(w.text for w in words[s.i0: s.i1 + 1])
        lines.append(f"[{fmt_ts(s.start)}]{f' {spk}:' if spk else ''} {text}")
    return "\n".join(lines)


def role_block(role: str) -> str:
    if role in PERSONAS:
        name, desc = PERSONAS[role]
        return f"\n# Your persona: {name}\n{desc}\n{JUDGE_OUTPUT}"
    return {"devils_advocate": DEVIL_ROLE, "arbiter": ARBITER_ROLE, "packaging": PACKAGING_ROLE,
            "diagnosis": DIAGNOSIS_ROLE}[role]


def system_blocks(handbook_text: str, video_text: str, role: str, ttl: str) -> list[dict]:
    cc = {"type": "ephemeral"} if ttl == "5m" else {"type": "ephemeral", "ttl": "1h"}
    return [
        {"type": "text", "text": handbook_text, "cache_control": cc},
        {"type": "text", "text": video_text, "cache_control": cc},
        {"type": "text", "text": role_block(role)},
    ]


# ---- per-clip packet ------------------------------------------------------------------------

def _clip_series(c: Candidate, sig: Signals, name: str) -> np.ndarray:
    return np.concatenate([sig.window(name, s, e) for s, e in c.segments]) if c.segments else np.zeros(0)


def _runs(x: np.ndarray, thr: float, hop: float, min_len: float = 0.2) -> list[tuple[float, float, float]]:
    out, start = [], None
    for i, v in enumerate(np.append(x, -np.inf)):
        if v >= thr and start is None:
            start = i
        elif v < thr and start is not None:
            if (i - start) * hop >= min_len:
                out.append((start * hop, i * hop, float(x[start:i].max())))
            start = None
    return out


def _fmt_runs(runs, with_peak=True) -> str:
    if not runs:
        return "none"
    parts = [f"{a:.1f}-{b:.1f}s" + (f" ({p:.2f})" if with_peak else "") for a, b, p in runs[:8]]
    return ", ".join(parts) + (f" (+{len(runs) - 8} more)" if len(runs) > 8 else "")


def timeline_text(c: Candidate, sig: Signals, measurements: dict) -> str:
    hop = sig.hop
    lines = ["MEASURED TIMELINE (clip time, from audio/video analysis):"]
    if sig.events_available:
        for name, label, thr in (("laughter", "Laughter", 0.3), ("scream", "Screams/shouts", 0.3),
                                 ("crowd", "Crowd/cheering", 0.3), ("gasp", "Gasps", 0.3),
                                 ("music", "Music", 0.5)):
            lines.append(f"- {label}: {_fmt_runs(_runs(_clip_series(c, sig, name), thr, hop))}")
    else:
        lines.append("- Audio events (laughter/music): NOT MEASURED")
    lines.append(f"- Loudness spikes above Joe's normal level: "
                 f"{_fmt_runs(_runs(_clip_series(c, sig, 'loud_spike'), 0.6, hop), with_peak=False)}")
    cuts = [c.to_content(t) for t in sig.cuts_between(c.start, c.end)
            if any(s <= t <= e for s, e in c.segments)]
    lines.append(f"- Scene cuts: {', '.join(f'{t:.1f}s' for t in cuts) or 'none'}")
    dead = _runs(_clip_series(c, sig, "word_active") < 0.5, 0.5, hop, min_len=0.8)
    lines.append(f"- Pauses without speech >= 0.8s: {_fmt_runs(dead, with_peak=False)}")
    if c.removed_gaps:
        jc = [f"{c.to_content(a):.1f}s ({b - a:.1f}s removed)" for a, b in c.removed_gaps]
        lines.append(f"- Jump cuts (silence removed): {', '.join(jc)}")
    m = measurements
    onset = "n/a" if m["first_activity_at"] is None else f"{m['first_activity_at']:.1f}s"
    music = f"{m['music_fraction']:.0%} of the clip" if sig.events_available else "not measured"
    lines.append(f"- Duration {m['duration']:.1f}s; first speech/reaction at {onset}; "
                 f"longest dead air {m['longest_dead_air']:.1f}s; music present: {music}")
    return "\n".join(lines)


def _sentence_lines(words: list[Word], to_clip, pause_break_s: float, prefix: str = "") -> list[str]:
    out = []
    for s in split_sentences(words, pause_break_s):
        spk = words[s.i0].speaker
        text = " ".join(w.text for w in words[s.i0: s.i1 + 1])
        out.append(f"[{prefix}{to_clip(s.start):.1f}s]{f' {spk}:' if spk else ''} {text}")
    return out


def clip_transcript_text(c: Candidate, transcript: Transcript, cfg: Config) -> str:
    lines = []
    for k, (s, e) in enumerate(c.segments):
        seg_words = [w for w in transcript.words if s - 1e-3 <= w.start and w.end <= e + 1e-3]
        lines += _sentence_lines(seg_words, c.to_content, cfg.candidates.pause_break_s)
        if k + 1 < len(c.segments):
            gap = c.segments[k + 1][0] - e
            lines.append(f"[JUMP CUT at {c.to_content(e):.1f}s: {gap:.1f}s of silence removed]")
    return "\n".join(lines) if lines else "(no speech)"


def context_text(c: Candidate, transcript: Transcript, cfg: Config) -> tuple[str, str]:
    w = cfg.judging.context_window_s
    before = [x for x in transcript.words if c.start - w <= x.start and x.end <= c.start + 1e-3]
    after = [x for x in transcript.words if c.end - 1e-3 <= x.start and x.end <= c.end + w]
    b = _sentence_lines(before, lambda t: t - c.start, cfg.candidates.pause_break_s) if before else []
    a = _sentence_lines(after, lambda t: t - c.end, cfg.candidates.pause_break_s, prefix="+") if after else []
    return ("\n".join(b) or "(nothing)"), ("\n".join(a) or "(nothing)")


def image_block(path: Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def clip_packet(c: Candidate, transcript: Transcript, sig: Signals, measurements: dict, frames: list,
                cfg: Config) -> list[dict]:
    before, after = context_text(c, transcript, cfg)
    header = (f"CLIP {c.id}: {c.duration:.1f}s as posted (source {fmt_ts(c.start)}-{fmt_ts(c.end)}"
              f"{f', {c.jump_cuts} jump cut(s)' if c.jump_cuts else ''}).\n\n"
              f"CONTEXT BEFORE (NOT IN CLIP; seconds relative to clip start):\n{before}\n\n"
              f"CLIP TRANSCRIPT (clip time):\n{clip_transcript_text(c, transcript, cfg)}\n\n"
              f"CONTEXT AFTER (NOT IN CLIP; seconds after the clip ends):\n{after}\n\n"
              f"{timeline_text(c, sig, measurements)}\n\nKEYFRAMES:")
    content: list[dict] = [{"type": "text", "text": header}]
    for k in frames:
        content.append({"type": "text", "text": f"Frame at {k.clip_t:.1f}s ({k.label}):"})
        content.append(image_block(k.path))
    return content
