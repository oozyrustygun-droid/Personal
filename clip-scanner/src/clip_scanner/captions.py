"""Word-by-word burned-in captions as an ASS subtitle file (rendered by ffmpeg/libass)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import CaptionsCfg


@dataclass
class TimedWord:
    start: float  # output-clip time
    end: float
    text: str


_STRIP = re.compile(r"^[\"'“”‘’(\[]+|[\"'“”‘’)\],.;:]+$")


def clean_word(text: str, uppercase: bool) -> str:
    t = _STRIP.sub("", text.strip())
    t = t.replace("\\", "").replace("{", "(").replace("}", ")")
    return t.upper() if uppercase else t


def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def caption_events(words: list[TimedWord], cfg: CaptionsCfg, clip_end: float) -> list[tuple[float, float, str]]:
    """Group words (words_per_caption) and time each group from its first word until the next group
    starts, but never longer than max_word_hold_s past its last word, and never past the clip end."""
    words = [w for w in words if clean_word(w.text, cfg.uppercase)]
    groups = [words[i: i + cfg.words_per_caption] for i in range(0, len(words), cfg.words_per_caption)]
    events = []
    for k, g in enumerate(groups):
        start = g[0].start
        nxt = groups[k + 1][0].start if k + 1 < len(groups) else clip_end
        end = min(nxt, g[-1].end + cfg.max_word_hold_s, clip_end)
        end = max(end, min(start + 0.08, clip_end))
        if end <= start:
            continue
        events.append((start, end, " ".join(clean_word(w.text, cfg.uppercase) for w in g)))
    return events


_NON_BMP = re.compile(r"[\U00010000-\U0010FFFF\uFE0F\u200D]")


def clean_title(text: str) -> str:
    """The on-screen text hook as libass can draw it: no emojis (the caption font has none), no ASS codes."""
    t = _NON_BMP.sub("", text).replace("\\", "").replace("{", "(").replace("}", ")")
    return " ".join(t.split())


def build_ass(words: list[TimedWord], cfg: CaptionsCfg, width: int, height: int, clip_end: float,
              title: str = "") -> str:
    """title: the on-screen text hook, shown at the top for the whole clip (smart-wrapped)."""
    bold = -1 if cfg.bold else 0
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,{cfg.font},{cfg.font_size},{cfg.primary_color},&H000000FF,{cfg.outline_color},&H80000000,{bold},0,0,0,100,100,0,0,1,{cfg.outline},{cfg.shadow},2,60,60,{cfg.margin_v},1
Style: Hook,{cfg.font},{cfg.hook_font_size},{cfg.primary_color},&H000000FF,{cfg.outline_color},&H80000000,{bold},0,0,0,100,100,0,0,1,{cfg.outline},{cfg.shadow},8,70,70,{cfg.hook_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    pop = r"{\fscx82\fscy82\t(0,90,\fscx100\fscy100)}"
    lines = [f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Word,,0,0,0,,{pop}{text}"
             for s, e, text in caption_events(words, cfg, clip_end)]
    title = clean_title(title)
    if title:
        lines.insert(0, f"Dialogue: 1,{ass_time(0.0)},{ass_time(clip_end)},Hook,,0,0,0,,{{\\q0}}"
                        f"{title.upper() if cfg.uppercase else title}")
    return header + "\n".join(lines) + "\n"
