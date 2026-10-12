"""Optional recall helper: a cheap model reads the transcript and flags candidate moments.

The heuristic signals catch loud moments. This pass catches deadpan, verbal or
story moments that are funny on paper but acoustically flat. It only proposes; it never decides.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .boundaries import split_sentences
from .config import Config, load_style_guide
from .llm import LLM, LLMError
from .transcribe import Transcript

PROPOSER_SYSTEM = """You are the recall stage of a clip-finding tool for a Joe Bartolozzi clipping account \
(TikTok / YouTube Shorts / Reels). You read a time-stamped transcript chunk of a long stream or video and \
flag moments that MIGHT become a strong 10-60 second vertical clip: explosive reactions, punchlines, \
absurd lines, confrontations with chat, blunt verdicts, story payoffs, jumpscares, laugh fits.

Rules:
- Propose up to 8 moments per chunk. Propose zero if nothing qualifies, because padding wastes money downstream.
- start_s is where a clip could begin (on a sentence that hooks), end_s is just after the payoff lands,
  payoff_s is the exact peak (the punchline word, the scream, the verdict).
- Use the timestamps given; never invent times outside the chunk.
- strength: 1 = maybe, 3 = solid, 5 = could go viral. Be honest; most moments are 1-3.

Style guide for this creator:
"""


class Moment(BaseModel):
    start_s: float
    end_s: float
    payoff_s: float
    kind: Literal["reaction", "punchline", "confrontation", "verdict", "story", "jumpscare", "laugh", "absurd",
                  "other"]
    why: str
    strength: int = Field(ge=1, le=5)


class Proposals(BaseModel):
    moments: list[Moment]


def transcript_lines(transcript: Transcript, start: float, end: float, pause_break_s: float) -> str:
    words = transcript.words_between(start, end)
    lines = []
    for s in split_sentences(words, pause_break_s):
        text = " ".join(w.text for w in words[s.i0: s.i1 + 1])
        spk = words[s.i0].speaker
        lines.append(f"[{s.start:.1f}]{f' {spk}:' if spk else ''} {text}")
    return "\n".join(lines)


def propose(llm: LLM, transcript: Transcript, duration: float, cfg: Config, *,
            console=None) -> tuple[list[Moment], bool]:
    """Returns (moments, complete). complete=False if any chunk failed for a retryable reason."""
    cc = cfg.candidates
    system = [{"type": "text", "text": PROPOSER_SYSTEM + load_style_guide(),
               "cache_control": {"type": "ephemeral"}}]
    out: list[Moment] = []
    complete = True
    step = max(60.0, cc.proposer_chunk_s - 30.0)
    t = 0.0
    while t < duration:
        lo, hi = t, min(duration, t + cc.proposer_chunk_s)
        lines = transcript_lines(transcript, lo, hi, cc.pause_break_s)
        t += step
        if len(lines) < 80:
            continue
        msg = (f"Transcript chunk {lo:.0f}s-{hi:.0f}s (timestamps are seconds from the start of the video):\n\n"
               f"{lines}\n\nReturn the moments as JSON.")
        try:
            res = llm.call(purpose="proposer", model=cc.proposer_model, system=system,
                           messages=[{"role": "user", "content": msg}], output=Proposals,
                           effort=cc.proposer_effort, max_tokens=8000,
                           meta={"chunk": [lo, hi], "lines": lines})
        except LLMError as e:
            if console:
                console.print(f"  [yellow]proposer skipped chunk {lo:.0f}-{hi:.0f}s: {e}[/yellow]")
            if e.kind not in ("refusal", "schema"):  # deterministic failures won't improve on retry
                complete = False
            if e.kind == "auth":
                break
            continue
        for m in Proposals.model_validate(res.data).moments:
            if lo - 1 <= m.start_s < m.end_s <= hi + 1 and m.start_s <= m.payoff_s <= m.end_s + 1:
                out.append(m)
    return out, complete
