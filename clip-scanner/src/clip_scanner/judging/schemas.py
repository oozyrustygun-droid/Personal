"""Structured outputs for every judging call.

Field order is deliberate: evidence and justification come before the verdict field they
support, so the model commits to reasons before numbers. Bounds (0-10) are enforced again
client-side, because the API's schema subset has no numeric constraints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class GateCheck(BaseModel):
    evidence: str = Field(description="Concrete evidence from the clip (quote + clip time, or frame) for the call.")
    passed: bool


class Gates(BaseModel):
    hook_in_first_2s: GateCheck
    standalone: GateCheck
    clear_payoff: GateCheck
    no_dead_air: GateCheck
    monetization_safe: GateCheck


class CriterionScore(BaseModel):
    justification: str
    score: float = Field(ge=0, le=10)


class Scores(BaseModel):
    hook_strength: CriterionScore
    payoff: CriterionScore
    emotional_intensity: CriterionScore
    shareability: CriterionScore
    comment_bait: CriterionScore
    rewatch_loop: CriterionScore
    joe_signature: CriterionScore
    visual_strength: CriterionScore


class JudgeVerdict(BaseModel):
    text_hook: str = Field("", description="The short on-screen text hook the account would put over the opening "
                                           "(at most 10 words, true to the clip, no spoiler of the payoff).")
    first_two_seconds: str = Field(description="Exactly what a viewer sees and hears in the first 2 seconds.")
    gates: Gates
    scores: Scores
    biggest_weakness: str
    overall_justification: str
    overall: float = Field(ge=0, le=10)
    hook_line: str = Field(description="The one-line hook: the opening line or beat that stops the scroll.")
    verdict: Literal["GREAT", "NOT_GREAT"]


class FlopReason(BaseModel):
    category: Literal["hook", "context", "payoff", "energy", "pacing", "dead_air", "monetization", "copyright",
                      "overdone", "audience", "visual", "audio", "other"]
    argument: str
    severity: Literal["minor", "major", "fatal"]


class DevilsCase(BaseModel):
    strongest_flop_argument: str
    reasons: list[FlopReason]
    predicted_outcome: str


class ArbiterRuling(BaseModel):
    strongest_point_for: str
    strongest_point_against: str
    analysis: str
    decisive_issue: str
    argument_convincing: bool
    survival_confidence: float = Field(ge=0, le=10,
                                       description="0-10: confidence the clip is still GREAT after the critique.")


class Packaging(BaseModel):
    hook_line: str = Field(description="One line: the opening beat that stops the scroll.")
    why_viral: str = Field(description="One or two sentences: why this will travel.")
    caption_title: str = Field(description="On-screen caption / title, at most 8 words.")
    posting_hook_text: str = Field(description="First line of the post description, written to pull the viewer in.")
    hashtags: list[str]

    @field_validator("caption_title")
    @classmethod
    def _max_8_words(cls, v: str) -> str:
        words = v.split()
        return " ".join(words[:8])


class Diagnosis(BaseModel):
    one_line_read: str = Field(description="One sentence: why this video lacks (or has few) clip-worthy moments.")
