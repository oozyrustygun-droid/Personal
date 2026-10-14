"""Offline, deterministic stand-in for Claude, used by the end-to-end test and for dry plumbing runs.

It is NOT a judge. It returns schema-valid responses driven by a small rule file, so the full
pipeline (gates → panel → devil's advocate → report → export) can be exercised without an API
key. Rules (JSON):
    {
      "great_phrases": ["never eating that again"],   # clip text containing one -> great_score
      "great_score": 9.2,
      "default_score": 6.0,
      "gate_failures": {"standalone": ["as i said"]},  # phrase -> that gate fails
      "devil_convincing_phrases": [],                    # phrase -> arbiter is convinced
      "hook_fail_openers": ["so anyway"],                # clip OPENS with this -> hook gate fails
      "persona_offsets": {"jaded_scroller": -0.2}       # per-persona score tweak
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import GATE_NAMES, RUBRIC_CRITERIA
from .llm import ScriptFn

DEFAULT_RULES = {"great_phrases": [], "great_score": 9.2, "default_score": 6.0, "gate_failures": {},
                 "devil_convincing_phrases": [], "hook_fail_openers": [], "persona_offsets": {}}


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def load_script(path: str | None) -> ScriptFn:
    rules = dict(DEFAULT_RULES)
    if path:
        rules.update(json.loads(Path(path).read_text()))
    return make_script(rules)


def make_script(rules: dict) -> ScriptFn:
    rules = {**DEFAULT_RULES, **rules}

    def has(text: str, phrases: list[str]) -> bool:
        t = _norm(text)
        return any(_norm(p) in t for p in phrases)

    def judge(meta: dict) -> dict:
        text = meta.get("clip_text", "")
        score = rules["great_score"] if has(text, rules["great_phrases"]) else rules["default_score"]
        score = max(0.0, min(10.0, score + rules["persona_offsets"].get(meta.get("persona", ""), 0.0)))
        gates = {}
        for g in GATE_NAMES:
            fail = has(text, rules["gate_failures"].get(g, []))
            if g == "hook_in_first_2s" and any(_norm(text).startswith(_norm(o)) for o in rules["hook_fail_openers"]):
                fail = True
            gates[g] = {"evidence": f"scripted: {'matched failure phrase' if fail else 'ok'}", "passed": not fail}
        scores = {c: {"justification": "scripted", "score": score} for c in RUBRIC_CRITERIA}
        great = score >= 8.5 and all(v["passed"] for v in gates.values())
        first = " ".join(text.split()[:8])
        return {"text_hook": f"Joe: {' '.join(text.split()[:6])}", "first_two_seconds": first, "gates": gates, "scores": scores, "biggest_weakness": "scripted",
                "overall_justification": "scripted verdict", "overall": score, "hook_line": first,
                "verdict": "GREAT" if great else "NOT_GREAT"}

    def script(purpose: str, meta: dict) -> dict:
        text = meta.get("clip_text", "")
        if purpose == "judge":
            return judge(meta)
        if purpose == "devils_advocate":
            return {"strongest_flop_argument": "scripted", "predicted_outcome": "scripted",
                    "reasons": [{"category": "other", "argument": "scripted", "severity": "minor"}]}
        if purpose == "arbiter":
            convinced = has(text, rules["devil_convincing_phrases"])
            return {"strongest_point_for": "scripted", "strongest_point_against": "scripted",
                    "analysis": "scripted", "decisive_issue": "scripted flaw" if convinced else "none",
                    "argument_convincing": convinced, "survival_confidence": 3.0 if convinced else 8.5}
        if purpose == "packaging":
            words = text.split()
            return {"hook_line": " ".join(words[:10]), "why_viral": "scripted packaging",
                    "caption_title": " ".join(words[:6]).upper(), "posting_hook_text": " ".join(words[:12]),
                    "hashtags": ["#joebartolozzi", "#fyp"]}
        if purpose == "diagnosis":
            return {"one_line_read": "scripted: no candidate cleared the bar"}
        if purpose == "proposer":
            return {"moments": rules.get("proposals", [])}
        raise ValueError(f"scripted backend has no rule for purpose {purpose!r}")

    script.fingerprint = json.dumps(rules, sort_keys=True)  # part of the response-cache key
    return script
