"""Bits shared by the CLI parsers (kept import-light so `clip-scanner --help` stays fast)."""

# "anthropic" is an alias of "api"; "scripted" is the offline test judge.
JUDGE_BACKENDS = ["claude-code", "api", "anthropic", "scripted"]
BACKEND_LABEL = {"claude-code": "your Claude subscription, through Claude Code (no API credits)",
                 "anthropic": "the Claude API (ANTHROPIC_API_KEY, billed per token)",
                 "scripted": "the SCRIPTED TEST JUDGE (verdicts are not real)"}


def resolve_backend(cfg, judge_backend: str | None) -> str:
    """CLI/UI choice, else config.toml. Returns the LLM backend name ("api" maps to "anthropic")."""
    name = judge_backend or cfg.judging.backend
    return {"api": "anthropic"}.get(name, name)
