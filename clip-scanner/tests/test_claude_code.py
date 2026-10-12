"""The Claude Code (subscription) backend, against a fake `claude` executable (tests/fake_claude.py).

Checks the exact command we run (tools off, no settings/CLAUDE.md/MCP, our system prompt, schema),
that API keys never reach Claude Code, and that every login/billing/limit problem fails closed and
stops the run instead of spending money or passing a clip.
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from clip_scanner.config import Config
from clip_scanner.cost import CostTracker
from clip_scanner.llm import LLM, LLMError

from test_gates import make_signals, talk

FAKE = Path(__file__).parent / "fake_claude.py"


class Out(BaseModel):
    reason: str
    score: float = Field(ge=0, le=10)


def _gate(ok=True):
    return {"evidence": "quote at 0:01", "passed": ok}


def _crit(score):
    return {"justification": "because", "score": score}


def verdict(score=9.5, great=True):
    return {"first_two_seconds": "Joe bites and screams", "gates": {g: _gate() for g in (
        "hook_in_first_2s", "standalone", "clear_payoff", "no_dead_air", "monetization_safe")},
            "scores": {c: _crit(score) for c in ("hook_strength", "payoff", "emotional_intensity", "shareability",
                                                 "comment_bait", "rewatch_loop", "joe_signature", "visual_strength")},
            "biggest_weakness": "short setup", "overall_justification": "huge reaction", "overall": score,
            "hook_line": "WORST THING I'VE EVER EATEN", "verdict": "GREAT" if great else "NOT_GREAT"}


RESPONSES = {
    "reason": {"reason": "ok", "score": 7.0},
    "first_two_seconds": verdict(),
    "strongest_flop_argument": {"strongest_flop_argument": "weak", "predicted_outcome": "does fine",
                                "reasons": [{"category": "pacing", "argument": "slow", "severity": "minor"}]},
    "strongest_point_for": {"strongest_point_for": "reaction", "strongest_point_against": "pacing",
                            "analysis": "holds", "decisive_issue": "none", "argument_convincing": False,
                            "survival_confidence": 9.0},
    "hook_line": {"hook_line": "WORST THING", "why_viral": "reaction", "caption_title": "Joe regrets this bite",
                  "posting_hook_text": "he was not ready", "hashtags": ["#joebartolozzi"]},
    "one_line_read": {"one_line_read": "Nothing landed."},
    "moments": {"moments": []},
}


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    exe = tmp_path / "bin" / "claude"
    exe.parent.mkdir()
    exe.write_text(f"#!{sys.executable}\n" + FAKE.read_text())
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "calls.jsonl"
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps(RESPONSES))
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setenv("FAKE_CLAUDE_RESPONSES", str(responses))
    monkeypatch.setenv("FAKE_CLAUDE_AUTH", "claude.ai")
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")
    # An API key in the user's shell must never reach Claude Code (it would bill the key).
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-should-not-leak")

    class F:
        path = exe
        tmp = tmp_path

        @staticmethod
        def calls():
            return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []

        @staticmethod
        def mode(m):
            monkeypatch.setenv("FAKE_CLAUDE_MODE", m)

        @staticmethod
        def auth(a):
            monkeypatch.setenv("FAKE_CLAUDE_AUTH", a)

    return F


def make_llm(fake, tmp_path, **cc):
    cfg = Config.model_validate({"judging": {"claude_code": {"path": str(fake.path), **cc}}})
    tracker = CostTracker(cfg)
    llm = LLM(cfg, tmp_path / "llm", tracker, backend="claude-code")
    tracker.billing = llm.billing
    return llm, tracker


SYSTEM = [{"type": "text", "text": "HANDBOOK", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
          {"type": "text", "text": "ROLE"}]
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}


def call(llm, text="clip", **kw):
    return llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}, IMG]}],
                    output=Out, effort="high", max_tokens=16000, **kw)


def test_request_runs_on_subscription_with_everything_else_off(fake, tmp_path):
    llm, tracker = make_llm(fake, tmp_path)
    res = call(llm)
    assert res.data == {"reason": "ok", "score": 7.0}
    assert res.served_model == "claude-opus-5-5" and not res.fallback_used
    [c] = fake.calls()
    argv = c["argv"]

    def after(flag):
        return argv[argv.index(flag) + 1]

    assert after("--model") == "claude-opus-5-5" and after("--effort") == "high"
    assert after("--tools") == ""                 # no Bash/Read/Edit/web for the judges
    assert after("--setting-sources") == ""       # no hooks, apiKeyHelper or env from settings files
    for flag in ("--safe-mode", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in argv
    assert "--bare" not in argv                   # bare mode ignores the subscription login
    assert after("--input-format") == "stream-json" and after("--output-format") == "stream-json"
    assert json.loads(after("--json-schema"))["additionalProperties"] is False
    assert c["system"] == "HANDBOOK\n\nROLE"
    assert c["env"]["ANTHROPIC_API_KEY"] is None and c["env"]["ANTHROPIC_AUTH_TOKEN"] is None
    assert c["env"]["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "16000"
    assert c["env"]["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    content = c["message"]["message"]["content"]
    assert all("cache_control" not in b for b in content)  # Claude Code places its own cache breakpoints
    assert content[1] == IMG and c["message"]["type"] == "user"
    assert Path(c["cwd"]).name.startswith("clip-scanner-cc-")  # never the user's project directory
    # Billed to the plan: no dollars, but the size is kept.
    s = tracker.summary()
    assert s["billing"] == "subscription" and s["total_usd"] == 0 and s["api_equivalent_usd"] > 0
    # Cached on disk: a re-run doesn't call Claude Code again.
    again = call(llm)
    assert again.cached and again.data == res.data and len(fake.calls()) == 1


@pytest.mark.parametrize("auth, kind", [("api_key", "api_billing"), ("api_key_helper", "api_billing"),
                                        ("third_party", "api_billing"), ("none", "auth")])
def test_refuses_anything_but_a_subscription_login(fake, tmp_path, auth, kind):
    fake.auth(auth)
    llm, _ = make_llm(fake, tmp_path)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == kind
    assert fake.calls() == []                     # nothing was sent
    assert llm.fatal is not None
    with pytest.raises(LLMError) as again:        # and nothing ever will be this run
        call(llm, text="another clip")
    assert again.value.kind == kind and fake.calls() == []


def test_oauth_token_login_counts_as_subscription(fake, tmp_path):
    fake.auth("oauth_token")  # `claude setup-token`
    llm, _ = make_llm(fake, tmp_path)
    assert call(llm).data["score"] == 7.0


def test_run_that_reports_an_api_key_is_killed(fake, tmp_path):
    fake.mode("init_api_key")
    llm, _ = make_llm(fake, tmp_path, timeout_s=20)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == "api_billing" and llm.fatal is not None


def test_usage_limit_stops_the_run(fake, tmp_path):
    fake.mode("usage_limit")
    llm, _ = make_llm(fake, tmp_path)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == "usage_limit" and "resets" in str(e.value)
    with pytest.raises(LLMError):
        call(llm, text="next")
    assert len(fake.calls()) == 1


def test_extra_usage_stops_after_the_call(fake, tmp_path):
    fake.mode("overage")
    llm, _ = make_llm(fake, tmp_path)
    assert call(llm).data["score"] == 7.0         # already paid for: kept
    assert llm.fatal is not None and llm.fatal.kind == "extra_usage"
    with pytest.raises(LLMError):
        call(llm, text="next")
    assert len(fake.calls()) == 1
    llm2, _ = make_llm(fake, tmp_path / "b", allow_extra_usage=True)
    call(llm2)
    call(llm2, text="next")
    assert llm2.fatal is None


def test_other_model_answering_is_flagged(fake, tmp_path):
    fake.mode("fallback")
    llm, _ = make_llm(fake, tmp_path)
    res = call(llm)
    assert res.fallback_used and res.served_model == "claude-sonnet-5-5"


@pytest.mark.parametrize("mode, kind, fatal", [("auth_fail", "auth", True), ("no_structured", "invalid_json", False),
                                               ("schema_retries", "schema", False)])
def test_errors_fail_closed(fake, tmp_path, mode, kind, fatal):
    fake.mode(mode)
    llm, _ = make_llm(fake, tmp_path)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == kind
    assert (llm.fatal is not None) is fatal


def test_timeout_kills_the_run(fake, tmp_path):
    fake.mode("hang")
    llm, _ = make_llm(fake, tmp_path, timeout_s=1)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == "timeout" and llm.fatal is None


def test_missing_cli(tmp_path):
    cfg = Config.model_validate({"judging": {"claude_code": {"path": str(tmp_path / "nope" / "claude")}}})
    llm = LLM(cfg, tmp_path / "llm", CostTracker(cfg), backend="claude-code")
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == "not_installed" and "Install Claude Code" in str(e.value)


# -- the whole judging stage through the fake CLI --------------------------------------------------

def judging_context(llm):
    from clip_scanner.candidates import Candidate
    from clip_scanner.judging.runner import JudgeContext

    sig = make_signals(120.0)
    tr = talk(sig, [(10.0, 40.0), (60.0, 85.0)])
    cands = [Candidate(id=f"c{i}", segments=[(a, b)], peak=a + 5, origin="signal", score=1.0 - i / 10, reason="")
             for i, (a, b) in enumerate([(10.0, 40.0), (60.0, 85.0)])]
    ctx = JudgeContext(cfg=llm.cfg, llm=llm, transcript=tr, signals=sig, frames={}, handbook="HANDBOOK",
                       duration=120.0, title="t")
    return cands, ctx


def test_full_judging_on_the_subscription(fake, tmp_path):
    from clip_scanner.judging.runner import run_judging

    llm, tracker = make_llm(fake, tmp_path)
    cands, ctx = judging_context(llm)
    seen = []
    res = run_judging(cands, ctx, max_clips=5, on_estimate=lambda est: seen.append(est) or True)
    assert seen[0]["billing"] == "subscription" and seen[0]["expected_calls"] >= 1
    assert res.status == "complete", res.notes
    assert [d.candidate_id for d in res.great] == ["c0", "c1"]
    assert res.packaging["c0"]["caption_title"] == "Joe regrets this bite"
    purposes = [json.loads(c["argv"][c["argv"].index("--json-schema") + 1])["properties"] for c in fake.calls()]
    assert sum("first_two_seconds" in p for p in purposes) == 6  # 3 judges x 2 clips, all the configured model
    assert tracker.summary()["total_usd"] == 0


def test_usage_limit_mid_scan_is_unavailable_not_no_great(fake, tmp_path):
    from clip_scanner.judging.runner import run_judging

    fake.mode("usage_limit")
    llm, _ = make_llm(fake, tmp_path)
    cands, ctx = judging_context(llm)
    res = run_judging(cands, ctx, max_clips=5)
    assert res.status == "unavailable" and not res.great
    assert any("usage limit" in n for n in res.notes)
    assert len(fake.calls()) == 1                 # the second clip never hit Claude Code
    assert res.diagnosis == ""                    # no "this video has nothing" read from a failed run


def test_api_backend_still_needs_approval_but_subscription_does_not():
    from clip_scanner.judging.runner import needs_approval

    cfg = Config()
    est = {"uncached": 10, "expected_usd": 50.0, "billing": "api"}
    assert needs_approval(est, cfg)
    assert not needs_approval({**est, "billing": "subscription"}, cfg)


def test_backend_resolution_defaults_to_subscription():
    from clip_scanner.cli_common import resolve_backend

    assert Config().judging.backend == "claude-code"
    assert resolve_backend(Config(), None) == "claude-code"
    assert resolve_backend(Config(), "api") == "anthropic"
    assert resolve_backend(Config.model_validate({"judging": {"backend": "api"}}), None) == "anthropic"


@pytest.mark.skipif(os.environ.get("CLIP_SCANNER_LIVE_CC") != "1", reason="set CLIP_SCANNER_LIVE_CC=1 to use the real CLI")
def test_live_claude_code_smoke(tmp_path):
    """One tiny real request on Haiku through the installed `claude` (counts toward your plan)."""
    from clip_scanner.judging.schemas import Diagnosis

    cfg = Config()
    llm = LLM(cfg, tmp_path / "llm", CostTracker(cfg), backend="claude-code")
    res = llm.call(purpose="diagnosis", model="claude-haiku-5-5", system="Answer in one short sentence.",
                   messages=[{"role": "user", "content": "Why might a 2-hour food review have no clip-worthy moments?"}],
                   output=Diagnosis, effort="low", max_tokens=4000)
    assert res.data["one_line_read"] and not res.fallback_used
