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


LEAKY = {"ANTHROPIC_API_KEY": "sk-ant-should-not-leak", "ANTHROPIC_AUTH_TOKEN": "tok-should-not-leak",
         "ANTHROPIC_BASE_URL": "https://gateway.example", "ANTHROPIC_PROFILE": "console-org",
         "CLAUDE_CODE_USE_MANTLE": "1", "AWS_BEARER_TOKEN_BEDROCK": "aws-should-not-leak"}


@pytest.fixture(autouse=True)
def _no_plan_stop_leaks():
    from clip_scanner import claude_code

    claude_code.clear_plan_stop()
    yield
    claude_code.clear_plan_stop()


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
    # Nothing in the user's shell that could route billing elsewhere may reach Claude Code.
    for var, val in LEAKY.items():
        monkeypatch.setenv(var, val)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    from clip_scanner import claude_code

    claude_code.clear_plan_stop()

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
        def auth(a, provider="firstParty"):
            monkeypatch.setenv("FAKE_CLAUDE_AUTH", a)
            monkeypatch.setenv("FAKE_CLAUDE_PROVIDER", provider)

        @staticmethod
        def setenv(k, v):
            monkeypatch.setenv(k, v)

    return F


def make_llm(fake, tmp_path, **cc):
    cfg = Config.model_validate({"general": {"cache_dir": str(fake.tmp / "cache")},
                                 "judging": {"claude_code": {"path": str(fake.path), **cc}}})
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
    for var in LEAKY:
        assert c["env"][var] is None, var
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


@pytest.mark.parametrize("auth, provider, kind", [
    ("api_key", "firstParty", "api_billing"), ("api_key_helper", "firstParty", "api_billing"),
    ("third_party", "bedrock", "api_billing"), ("none", "firstParty", "auth"),
    ("oauth_token", "firstParty", "api_billing"),   # a Console profile / federation login reports this too
    ("claude.ai", "vertex", "api_billing")])
def test_refuses_anything_but_a_subscription_login(fake, tmp_path, auth, provider, kind):
    fake.auth(auth, provider)
    llm, _ = make_llm(fake, tmp_path)
    with pytest.raises(LLMError) as e:
        call(llm)
    assert e.value.kind == kind
    assert fake.calls() == []                     # nothing was sent
    assert llm.fatal is not None
    with pytest.raises(LLMError) as again:        # and nothing ever will be this run
        call(llm, text="another clip")
    assert again.value.kind == kind and fake.calls() == []


def test_setup_token_counts_as_subscription(fake, tmp_path):
    fake.auth("oauth_token")
    fake.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-subscription")  # `claude setup-token`
    llm, _ = make_llm(fake, tmp_path)
    assert call(llm).data["score"] == 7.0
    assert fake.calls()[0]["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat-subscription"


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


def test_extra_usage_kills_the_run_and_stops_every_later_scan(fake, tmp_path):
    fake.mode("overage")
    llm, tracker = make_llm(fake, tmp_path)
    with pytest.raises(LLMError) as e:
        call(llm)                                 # killed as soon as Claude Code reports paid usage
    assert e.value.kind == "extra_usage" and llm.fatal is not None
    assert tracker.summary()["extra_usage_used"]
    with pytest.raises(LLMError):
        call(llm, text="next")
    # A new scan (a queued UI link, the next calibration clip, a new process) doesn't try again either.
    llm2, _ = make_llm(fake, tmp_path / "next-scan")
    with pytest.raises(LLMError) as e2:
        call(llm2, text="another video")
    assert e2.value.kind == "extra_usage" and "delete" in str(e2.value)
    from clip_scanner import claude_code

    claude_code.clear_plan_stop()                 # forget the in-memory stop: the file alone must hold
    llm3, _ = make_llm(fake, tmp_path / "new-process")
    with pytest.raises(LLMError):
        call(llm3, text="after restart")
    assert len(fake.calls()) == 1


def test_plan_stop_expires_at_the_reset(fake, tmp_path):
    from clip_scanner import claude_code

    llm, _ = make_llm(fake, tmp_path)
    llm.plan_stop_file.parent.mkdir(parents=True, exist_ok=True)
    llm.plan_stop_file.write_text(json.dumps({"kind": "usage_limit", "message": "limit", "until": 1.0}))
    assert claude_code.plan_stop(llm.plan_stop_file) is None and not llm.plan_stop_file.exists()
    assert call(llm).data["score"] == 7.0


def test_a_stop_kills_requests_already_running(fake, tmp_path):
    import threading
    import time

    fake.setenv("FAKE_CLAUDE_SLOW", "30")
    llm, _ = make_llm(fake, tmp_path)
    errors = []

    def slow():
        try:
            call(llm, text="FAKE_MODE:slow")
        except LLMError as e:
            errors.append(e)

    t0 = time.time()
    th = threading.Thread(target=slow)
    th.start()
    time.sleep(1.5)                               # the slow request is in flight
    with pytest.raises(LLMError):
        call(llm, text="FAKE_MODE:overage")
    th.join(timeout=15)
    assert not th.is_alive() and time.time() - t0 < 15
    assert errors and errors[0].kind == "extra_usage"


def test_extra_usage_allowed_is_reported(fake, tmp_path):
    fake.mode("overage")
    llm, tracker = make_llm(fake, tmp_path, allow_extra_usage=True)
    assert call(llm).data["score"] == 7.0
    fake.mode("overage_error")
    with pytest.raises(LLMError):
        call(llm, text="next")
    assert llm.fatal is None and tracker.summary()["extra_usage_used"]
    assert not llm.plan_stop_file.exists()        # paid usage is allowed here: not a plan stop


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
    cfg = Config.model_validate({"general": {"cache_dir": str(tmp_path / "cache")},
                                 "judging": {"claude_code": {"path": str(tmp_path / "nope" / "claude")}}})
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


def test_cut_short_recall_is_never_no_great_clips(fake, tmp_path):
    from clip_scanner.judging.runner import run_judging

    RESP = dict(RESPONSES, first_two_seconds=verdict(score=6.0, great=False))
    (fake.tmp / "responses.json").write_text(json.dumps(RESP))
    llm, _ = make_llm(fake, tmp_path)
    cands, ctx = judging_context(llm)
    assert run_judging(cands, ctx, max_clips=5).status == "complete"     # everything read: a real verdict
    res = run_judging(cands, ctx, max_clips=5, recall_complete=False)
    assert res.status == "incomplete" and not res.great and res.diagnosis == ""
    assert any("proposer" in n for n in res.notes)


def test_ctrl_c_kills_running_requests_and_queued_candidates(fake, tmp_path, monkeypatch):
    import os
    import signal
    import threading
    import time

    from clip_scanner.candidates import Candidate
    from clip_scanner.judging import runner

    fake.setenv("FAKE_CLAUDE_SLOW", "30")
    llm, _ = make_llm(fake, tmp_path, timeout_s=60)
    llm.cfg.judging.max_concurrency = 2
    _, ctx = judging_context(llm)
    cands = [Candidate(id=f"k{i}", segments=[(10.0, 40.0)], peak=15.0, origin="signal", score=1.0 - i / 10,
                       reason="") for i in range(8)]
    seen = []

    def judge(c, *a, **k):
        seen.append(c.id)
        if c.id != "k0":
            try:
                call(llm, text=f"FAKE_MODE:slow {c.id}")  # 30 s requests in flight
            except LLMError:
                pass
        return runner.decide(c.id, [], None, None, None, ctx.cfg)

    monkeypatch.setattr(runner, "judge_candidate", judge)
    timer = threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGINT))  # the user presses Ctrl-C
    t0 = time.time()
    timer.start()
    with pytest.raises(KeyboardInterrupt):
        runner.run_judging(cands, ctx, max_clips=5)
    timer.cancel()
    assert time.time() - t0 < 15                  # running `claude -p` requests were killed, not waited for
    assert len(seen) <= 4, seen                   # queued candidates never started


def test_honouring_a_stop_never_shortens_it(fake, tmp_path):
    from clip_scanner import claude_code

    fake.mode("overage")
    llm, _ = make_llm(fake, tmp_path)
    with pytest.raises(LLMError):
        call(llm)
    saved = json.loads(llm.plan_stop_file.read_text())
    assert saved["until"] == 1893456000           # the real reset time from Claude Code
    for i in range(3):                            # later scans that are refused by the stop
        with pytest.raises(LLMError):
            call(make_llm(fake, tmp_path / f"s{i}")[0], text=f"scan {i}")
    again = json.loads(llm.plan_stop_file.read_text())
    assert again["until"] == 1893456000 and again["message"] == saved["message"]
    # Deleting the file is the documented way to retry sooner, and works without a restart.
    llm.plan_stop_file.unlink()
    fake.mode("ok")
    assert call(make_llm(fake, tmp_path / "retry")[0], text="retry").data["score"] == 7.0
    assert claude_code.plan_stop(llm.plan_stop_file) is None


def test_fixing_the_login_works_without_a_restart(fake, tmp_path):
    fake.auth("api_key")
    with pytest.raises(LLMError) as e:
        call(make_llm(fake, tmp_path)[0])
    assert e.value.kind == "api_billing"
    fake.auth("claude.ai")                        # user logs in with their Claude account
    assert call(make_llm(fake, tmp_path / "after")[0]).data["score"] == 7.0


def test_proposer_retries_claude_code_schema_failures(fake, tmp_path):
    from clip_scanner.proposer import propose
    from clip_scanner.signals import HOP  # noqa: F401  (signals import side effects)
    from clip_scanner.transcribe import Transcript, Word

    fake.mode("schema_retries")
    llm, _ = make_llm(fake, tmp_path)
    words = [Word(round(i * 0.4, 2), round(i * 0.4 + 0.3, 2), f"word{i % 7}") for i in range(1500)]
    moments, complete = propose(llm, Transcript(words=words), 600.0, llm.cfg)
    assert not complete and moments == []         # not cached, not deterministic: retry next run


def test_simultaneous_stops_merge_and_never_corrupt_the_file(tmp_path):
    import threading
    import time

    from clip_scanner import claude_code as cc

    f = tmp_path / "cache" / "claude-plan-stop.json"
    far = time.time() + 3 * 86400
    jobs = [("extra_usage", far), ("usage_limit", time.time() + 600)] * 8
    barrier = threading.Barrier(len(jobs))

    def hit(kind, until):
        barrier.wait()
        cc.set_plan_stop(kind, f"{kind} message", until, f)

    threads = [threading.Thread(target=hit, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    saved = json.loads(f.read_text())                       # always valid JSON
    assert saved["kind"] == "extra_usage" and saved["until"] == far
    assert not list(f.parent.glob("*.tmp"))
    stop = cc.plan_stop(f)
    assert stop.kind == "extra_usage" and str(f) in str(stop)  # the delete hint survives the race


def test_unreadable_stop_file_still_stops(tmp_path):
    import os
    import time

    from clip_scanner import claude_code as cc

    f = tmp_path / "claude-plan-stop.json"
    f.write_text('{"kind": "extra_usage", "until": 9}}')      # damaged
    assert cc.plan_stop(f) is not None                        # fail closed
    old = time.time() - cc.DEFAULT_STOP_S - 10
    os.utime(f, (old, old))
    assert cc.plan_stop(f) is None and not f.exists()          # ...but not forever


def test_bad_answers_are_asked_again_on_the_subscription(fake, tmp_path):
    RESP = dict(RESPONSES, reason={"reason": "ok", "score": 70})   # outside 0-10: pydantic rejects it
    (fake.tmp / "responses.json").write_text(json.dumps(RESP))
    llm, _ = make_llm(fake, tmp_path)
    for _ in range(2):
        with pytest.raises(LLMError) as e:
            call(llm)
        assert e.value.kind == "schema" and not e.value.cached
    assert len(fake.calls()) == 2                                # asked again, not replayed


def test_stale_cached_failures_from_older_builds_are_ignored(fake, tmp_path):
    llm, _ = make_llm(fake, tmp_path)
    call(llm)                                                    # learn the request hash's cache path
    [entry] = list((tmp_path / "llm" / "judge").glob("*.json"))
    entry.write_text(json.dumps({"error": {"kind": "schema", "message": "old"}, "usages": []}))
    assert call(llm).data["score"] == 7.0 and len(fake.calls()) == 2
