"""Run Claude requests through the Claude Code CLI (`claude -p`) on the user's Claude subscription.

Nothing here touches API credits or paid extra usage:
  * Every ANTHROPIC_* variable (API keys, auth tokens, gateway URLs, Console profiles, federation
    credentials) and every CLAUDE_CODE_USE_* provider switch is removed from the child's environment.
    In print mode Claude Code would otherwise bill a present API key instead of the subscription.
  * Before the first request, `claude auth status` must report a Claude subscription login: claude.ai,
    or an OAuth token from `claude setup-token` in CLAUDE_CODE_OAUTH_TOKEN. Anything else is refused.
  * Every run's `system/init` event is checked again: a run that reports an API key source is killed.
  * The moment a run reports paid extra usage (rate_limit_event isUsingOverage), it is killed, every
    other live run is killed, and a plan stop is recorded: no request starts again, in this process
    or a later one, until the limit resets. A plan usage limit is recorded the same way.

Each request is one headless run with all tools off (`--tools ""`), our own system prompt, no
CLAUDE.md, memory, MCP servers, plugins, hooks or settings files, and no saved session. The answer
comes back through `--json-schema` (validated by Claude Code, then again by pydantic).
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .cost import Usage

# Kept: CLAUDE_CODE_OAUTH_TOKEN (a subscription token). Everything below can route billing elsewhere.
BILLING_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
BILLING_EXTRA = ("AWS_BEARER_TOKEN_BEDROCK",)
INSTALL_HINT = ("Install Claude Code (in Terminal: curl -fsSL https://claude.ai/install.sh | bash), run `claude` "
                "once and log in with your Claude account, then re-run. Or use the paid API instead: "
                "--judge-backend api.")
LOGIN_HINT = "Run `claude` in Terminal and log in with your Claude account (/login), then re-run."
DEFAULT_STOP_S = 3600.0  # how long a plan stop lasts when Claude Code doesn't say when the limit resets
_USAGE_LIMIT = re.compile(r"usage limit|limit reached|hit your (?:\w+ )*limit|out of (?:extra )?usage|"
                          r"(?:5-hour|five-hour|weekly|session) limit", re.I)
_AUTH = re.compile(r"not logged in|please run /login|invalid api key|authentication|oauth token (?:has )?expired|"
                   r"invalid bearer|unauthori[sz]ed", re.I)
_REFUSAL = re.compile(r"usage policy|unable to respond to this request", re.I)


class ClaudeCodeError(RuntimeError):
    """`fatal` errors stop every later request in the run (login, billing, plan limits)."""

    def __init__(self, kind: str, message: str, fatal: bool = False, rate_limit: dict | None = None):
        super().__init__(message)
        self.kind = kind
        self.fatal = fatal
        self.rate_limit = rate_limit or {}
        self.from_stop = False  # raised because a recorded plan stop is in force (not a new event)


@dataclass
class RunResult:
    data: dict
    usages: list[Usage]
    served_model: str                  # the model that wrote the answer
    answer_models: list[str]           # every model seen on the main thread (fallback detection)
    rate_limit: dict = field(default_factory=dict)
    overage_seen: bool = False         # this run was billed as paid extra usage (only when allowed)


# Errors that mean no request should start again until the plan's limit resets.
PLAN_KINDS = ("usage_limit", "extra_usage")


# -- process-wide plan stop and live runs ---------------------------------------------------------

_lock = threading.Lock()
_stop_write_lock = threading.Lock()  # serializes set_plan_stop (never taken inside _lock)
_live: set[subprocess.Popen] = set()
_stop: dict | None = None  # {"kind", "message", "until"}; shared by every LLM in this process


def stop_all_runs() -> None:
    """Kill every `claude -p` this process has running (plan stop, Ctrl-C)."""
    with _lock:
        procs = list(_live)
    for p in procs:
        try:
            p.kill()
        except OSError:
            pass


def _read_stop_file(state_file: Path) -> dict | None:
    try:
        entry = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return entry if isinstance(entry, dict) else None


def set_plan_stop(kind: str, message: str, until: float, state_file: Path | None = None) -> None:
    """Record a plan stop that every later request honours, remember it on disk (so a new process
    doesn't spend one more paid call to rediscover it), then kill the live runs.

    Several runs can hit the limit at once, so the read-merge-write is serialized and never shortens
    a stop: the latest reset wins, and an extra-usage stop is never downgraded to a usage-limit one."""
    global _stop
    with _stop_write_lock:
        with _lock:
            existing = [e for e in (_stop,) if e]
        if state_file is not None and state_file.exists():
            disk = _read_stop_file(state_file)
            if disk is None:  # unreadable: assume the worst rather than forget it
                disk = {"kind": "extra_usage", "until": _mtime(state_file) + DEFAULT_STOP_S, "message": message}
            existing.append(disk)
        untils = [until] + [e["until"] for e in existing if isinstance(e.get("until"), (int, float))]
        kinds = {kind} | {e.get("kind") for e in existing}
        entry = {"kind": "extra_usage" if "extra_usage" in kinds else kind,
                 "message": next((e.get("message") for e in existing if e.get("kind") == "extra_usage"
                                  and kind != "extra_usage"), None) or message,
                 "until": max(untils), "file": str(state_file) if state_file is not None else None}
        if state_file is not None:  # before the kill, so the killed runs find it
            try:
                state_file.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=state_file.parent, prefix=".plan-stop-", suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(entry, f)
                os.replace(tmp, state_file)
            except OSError:
                if not state_file.exists():
                    entry["file"] = None
        with _lock:
            _stop = entry
    stop_all_runs()


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return time.time()


def plan_stop(state_file: Path | None = None) -> ClaudeCodeError | None:
    """The active stop, if any (in memory, else from the state file). Expired stops are cleared, and
    deleting the state file lifts a stop at once (the documented way to retry sooner). A state file
    that can't be read counts as a stop (fail closed) for DEFAULT_STOP_S after it was written."""
    global _stop
    now = time.time()
    with _lock:
        entry = _stop
        if entry and (now >= entry.get("until", 0) or (entry.get("file") and not Path(entry["file"]).exists())):
            _stop = entry = None
    if entry is None and state_file is not None and state_file.exists():
        entry = _read_stop_file(state_file)
        if entry is None or not isinstance(entry.get("until"), (int, float)):
            entry = {"kind": "extra_usage", "until": _mtime(state_file) + DEFAULT_STOP_S,
                     "message": "an earlier scan stopped for a plan limit or paid extra usage, and its record "
                                "couldn't be read", "file": str(state_file)}
        if now >= entry["until"]:
            try:
                state_file.unlink()
            except OSError:
                pass
            entry = None
        else:
            entry.setdefault("file", str(state_file))
    if not entry:
        return None
    msg = entry.get("message", "judging is paused")
    if entry.get("file"):
        msg += f" (To retry sooner, delete {entry['file']}.)"
    err = ClaudeCodeError(entry.get("kind", "paused"), msg, fatal=True, rate_limit={"resetsAt": entry["until"]})
    err.from_stop = True
    return err


def clear_plan_stop(state_file: Path | None = None) -> None:
    """Tests and `doctor`: forget any stop."""
    global _stop
    with _lock:
        _stop = None
    if state_file is not None:
        try:
            state_file.unlink()
        except OSError:
            pass


def stop_until(rate_limit: dict) -> float:
    try:
        reset = float(rate_limit.get("resetsAt"))
    except (TypeError, ValueError):
        reset = 0.0
    return reset if reset > time.time() else time.time() + DEFAULT_STOP_S


# -- environment and login ------------------------------------------------------------------------

def find_claude(configured: str = "") -> str | None:
    if configured:
        p = Path(os.path.expanduser(configured))
        found = str(p.resolve()) if p.is_file() and os.access(p, os.X_OK) else shutil.which(configured)
        return str(Path(found).resolve()) if found else None
    found = shutil.which("claude")
    if found:
        return found
    # Shells started from a GUI (or a launchd job) may not have the user's PATH.
    for p in (Path.home() / ".claude" / "local" / "claude", Path.home() / ".local" / "bin" / "claude",
              Path("/opt/homebrew/bin/claude"), Path("/usr/local/bin/claude")):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def child_env(max_output_tokens: int | None = None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(BILLING_PREFIXES) and k not in BILLING_EXTRA}
    env.update({
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",     # the judges see our prompt, not the user's CLAUDE.md
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "DISABLE_AUTOUPDATER": "1",                # no self-update in the middle of a scan
    })
    if max_output_tokens:
        env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)
    return env


def auth_status(exe: str, timeout: float = 60.0) -> dict:
    try:
        p = subprocess.run([exe, "auth", "status"], capture_output=True, text=True, timeout=timeout,
                           env=child_env(), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise ClaudeCodeError("not_installed", f"could not run `{exe} auth status` ({e}). {INSTALL_HINT}",
                              fatal=True) from e
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("loggedIn", p.returncode == 0)
    data["_exit"] = p.returncode
    return data


def is_subscription(st: dict, env: dict | None = None) -> bool:
    """claude.ai login, or a `claude setup-token` token. "oauth_token" alone is not enough: Console
    profiles and federation credentials are reported the same way and bill API credits."""
    env = child_env() if env is None else env
    if st.get("apiProvider") not in (None, "firstParty"):
        return False
    method = st.get("authMethod")
    return method == "claude.ai" or (method == "oauth_token" and bool(env.get("CLAUDE_CODE_OAUTH_TOKEN")))


def check_subscription(exe: str) -> dict:
    """Raise unless Claude Code is logged in to a Claude subscription."""
    st = auth_status(exe)
    if not st.get("loggedIn"):
        raise ClaudeCodeError("auth", f"Claude Code is not logged in. {LOGIN_HINT}", fatal=True)
    if not is_subscription(st):
        raise ClaudeCodeError(
            "api_billing",
            f"Claude Code is signed in with '{st.get('authMethod') or 'unknown'}' "
            f"({st.get('apiProvider') or 'unknown provider'}), which bills API credits or a cloud account, not "
            "your Claude subscription, so judging did not run. Run `claude auth logout`, then `claude` and log in "
            "with your Claude account. (To pay per token on purpose, use --judge-backend api.)", fatal=True)
    return st


# -- one request ----------------------------------------------------------------------------------

def to_stream_message(messages: list[dict]) -> str:
    """Our Messages-API style request as one stream-json user message.

    cache_control markers are dropped: Claude Code places its own cache breakpoints, and a 5m marker
    after its 1h ones is rejected by the API."""
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ClaudeCodeError("config", "the Claude Code backend sends exactly one user message per request")
    content = messages[0]["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    blocks = []
    for b in copy.deepcopy(content):
        b.pop("cache_control", None)
        blocks.append(b)
    return json.dumps({"type": "user", "message": {"role": "user", "content": blocks},
                       "parent_tool_use_id": None}, ensure_ascii=False) + "\n"


def system_text(system: list[dict] | str) -> str:
    if isinstance(system, str):
        return system
    return "\n\n".join(b.get("text", "") for b in system if b.get("type") == "text")


def fmt_reset(epoch) -> str:
    try:
        return time.strftime("%a %H:%M", time.localtime(float(epoch)))
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def usages_from_result(result: dict, model: str) -> list[Usage]:
    out = []
    for name, u in (result.get("modelUsage") or {}).items():
        if not isinstance(u, dict):
            continue
        out.append(Usage(model=name, input_tokens=int(u.get("inputTokens") or 0),
                         output_tokens=int(u.get("outputTokens") or 0),
                         cache_write_tokens=int(u.get("cacheCreationInputTokens") or 0),
                         cache_read_tokens=int(u.get("cacheReadInputTokens") or 0)))
    if not out:
        u = result.get("usage") or {}
        out.append(Usage(model=model, input_tokens=int(u.get("input_tokens") or 0),
                         output_tokens=int(u.get("output_tokens") or 0),
                         cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
                         cache_read_tokens=int(u.get("cache_read_input_tokens") or 0)))
    # The requested model first, so callers can tell its spend from any helper model's.
    out.sort(key=lambda x: x.model != model)
    return out


def usage_limit_error(rate_limit: dict) -> ClaudeCodeError:
    reset = fmt_reset(rate_limit.get("resetsAt"))
    return ClaudeCodeError("usage_limit", "your Claude plan's usage limit is reached"
                           + (f" (resets {reset})" if reset else "")
                           + ". Finished judgments are cached; re-run after the reset to continue.",
                           fatal=True, rate_limit=rate_limit)


def extra_usage_error(rate_limit: dict) -> ClaudeCodeError:
    reset = fmt_reset(rate_limit.get("resetsAt"))
    return ClaudeCodeError("extra_usage", "your Claude plan's included usage ran out and Claude Code started using "
                           "paid extra usage, so judging stopped"
                           + (f" until your limit resets ({reset})" if reset else "")
                           + ". Finished judgments are cached; re-run after the reset to continue.",
                           fatal=True, rate_limit=rate_limit)


def classify_error(result: dict | None, rate_limit: dict, stderr: str, returncode: int | None) -> ClaudeCodeError:
    text = ""
    status = None
    if result:
        text = str(result.get("result") or "") + " " + " ".join(map(str, result.get("errors") or []))
        status = result.get("api_error_status")
    text = (text + " " + stderr[-2000:]).strip()
    # With extra usage allowed, the CLI reports status "rejected" together with isUsingOverage: that is
    # the plan's included usage running out while paid usage carries on, not a hard stop.
    if (rate_limit.get("status") == "rejected" and not rate_limit.get("isUsingOverage")) or _USAGE_LIMIT.search(text):
        return usage_limit_error(rate_limit)
    if status in (401, 403) or _AUTH.search(text):
        return ClaudeCodeError("auth", f"Claude Code could not authenticate ({text[:200]}). {LOGIN_HINT}", fatal=True)
    if result and result.get("subtype") == "error_max_structured_output_retries":
        return ClaudeCodeError("schema", "no valid structured answer after Claude Code's retries")
    if _REFUSAL.search(text):
        return ClaudeCodeError("refusal", text[:300])
    if status == 429:
        return ClaudeCodeError("rate_limit", text[:300])
    if status == 400:
        return ClaudeCodeError("bad_request", text[:500])
    if status == 404 or ("model" in text.lower() and "not found" in text.lower()):
        return ClaudeCodeError("model_not_found", f"{text[:300]} (is this model available on your plan?)")
    if status:
        return ClaudeCodeError("api", f"{status}: {text[:300]}")
    return ClaudeCodeError("claude_code", f"Claude Code run failed (exit {returncode}): {text[:400]}")


@dataclass
class _Events:
    result: dict | None = None
    rate_limit: dict = field(default_factory=dict)
    overage_seen: bool = False
    main_models: list[str] = field(default_factory=list)
    answer_model: str = ""
    killed_for: str = ""   # "api_key:<source>" | "extra_usage"
    timed_out: bool = False


def _read_events(proc: subprocess.Popen, payload: str, timeout: float, allow_extra_usage: bool) -> _Events:
    """Feed the request, read the event stream, enforce the timeout and the billing guards."""
    ev_state = _Events()
    timed_out = threading.Event()

    def kill_on_timeout() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout, kill_on_timeout)
    timer.daemon = True
    timer.start()

    def feed() -> None:
        try:
            proc.stdin.write(payload)
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

    writer = threading.Thread(target=feed, daemon=True)
    writer.start()
    try:
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            kind = ev.get("type")
            if kind == "system" and ev.get("subtype") == "init":
                source = ev.get("apiKeySource")
                if source not in (None, "", "none"):
                    ev_state.killed_for = f"api_key:{source}"
                    proc.kill()
                    break
            elif kind == "assistant" and ev.get("parent_tool_use_id") is None:
                msg = ev.get("message") or {}
                m = msg.get("model") if isinstance(msg, dict) else None
                if m and m != "<synthetic>":
                    ev_state.main_models.append(m)
                    if any(isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "StructuredOutput"
                           for b in msg.get("content") or []):
                        ev_state.answer_model = m
            elif kind == "rate_limit_event":
                info = ev.get("rate_limit_info")
                if isinstance(info, dict):
                    ev_state.rate_limit = info
                    if info.get("isUsingOverage"):
                        ev_state.overage_seen = True
                        if not allow_extra_usage:
                            # Paid from here on: stop now rather than after the answer is written.
                            ev_state.killed_for = "extra_usage"
                            proc.kill()
                            break
            elif kind == "result":
                ev_state.result = ev
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        writer.join(timeout=5)
    ev_state.timed_out = timed_out.is_set()
    return ev_state


def run_structured(exe: str, *, model: str, system: list[dict] | str, messages: list[dict], schema: dict,
                   effort: str | None, max_tokens: int, timeout: float = 900.0, allow_extra_usage: bool = False,
                   state_file: Path | None = None) -> RunResult:
    """One headless request. Raises ClaudeCodeError (fatal for login/billing/plan-limit problems)."""
    stop = plan_stop(state_file)
    if stop is not None:
        raise stop
    with tempfile.TemporaryDirectory(prefix="clip-scanner-cc-") as tmp:
        prompt_file = Path(tmp) / "system.md"
        prompt_file.write_text(system_text(system), encoding="utf-8")
        cmd = [exe, "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--model", model, "--tools", "", "--json-schema", json.dumps(schema),
               "--system-prompt-file", str(prompt_file), "--no-session-persistence", "--strict-mcp-config",
               "--disable-slash-commands", "--setting-sources", "", "--safe-mode"]
        if effort:
            cmd += ["--effort", effort]
        payload = to_stream_message(messages)
        with open(Path(tmp) / "stderr.txt", "w+", encoding="utf-8", errors="replace") as errf:
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, cwd=tmp,
                                        env=child_env(max_tokens), text=True, encoding="utf-8", errors="replace")
            except OSError as e:
                raise ClaudeCodeError("not_installed", f"could not start Claude Code ({e}). {INSTALL_HINT}",
                                      fatal=True) from e
            with _lock:
                _live.add(proc)
            try:
                ev = _read_events(proc, payload, timeout, allow_extra_usage)
            finally:
                with _lock:
                    _live.discard(proc)
            errf.seek(0)
            stderr = errf.read()

    rl = ev.rate_limit
    if ev.killed_for.startswith("api_key:"):
        raise ClaudeCodeError(
            "api_billing", f"Claude Code was about to use an API key ({ev.killed_for[8:]}), which bills API credits, "
            "so the run was stopped. Remove that key from Claude Code's settings or log in with your Claude account.",
            fatal=True, rate_limit=rl)
    if ev.killed_for == "extra_usage" or (ev.overage_seen and not allow_extra_usage):
        raise extra_usage_error(rl)
    stop = plan_stop(state_file)  # another run hit a plan stop and killed this one
    if stop is not None and (ev.result is None or ev.result.get("is_error")):
        raise stop
    if ev.timed_out:
        raise ClaudeCodeError("timeout", f"no answer within {timeout:.0f}s", rate_limit=rl)
    result = ev.result
    if result is None:
        err = classify_error(None, rl, stderr, proc.returncode)
        err.rate_limit = rl
        raise err
    if result.get("is_error") or result.get("subtype") != "success":
        err = classify_error(result, rl, stderr, proc.returncode)
        err.rate_limit = rl
        raise err
    data = result.get("structured_output")
    if not isinstance(data, dict):
        raise ClaudeCodeError("invalid_json", f"no structured answer: {str(result.get('result'))[:300]}",
                              rate_limit=rl)
    usages = usages_from_result(result, model)
    served = ev.answer_model or (ev.main_models[-1] if ev.main_models else "")
    if not served and len(result.get("modelUsage") or {}) == 1:
        served = next(iter(result["modelUsage"]))
    return RunResult(data=data, usages=usages, served_model=served, answer_models=ev.main_models, rate_limit=rl,
                     overage_seen=ev.overage_seen)


def version(exe: str) -> str:
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30,
                              env=child_env(), stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
