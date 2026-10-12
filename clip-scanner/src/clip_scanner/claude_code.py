"""Run Claude requests through the Claude Code CLI (`claude -p`) on the user's Claude subscription.

Nothing here touches API credits:
  * ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN (and third-party provider switches) are removed from the
    child's environment. In print mode Claude Code would otherwise bill a present API key instead of
    the subscription.
  * Before the first request, `claude auth status` must report a subscription login (claude.ai or a
    `claude setup-token` OAuth token). An API-key or cloud-provider login is refused.
  * Every run's `system/init` event is checked again: a run that reports an API key source is killed
    and judging stops.
  * If Claude Code reports that it is drawing on paid extra usage, judging stops after that call.

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

# Variables that make Claude Code bill something other than the logged-in subscription.
BILLING_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
               "CLAUDE_CODE_USE_FOUNDRY", "CLAUDE_CODE_USE_ANTHROPIC_AWS", "CLAUDE_CODE_USE_ANTHROPIC_PLATFORM_AWS")
SUBSCRIPTION_AUTH = ("claude.ai", "oauth_token")
INSTALL_HINT = ("Install Claude Code (https://claude.com/claude-code), run `claude` once and log in with your "
                "Claude account, then re-run. Or use the paid API instead: --judge-backend api.")
LOGIN_HINT = "Run `claude` in Terminal and log in with your Claude account (/login), then re-run."
_USAGE_LIMIT = re.compile(r"usage limit|limit reached|hit your (?:\w+ )*limit|out of (?:extra )?usage|"
                          r"(?:5-hour|five-hour|weekly|session) limit", re.I)
_AUTH = re.compile(r"not logged in|please run /login|invalid api key|authentication|oauth token (?:has )?expired|"
                   r"invalid bearer|unauthori[sz]ed", re.I)
_REFUSAL = re.compile(r"usage policy|unable to respond to this request", re.I)


class ClaudeCodeError(RuntimeError):
    """`fatal` errors stop every later request in the run (login, billing, plan limits)."""

    def __init__(self, kind: str, message: str, fatal: bool = False):
        super().__init__(message)
        self.kind = kind
        self.fatal = fatal


@dataclass
class RunResult:
    data: dict
    usages: list[Usage]
    served_model: str                  # the model that wrote the answer
    answer_models: list[str]           # every model seen on the main thread (fallback detection)
    rate_limit: dict = field(default_factory=dict)


def find_claude(configured: str = "") -> str | None:
    if configured:
        p = Path(os.path.expanduser(configured))
        return str(p) if p.is_file() and os.access(p, os.X_OK) else shutil.which(configured)
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
    env = {k: v for k, v in os.environ.items() if k not in BILLING_ENV}
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


def check_subscription(exe: str, allow_api_billing: bool = False) -> dict:
    """Raise unless Claude Code is logged in to a Claude subscription."""
    st = auth_status(exe)
    if not st.get("loggedIn"):
        raise ClaudeCodeError("auth", f"Claude Code is not logged in. {LOGIN_HINT}", fatal=True)
    method = st.get("authMethod")
    if method not in SUBSCRIPTION_AUTH and not allow_api_billing:
        raise ClaudeCodeError(
            "api_billing",
            f"Claude Code is logged in with '{method or 'unknown'}', which bills API credits or a cloud account, "
            "not your Claude subscription, so judging did not run. Run `claude auth logout`, then `claude` and "
            "log in with your Claude account. (To allow it anyway, set judging.claude_code.allow_api_billing = true.)",
            fatal=True)
    return st


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


def _fmt_reset(epoch) -> str:
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


def classify_error(result: dict | None, rate_limit: dict, stderr: str, returncode: int | None) -> ClaudeCodeError:
    text = ""
    status = None
    if result:
        text = str(result.get("result") or "") + " " + " ".join(map(str, result.get("errors") or []))
        status = result.get("api_error_status")
    text = (text + " " + stderr[-2000:]).strip()
    if rate_limit.get("status") == "rejected" or _USAGE_LIMIT.search(text):
        reset = _fmt_reset(rate_limit.get("resetsAt"))
        return ClaudeCodeError("usage_limit",
                               "your Claude plan's usage limit is reached" + (f" (resets {reset})" if reset else "")
                               + ". Finished judgments are cached; re-run after the reset to continue.", fatal=True)
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


def _read_events(proc: subprocess.Popen, payload: str, timeout: float, allow_api_billing: bool):
    """Feed the request, read the event stream, enforce the timeout and the API-key guard."""
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
    result: dict | None = None
    rate_limit: dict = {}
    main_models: list[str] = []
    answer_model = ""
    killed_for_billing = ""
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
                if source not in (None, "", "none") and not allow_api_billing:
                    killed_for_billing = str(source)
                    proc.kill()
                    break
            elif kind == "assistant" and ev.get("parent_tool_use_id") is None:
                msg = ev.get("message") or {}
                m = msg.get("model") if isinstance(msg, dict) else None
                if m and m != "<synthetic>":
                    main_models.append(m)
                    if any(isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "StructuredOutput"
                           for b in msg.get("content") or []):
                        answer_model = m
            elif kind == "rate_limit_event":
                info = ev.get("rate_limit_info")
                if isinstance(info, dict):
                    rate_limit = info
            elif kind == "result":
                result = ev
        proc.wait()
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        writer.join(timeout=5)
    return result, rate_limit, main_models, answer_model, killed_for_billing, timed_out.is_set()


def run_structured(exe: str, *, model: str, system: list[dict] | str, messages: list[dict], schema: dict,
                   effort: str | None, max_tokens: int, timeout: float = 900.0,
                   allow_api_billing: bool = False) -> RunResult:
    """One headless request. Raises ClaudeCodeError (fatal for login/billing/plan-limit problems)."""
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
            ev_state = _read_events(proc, payload, timeout, allow_api_billing)
            errf.seek(0)
            stderr = errf.read()
        result, rate_limit, main_models, answer_model, killed_for_billing, timed_out = ev_state

    if killed_for_billing:
        raise ClaudeCodeError(
            "api_billing", f"Claude Code was about to use an API key ({killed_for_billing}), which bills API credits, "
            "so the run was stopped. Remove that key from Claude Code's settings or log in with your Claude account.",
            fatal=True)
    if timed_out:
        raise ClaudeCodeError("timeout", f"no answer within {timeout:.0f}s")
    if result is None:
        raise classify_error(None, rate_limit, stderr, proc.returncode)
    if result.get("is_error") or result.get("subtype") != "success":
        raise classify_error(result, rate_limit, stderr, proc.returncode)
    data = result.get("structured_output")
    if not isinstance(data, dict):
        raise ClaudeCodeError("invalid_json", f"no structured answer: {str(result.get('result'))[:300]}")
    usages = usages_from_result(result, model)
    served = answer_model or (main_models[-1] if main_models else "")
    if not served and len(result.get("modelUsage") or {}) == 1:
        served = next(iter(result["modelUsage"]))
    return RunResult(data=data, usages=usages, served_model=served, answer_models=main_models,
                     rate_limit=rate_limit)


def version(exe: str) -> str:
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30,
                              env=child_env(), stdin=subprocess.DEVNULL).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""
