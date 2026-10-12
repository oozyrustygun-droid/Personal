"""Claude calls with structured JSON output, request-level caching and cost tracking.

Backends: "claude-code" runs each request through the Claude Code CLI on the user's Claude
subscription (see claude_code.py); "anthropic" calls the Claude API (billed per token);
"scripted" is the offline test judge.

Every call:
  * asks for JSON constrained by a schema (`output_config.format`), then validates it again
    client-side with pydantic, because numeric bounds are not enforced by the API;
  * runs adaptive thinking at a configured effort;
  * is cached on disk by a hash of the full request, so re-runs and threshold changes are free;
  * fails closed: refusals, truncation, invalid JSON or schema violations raise LLMError, and
    callers turn that into a rejection rather than a pass.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .cache import read_json, write_json
from .config import Config
from .cost import CostTracker, Usage, usages_from_response

BACKENDS = ("claude-code", "anthropic", "scripted")
# Who pays: the Claude subscription (Claude Code), API credits, or nobody (offline test judge).
BILLING = {"claude-code": "subscription", "anthropic": "api", "scripted": "none"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side `fallbacks: "default"` parameter on the Claude API.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class LLMError(RuntimeError):
    def __init__(self, kind: str, message: str, cached: bool = False):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.cached = cached  # replayed from the on-disk response cache (a deterministic failure)


@dataclass
class LLMResult:
    data: dict
    usages: list[Usage]
    cached: bool
    served_model: str
    request_hash: str
    fallback_used: bool = False  # True if a model other than the requested one produced the answer


ScriptFn = Callable[[str, dict], dict]

_UNSUPPORTED_KEYS = {"title", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                     "multipleOf", "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
                     "examples"}


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema reduced to what structured outputs accept.

    Every object gets additionalProperties false and all properties required, and unsupported
    constraints are removed. Bounds are re-checked by pydantic after the response arrives.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: walk(v) for k, v in node.items() if k not in _UNSUPPORTED_KEYS}
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}).keys())
            return node
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def same_model(served: str | None, requested: str) -> bool:
    """The API may report a dated snapshot id for the requested alias; anything else is a different model."""
    return bool(served) and re.fullmatch(re.escape(requested) + r"(-\d{8})?", served) is not None


def _hash_request(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


class LLM:
    def __init__(self, cfg: Config, store_dir: Path, tracker: CostTracker, *, backend: str = "anthropic",
                 script: ScriptFn | None = None):
        if backend not in BACKENDS:
            raise ValueError(f"unknown LLM backend {backend!r}")
        self.cfg = cfg
        self.store_dir = Path(store_dir)
        self.tracker = tracker
        self.backend = backend
        self.script = script
        self._client = None
        self._lock = threading.Lock()
        self._cc_exe: str | None = None
        self._warned: set[str] = set()
        self.refresh = False  # True: ignore cached responses (--rerun judging)
        # Set by an error that no later request can get past (no login, API billing, plan limit):
        # every later live request fails at once instead of repeating it.
        self.fatal: LLMError | None = None

    @property
    def billing(self) -> str:
        return BILLING[self.backend]

    def _set_fatal(self, err: LLMError) -> None:
        with self._lock:
            if self.fatal is None:
                self.fatal = err

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                import anthropic

                try:
                    self._client = anthropic.Anthropic(max_retries=4, timeout=600.0)
                except anthropic.AnthropicError as e:
                    raise LLMError("auth", f"cannot create Anthropic client ({e}); set ANTHROPIC_API_KEY") from e
            return self._client

    def cached_result(self, purpose: str, request_hash: str) -> dict | None:
        path = self.store_dir / purpose / f"{request_hash}.json"
        if path.exists():
            try:
                return read_json(path)
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def use_fallbacks(self, model: str, allow_fallback: bool) -> bool:
        return (self.backend == "anthropic" and allow_fallback and self.cfg.judging.server_fallbacks
                and model in FALLBACK_MODELS)

    def request_hash(self, *, model: str, system: list[dict], messages: list[dict], schema: dict,
                     effort: str, fallbacks: bool = False) -> str:
        return _hash_request({"backend": self.backend, "model": model, "system": system, "messages": messages,
                              "schema": schema, "effort": effort, "fallbacks": fallbacks})

    def scripted_hash(self, purpose: str, meta: dict, rh: str) -> str:
        return _hash_request({"scripted": meta, "purpose": purpose, "h": rh,
                              "rules": getattr(self.script, "fingerprint", "")})

    def call(self, *, purpose: str, model: str, system: list[dict], messages: list[dict],
             output: type[BaseModel], effort: str, max_tokens: int, meta: dict | None = None,
             allow_fallback: bool = False) -> LLMResult:
        """allow_fallback: let the API re-run a refused request on another model. Never set it for
        verdicts. A substitute model's opinion is not the configured judge's opinion."""
        schema = strict_schema(output)
        fallbacks = self.use_fallbacks(model, allow_fallback)
        rh = self.request_hash(model=model, system=system, messages=messages, schema=schema, effort=effort,
                               fallbacks=fallbacks)
        if self.backend == "scripted":
            rh = self.scripted_hash(purpose, meta or {}, rh)
        hit = None if self.refresh else self.cached_result(purpose, rh)
        if hit is not None:
            usages = [Usage.from_json(u) for u in hit.get("usages", [])]
            self.tracker.record(purpose, usages, cached=True)
            if "error" in hit:
                raise LLMError(hit["error"]["kind"], hit["error"]["message"] + " (cached)", cached=True)
            served = hit.get("served_model", model)
            return LLMResult(data=self._validate(output, hit["data"]), usages=usages, cached=True,
                             served_model=served, request_hash=rh,
                             fallback_used=hit.get("fallback_used", not same_model(served, model)))

        if self.fatal is not None:
            raise LLMError(self.fatal.kind, str(self.fatal).split(": ", 1)[-1])
        if self.backend == "scripted":
            if self.script is None:
                raise LLMError("config", "scripted backend without a script")
            data = self.script(purpose, meta or {})
            usages = [Usage(model=model)]
            served = model
            fallback_used = False
        elif self.backend == "claude-code":
            data, usages, served, answer_models = self._claude_code(
                model=model, system=system, messages=messages, schema=schema, effort=effort, max_tokens=max_tokens,
                purpose=purpose)
            # Only the models that wrote on the main thread count; Claude Code may bill a helper model.
            fallback_used = not same_model(served, model) or any(not same_model(m, model) for m in answer_models)
        else:
            data, usages, served = self._anthropic(model=model, system=system, messages=messages, schema=schema,
                                                   effort=effort, max_tokens=max_tokens, purpose=purpose, rh=rh,
                                                   fallbacks=fallbacks)
            fallback_used = not same_model(served, model) or any(u.model and not same_model(u.model, model)
                                                                   for u in usages[1:])
        self.tracker.record(purpose, usages, cached=False)
        validated = self._validate(output, data, purpose=purpose, rh=rh, usages=usages)
        write_json(self.store_dir / purpose / f"{rh}.json",
                   {"data": data, "usages": [u.to_json() for u in usages], "served_model": served,
                    "requested_model": model, "fallback_used": fallback_used})
        return LLMResult(data=validated, usages=usages, cached=False, served_model=served, request_hash=rh,
                         fallback_used=fallback_used)

    def _validate(self, output: type[BaseModel], data: dict, *, purpose: str = "", rh: str = "",
                  usages: list[Usage] | None = None) -> dict:
        try:
            return output.model_validate(data).model_dump()
        except ValidationError as e:
            if rh and self.backend != "claude-code":
                # Cached so a re-run doesn't pay for it again. On the subscription nothing is paid, and
                # asking again is the only way a bad answer (e.g. a proposer chunk) gets a second chance.
                self._store_error(purpose, rh, "schema", str(e)[:2000], usages or [])
            raise LLMError("schema", str(e)[:500]) from e

    def _store_error(self, purpose: str, rh: str, kind: str, message: str, usages: list[Usage]) -> None:
        # Deterministic failures (refusal, schema) are cached so a re-run doesn't pay for them again.
        write_json(self.store_dir / purpose / f"{rh}.json",
                   {"error": {"kind": kind, "message": message}, "usages": [u.to_json() for u in usages]})

    def _anthropic(self, *, model: str, system: list[dict], messages: list[dict], schema: dict, effort: str,
                   max_tokens: int, purpose: str, rh: str, fallbacks: bool) -> tuple[dict, list[Usage], str]:
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if fallbacks:
                resp = self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            err = LLMError("auth", "Anthropic credentials missing or invalid (set ANTHROPIC_API_KEY, or use the "
                                   "default claude-code backend to judge on your Claude subscription)")
            self._set_fatal(err)
            raise err from e
        except TypeError as e:  # SDK raises this when no credential source resolves at all
            if "authentication" in str(e).lower():
                err = LLMError("auth", "no Anthropic credentials found (set ANTHROPIC_API_KEY, or use the default "
                                       "claude-code backend to judge on your Claude subscription)")
                self._set_fatal(err)
                raise err from e
            raise
        except anthropic.BadRequestError as e:
            raise LLMError("bad_request", str(e)[:500]) from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate_limit", str(e)[:300]) from e
        except anthropic.APIStatusError as e:
            raise LLMError("api", f"{e.status_code}: {str(e)[:300]}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("connection", str(e)[:300]) from e
        except anthropic.AnthropicError as e:  # anything else from the SDK: fail closed, never crash the run
            raise LLMError("api", f"{type(e).__name__}: {str(e)[:300]}") from e

        try:
            return self._parse_response(resp, model, purpose, rh)
        except LLMError:
            raise
        except (AttributeError, TypeError, KeyError, ValueError) as e:  # malformed response: fail closed
            raise LLMError("malformed_response", f"{type(e).__name__}: {str(e)[:200]}") from e

    @property
    def plan_stop_file(self) -> Path:
        return self.cfg.cache_dir / "claude-plan-stop.json"

    def _claude_exe(self) -> str:
        """Locate `claude` and confirm it is logged in to a subscription (once per run)."""
        from . import claude_code as cc

        with self._lock:
            if self._cc_exe is None:
                exe = cc.find_claude(self.cfg.judging.claude_code.path)
                if exe is None:
                    raise cc.ClaudeCodeError("not_installed", f"Claude Code (`claude`) not found. {cc.INSTALL_HINT}",
                                             fatal=True)
                cc.check_subscription(exe)
                self._cc_exe = exe
            return self._cc_exe

    def _claude_code(self, *, model: str, system: list[dict], messages: list[dict], schema: dict, effort: str,
                     max_tokens: int, purpose: str) -> tuple[dict, list[Usage], str, list[str]]:
        from . import claude_code as cc

        ccfg = self.cfg.judging.claude_code
        try:
            stop = cc.plan_stop(self.plan_stop_file)
            if stop is not None:
                raise stop
            run = cc.run_structured(self._claude_exe(), model=model, system=system, messages=messages, schema=schema,
                                    effort=effort, max_tokens=max_tokens, timeout=ccfg.timeout_s,
                                    allow_extra_usage=ccfg.allow_extra_usage, state_file=self.plan_stop_file)
        except cc.ClaudeCodeError as e:
            if e.rate_limit.get("isUsingOverage"):
                self.tracker.extra_usage_used = True
            err = LLMError(e.kind, str(e))
            if e.kind in cc.PLAN_KINDS and not e.from_stop:
                # Shared by every scan in this process (and remembered on disk until the reset), so the
                # next queued video or calibration clip doesn't spend one more paid call to find out.
                # A login/billing problem is not shared: each new scan re-checks `claude auth status`.
                cc.set_plan_stop(e.kind, str(e), cc.stop_until(e.rate_limit), self.plan_stop_file)
            if e.fatal:
                self._set_fatal(err)
            raise err from e
        if run.overage_seen:  # only reachable with allow_extra_usage = true
            self.tracker.extra_usage_used = True
        if run.rate_limit.get("status") == "allowed_warning":
            self._warn_once("limit", "Heads up: Claude Code reports you are close to your plan's usage limit.")
        return run.data, run.usages, run.served_model, run.answer_models

    def _warn_once(self, key: str, text: str) -> None:
        with self._lock:
            if key in self._warned:
                return
            self._warned.add(key)
        from .ui import console

        console.print(f"  [yellow]{text}[/yellow]")

    def _parse_response(self, resp, model: str, purpose: str, rh: str) -> tuple[dict, list[Usage], str]:
        usages = usages_from_response(resp, model)
        served = getattr(resp, "model", model) or model
        if resp.stop_reason == "refusal":
            self.tracker.record(purpose, usages, cached=False)
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            self._store_error(purpose, rh, "refusal", f"declined (category={cat})", usages)
            raise LLMError("refusal", f"model declined (category={cat})")
        if resp.stop_reason == "max_tokens":
            self.tracker.record(purpose, usages, cached=False)
            raise LLMError("truncated", "hit max_tokens before finishing the JSON")
        text = "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            self.tracker.record(purpose, usages, cached=False)
            raise LLMError("invalid_json", text[:300]) from e
        return data, usages, served
