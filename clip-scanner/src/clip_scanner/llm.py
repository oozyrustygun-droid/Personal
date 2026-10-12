"""Claude calls with structured JSON output, request-level caching and cost tracking.

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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from .cache import read_json, write_json
from .config import Config
from .cost import CostTracker, Usage, usages_from_response

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side `fallbacks: "default"` parameter on the Claude API.
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class LLMError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class LLMResult:
    data: dict
    usages: list[Usage]
    cached: bool
    served_model: str
    request_hash: str


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


def _hash_request(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


class LLM:
    def __init__(self, cfg: Config, store_dir: Path, tracker: CostTracker, *, backend: str = "anthropic",
                 script: ScriptFn | None = None):
        self.cfg = cfg
        self.store_dir = Path(store_dir)
        self.tracker = tracker
        self.backend = backend
        self.script = script
        self._client = None
        self._lock = threading.Lock()
        self.refresh = False  # True: ignore cached responses (--rerun judging)

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

    def request_hash(self, *, model: str, system: list[dict], messages: list[dict], schema: dict,
                     effort: str) -> str:
        return _hash_request({"backend": self.backend, "model": model, "system": system, "messages": messages,
                              "schema": schema, "effort": effort})

    def call(self, *, purpose: str, model: str, system: list[dict], messages: list[dict],
             output: type[BaseModel], effort: str, max_tokens: int, meta: dict | None = None) -> LLMResult:
        schema = strict_schema(output)
        rh = self.request_hash(model=model, system=system, messages=messages, schema=schema, effort=effort)
        if self.backend == "scripted":
            rh = _hash_request({"scripted": meta or {}, "purpose": purpose, "h": rh})
        hit = None if self.refresh else self.cached_result(purpose, rh)
        if hit is not None:
            usages = [Usage.from_json(u) for u in hit.get("usages", [])]
            self.tracker.record(purpose, usages, cached=True)
            if "error" in hit:
                raise LLMError(hit["error"]["kind"], hit["error"]["message"] + " (cached)")
            return LLMResult(data=self._validate(output, hit["data"]), usages=usages, cached=True,
                             served_model=hit.get("served_model", model), request_hash=rh)

        if self.backend == "scripted":
            if self.script is None:
                raise LLMError("config", "scripted backend without a script")
            data = self.script(purpose, meta or {})
            usages = [Usage(model=model)]
            served = model
        else:
            data, usages, served = self._anthropic(model=model, system=system, messages=messages, schema=schema,
                                                   effort=effort, max_tokens=max_tokens, purpose=purpose, rh=rh)
        self.tracker.record(purpose, usages, cached=False)
        validated = self._validate(output, data, purpose=purpose, rh=rh, usages=usages)
        write_json(self.store_dir / purpose / f"{rh}.json",
                   {"data": data, "usages": [u.to_json() for u in usages], "served_model": served,
                    "requested_model": model})
        return LLMResult(data=validated, usages=usages, cached=False, served_model=served, request_hash=rh)

    def _validate(self, output: type[BaseModel], data: dict, *, purpose: str = "", rh: str = "",
                  usages: list[Usage] | None = None) -> dict:
        try:
            return output.model_validate(data).model_dump()
        except ValidationError as e:
            if rh:
                self._store_error(purpose, rh, "schema", str(e)[:2000], usages or [])
            raise LLMError("schema", str(e)[:500]) from e

    def _store_error(self, purpose: str, rh: str, kind: str, message: str, usages: list[Usage]) -> None:
        # Deterministic failures (refusal, schema) are cached so a re-run doesn't pay for them again.
        write_json(self.store_dir / purpose / f"{rh}.json",
                   {"error": {"kind": kind, "message": message}, "usages": [u.to_json() for u in usages]})

    def _anthropic(self, *, model: str, system: list[dict], messages: list[dict], schema: dict, effort: str,
                   max_tokens: int, purpose: str, rh: str) -> tuple[dict, list[Usage], str]:
        import anthropic

        kwargs: dict[str, Any] = dict(
            model=model, max_tokens=max_tokens, system=system, messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        )
        try:
            if self.cfg.judging.server_fallbacks and model in FALLBACK_MODELS:
                resp = self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                resp = self.client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("auth", "Anthropic credentials missing or invalid (set ANTHROPIC_API_KEY)") from e
        except TypeError as e:  # SDK raises this when no credential source resolves at all
            if "authentication" in str(e).lower():
                raise LLMError("auth", "no Anthropic credentials found (set ANTHROPIC_API_KEY or `ant auth login`)") from e
            raise
        except anthropic.BadRequestError as e:
            raise LLMError("bad_request", str(e)[:500]) from e
        except anthropic.RateLimitError as e:
            raise LLMError("rate_limit", str(e)[:300]) from e
        except anthropic.APIStatusError as e:
            raise LLMError("api", f"{e.status_code}: {str(e)[:300]}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("connection", str(e)[:300]) from e

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
