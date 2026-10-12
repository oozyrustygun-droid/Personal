"""The real Anthropic code path against a local mock of the Messages API.

Checks the exact request we send (structured output, adaptive thinking, effort, cache_control,
server-side fallbacks) and how responses, usage, refusals and bad JSON are handled.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import pytest
from pydantic import BaseModel, Field

from clip_scanner.config import Config
from clip_scanner.cost import CostTracker
from clip_scanner.llm import LLM, LLMError, strict_schema


class Out(BaseModel):
    reason: str
    score: float = Field(ge=0, le=10)


class Mock:
    def __init__(self):
        self.requests = []
        self.responses = []
        mock = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                mock.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                resp = mock.responses.pop(0)
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"


def message(text, stop="end_turn", usage=None, model="claude-opus-5-5"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "stop_reason": stop,
            "stop_sequence": None,
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
            "usage": usage or {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 4000,
                               "cache_read_input_tokens": 0}}


@pytest.fixture()
def mock():
    m = Mock()
    yield m
    m.server.shutdown()


def make(mock, tmp_path):
    cfg = Config()
    tracker = CostTracker(cfg)
    llm = LLM(cfg, tmp_path / "llm", tracker)
    llm._client = anthropic.Anthropic(api_key="test-key", base_url=mock.url, max_retries=0)
    return llm, tracker, cfg


SYSTEM = [{"type": "text", "text": "handbook", "cache_control": {"type": "ephemeral"}},
          {"type": "text", "text": "persona"}]
MSGS = [{"role": "user", "content": "judge this"}]


def test_request_shape_and_cost(mock, tmp_path):
    llm, tracker, cfg = make(mock, tmp_path)
    mock.responses.append(message(json.dumps({"reason": "ok", "score": 9.1})))
    res = llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                   effort="high", max_tokens=16000)
    assert res.data == {"reason": "ok", "score": 9.1} and not res.cached
    req = mock.requests[0]
    body = req["body"]
    assert req["path"].startswith("/v1/messages")
    # Verdict calls never ask for a substitute model: a refusal must stay a refusal.
    assert "fallbacks" not in body
    assert "server-side-fallback" not in req["headers"].get("anthropic-beta", "")
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"]["effort"] == "high"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert "maximum" not in json.dumps(fmt["schema"])  # numeric bounds stripped, re-checked locally
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in body
    # $4/M input, $20/M output, $5/M 5-minute cache write
    assert tracker.total() == pytest.approx((1000 * 4 + 500 * 20 + 4000 * 5) / 1e6)

    # Identical request: served from the on-disk cache, no HTTP call, $0.
    again = llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                     effort="high", max_tokens=16000)
    assert again.cached and len(mock.requests) == 1
    assert tracker.total() == pytest.approx((1000 * 4 + 500 * 20 + 4000 * 5) / 1e6)


def test_haiku_has_no_fallbacks_param(mock, tmp_path):
    llm, _, _ = make(mock, tmp_path)
    mock.responses.append(message(json.dumps({"reason": "ok", "score": 3}), model="claude-haiku-5-5"))
    llm.call(purpose="proposer", model="claude-haiku-5-5", system=SYSTEM, messages=MSGS, output=Out, effort="low",
             max_tokens=8000)
    assert "fallbacks" not in mock.requests[0]["body"]


def test_out_of_range_score_fails_closed_and_is_cached(mock, tmp_path):
    llm, _, _ = make(mock, tmp_path)
    mock.responses.append(message(json.dumps({"reason": "wow", "score": 11})))
    with pytest.raises(LLMError) as e:
        llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out, effort="high",
                 max_tokens=16000)
    assert e.value.kind == "schema"
    with pytest.raises(LLMError):  # cached failure: no second HTTP call
        llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out, effort="high",
                 max_tokens=16000)
    assert len(mock.requests) == 1


def test_refusal_and_truncation_raise(mock, tmp_path):
    llm, tracker, _ = make(mock, tmp_path)
    mock.responses.append(message("", stop="refusal"))
    with pytest.raises(LLMError) as e:
        llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out, effort="high",
                 max_tokens=16000)
    assert e.value.kind == "refusal"
    mock.responses.append(message('{"reason": "trunc', stop="max_tokens"))
    with pytest.raises(LLMError) as e2:
        llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=[{"role": "user", "content": "b"}],
                 output=Out, effort="high", max_tokens=16000)
    assert e2.value.kind == "truncated"
    assert tracker.total() > 0  # failed calls are still billed and counted


def test_fallback_iterations_are_all_billed(mock, tmp_path):
    llm, tracker, _ = make(mock, tmp_path)
    usage = {"input_tokens": 100, "output_tokens": 50, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
             "iterations": [
                 {"type": "message", "model": "claude-opus-5-5", "input_tokens": 1000, "output_tokens": 10,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                 {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 100, "output_tokens": 50,
                  "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}]}
    mock.responses.append(message(json.dumps({"reason": "ok", "score": 5}), usage=usage, model="claude-opus-4-8"))
    res = llm.call(purpose="packaging", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                   effort="high", max_tokens=16000, allow_fallback=True)
    req = mock.requests[0]
    assert req["body"]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert res.served_model == "claude-opus-4-8" and res.fallback_used
    assert len(res.usages) == 2
    opus = (1000 * 4 + 10 * 20) / 1e6
    assert tracker.total() >= opus


def test_strict_schema_requires_all_fields():
    s = strict_schema(Out)
    assert set(s["required"]) == {"reason", "score"}
    assert s["additionalProperties"] is False


def test_fallback_flag_is_part_of_the_cache_key(mock, tmp_path):
    """A cached answer from a fallback-enabled request must not be served to a fallback-free request."""
    llm, _, _ = make(mock, tmp_path)
    mock.responses.append(message(json.dumps({"reason": "a", "score": 9}), model="claude-sonnet-5-5"))
    first = llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                     effort="high", max_tokens=16000, allow_fallback=True)
    assert first.fallback_used
    mock.responses.append(message(json.dumps({"reason": "b", "score": 6})))
    second = llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                      effort="high", max_tokens=16000)
    assert not second.cached and not second.fallback_used and second.data["score"] == 6
    assert len(mock.requests) == 2


def test_unexpected_sdk_error_fails_closed(mock, tmp_path):
    llm, _, _ = make(mock, tmp_path)
    mock.responses.append({"not": "a message"})  # malformed body -> SDK can't build a Message
    with pytest.raises(LLMError):
        llm.call(purpose="judge", model="claude-opus-5-5", system=SYSTEM, messages=MSGS, output=Out,
                 effort="high", max_tokens=16000)
