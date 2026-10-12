"""A stand-in for the Claude Code CLI, driven by environment variables (see tests/test_claude_code.py).

FAKE_CLAUDE_AUTH   claude.ai | oauth_token | api_key | none        (`claude auth status`)
FAKE_CLAUDE_PROVIDER  apiProvider reported by `claude auth status` (default firstParty)
FAKE_CLAUDE_MODE   ok | init_api_key | usage_limit | overage | overage_error | fallback | auth_fail | hang |
                   slow | no_structured | schema_retries | warning
                   (a request whose text contains "FAKE_MODE:<mode>" uses that mode instead)
FAKE_CLAUDE_RESPONSES  JSON file: {schema property name: structured output}; the first property name
                   found in the request's --json-schema picks the answer
FAKE_CLAUDE_LOG    JSONL file: one record per `-p` run (argv, env flags, system prompt, stdin message)
"""

import json
import os
import sys
import time


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def arg(argv, name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    argv = sys.argv[1:]
    auth = os.environ.get("FAKE_CLAUDE_AUTH", "claude.ai")
    if argv[:2] == ["auth", "status"]:
        if auth == "none":
            print(json.dumps({"loggedIn": False, "authMethod": "none"}))
            return 1
        print(json.dumps({"loggedIn": True, "authMethod": auth,
                          "apiProvider": os.environ.get("FAKE_CLAUDE_PROVIDER", "firstParty")}))
        return 0
    if argv == ["--version"]:
        print("9.9.9 (Claude Code)")
        return 0
    assert "-p" in argv, argv
    mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
    model = arg(argv, "--model")
    schema = json.loads(arg(argv, "--json-schema"))
    message = json.loads(sys.stdin.readline())
    text = " ".join(b.get("text", "") for b in message["message"]["content"] if isinstance(b, dict))
    if "FAKE_MODE:" in text:
        mode = text.split("FAKE_MODE:", 1)[1].split()[0]
    system = open(arg(argv, "--system-prompt-file"), encoding="utf-8").read()
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": argv, "system": system, "message": message, "cwd": os.getcwd(),
                                "env": {k: os.environ.get(k) for k in (
                                    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK",
                                    "ANTHROPIC_BASE_URL", "ANTHROPIC_PROFILE", "CLAUDE_CODE_USE_MANTLE",
                                    "AWS_BEARER_TOKEN_BEDROCK", "CLAUDE_CODE_OAUTH_TOKEN",
                                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_DISABLE_CLAUDE_MDS",
                                    "CLAUDE_CODE_DISABLE_AUTO_MEMORY")}}) + "\n")
    emit({"type": "system", "subtype": "init", "model": model, "tools": ["StructuredOutput"],
          "apiKeySource": "ANTHROPIC_API_KEY" if mode == "init_api_key" else "none"})
    if mode in ("init_api_key", "hang"):
        time.sleep(30)  # the caller must kill us
        return 0
    if mode in ("overage", "overage_error"):
        # The real CLI reports this from the response headers, before the answer streams.
        # As the real CLI does: the plan's included usage is "rejected" and paid overage is carrying on.
        emit({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "isUsingOverage": True,
                                                              "resetsAt": 1893456000}})
        if mode == "overage_error":
            emit({"type": "result", "subtype": "error_max_structured_output_retries", "is_error": True,
                  "errors": ["validation failed"], "modelUsage": {}})
            return 1
    if mode == "slow":
        time.sleep(float(os.environ.get("FAKE_CLAUDE_SLOW", "20")))
    if mode == "usage_limit":
        emit({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": 1893456000}})
        emit({"type": "assistant", "parent_tool_use_id": None, "message": {
            "model": "<synthetic>", "content": [{"type": "text", "text": "Claude AI usage limit reached"}]}})
        emit({"type": "result", "subtype": "success", "is_error": True, "result": "Claude AI usage limit reached",
              "api_error_status": 429, "modelUsage": {}})
        return 1
    if mode == "auth_fail":
        emit({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 401,
              "result": "Invalid API key · Please run /login", "modelUsage": {}})
        return 1
    if mode == "schema_retries":
        emit({"type": "result", "subtype": "error_max_structured_output_retries", "is_error": True,
              "errors": ["validation failed"], "modelUsage": {}})
        return 1
    with open(os.environ["FAKE_CLAUDE_RESPONSES"], encoding="utf-8") as f:
        responses = json.load(f)
    props = list(schema.get("properties", {}))
    answer = next((responses[p] for p in props if p in responses), None)
    served = "claude-sonnet-5-5" if mode == "fallback" else model
    emit({"type": "assistant", "parent_tool_use_id": None, "message": {
        "model": served, "content": [{"type": "thinking", "thinking": ""}]}})
    emit({"type": "assistant", "parent_tool_use_id": None, "message": {
        "model": served, "content": [{"type": "tool_use", "name": "StructuredOutput", "input": answer or {}}]}})
    rl = {"status": "rejected" if mode == "overage" else "allowed_warning" if mode == "warning" else "allowed",
          "isUsingOverage": mode == "overage", "resetsAt": 1893456000}
    emit({"type": "rate_limit_event", "rate_limit_info": rl})
    usage = {served: {"inputTokens": 1200, "outputTokens": 300, "cacheReadInputTokens": 0,
                      "cacheCreationInputTokens": 5000, "costUSD": 0.05}}
    result = {"type": "result", "subtype": "success", "is_error": False, "result": json.dumps(answer),
              "modelUsage": usage, "total_cost_usd": 0.05}
    if mode != "no_structured":
        result["structured_output"] = answer
    emit(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
