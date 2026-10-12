"""Token accounting and dollar costs for every Claude call."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .config import Config


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self, cfg: Config, ttl: str = "5m") -> float:
        p = cfg.price(self.model)
        write = p.cache_write_1h if ttl == "1h" else p.cache_write_5m
        return (self.input_tokens * p.input + self.output_tokens * p.output
                + self.cache_write_tokens * write + self.cache_read_tokens * p.cache_read) / 1e6

    def to_json(self) -> dict:
        return dict(model=self.model, input_tokens=self.input_tokens, output_tokens=self.output_tokens,
                    cache_write_tokens=self.cache_write_tokens, cache_read_tokens=self.cache_read_tokens)

    @classmethod
    def from_json(cls, d: dict) -> "Usage":
        return cls(**d)


def usages_from_response(resp, requested_model: str) -> list[Usage]:
    """Per-attempt usage. With server-side fallbacks, `usage.iterations` lists every attempt
    (a declined attempt can still bill), so read it when present and fall back to top-level usage."""
    u = resp.usage
    iterations = getattr(u, "iterations", None) or []
    out: list[Usage] = []
    for it in iterations:
        model = getattr(it, "model", None) or (requested_model if getattr(it, "type", "") == "message"
                                                 else getattr(resp, "model", requested_model))
        out.append(Usage(model=model, input_tokens=getattr(it, "input_tokens", 0) or 0,
                         output_tokens=getattr(it, "output_tokens", 0) or 0,
                         cache_write_tokens=getattr(it, "cache_creation_input_tokens", 0) or 0,
                         cache_read_tokens=getattr(it, "cache_read_input_tokens", 0) or 0))
    if not out:
        out.append(Usage(model=getattr(resp, "model", None) or requested_model,
                         input_tokens=u.input_tokens or 0, output_tokens=u.output_tokens or 0,
                         cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                         cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0))
    return out


@dataclass
class CostTracker:
    cfg: Config
    calls: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, purpose: str, usages: list[Usage], cached: bool) -> None:
        with self._lock:
            for u in usages:
                self.calls.append({"purpose": purpose, "cached": cached, **u.to_json(),
                                   "usd": 0.0 if cached else u.cost(self.cfg, self.cfg.judging.cache_ttl)})

    def total(self, *, include_cached: bool = False) -> float:
        return sum(c["usd"] for c in self.calls if include_cached or not c["cached"])

    def summary(self) -> dict:
        by: dict[str, dict] = {}
        for c in self.calls:
            b = by.setdefault(c["purpose"], {"calls": 0, "cached_calls": 0, "usd": 0.0, "input_tokens": 0,
                                             "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0})
            b["calls"] += 1
            b["cached_calls"] += int(c["cached"])
            b["usd"] += c["usd"]
            if not c["cached"]:
                for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                    b[k] += c[k]
        return {"total_usd": round(self.total(), 4), "by_purpose": by}


def estimate_tokens(text: str) -> int:
    """Offline token estimate (~3.6 chars/token for English prose). Used only before calling the API."""
    return max(1, int(len(text) / 3.6))


def image_tokens(width: int, height: int) -> int:
    return max(1, int(width * height / 750))
