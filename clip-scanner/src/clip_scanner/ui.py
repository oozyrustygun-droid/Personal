"""Console helpers (rich)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable

from rich.console import Console

console = Console(highlight=False)

# Structured progress events for the web UI: listener(kind, data). The CLI has none.
listeners: list[Callable[[str, dict], None]] = []


def emit(kind: str, **data) -> None:
    for fn in list(listeners):
        try:
            fn(kind, data)
        except Exception:  # a broken listener must never break a scan
            pass


@contextmanager
def stage(name: str, *, cached_msg: str = "cached"):
    """Print a stage header and its wall time. Yields a dict whose 'cached' / 'skipped' flags the caller can set."""
    state = {"cached": False, "skipped": False, "note": ""}
    console.print(f"[bold cyan]▸ {name}[/bold cyan]")
    emit("stage_start", name=name)
    t0 = time.perf_counter()
    try:
        yield state
    finally:
        dt = time.perf_counter() - t0
        tag = ("[dim]skipped[/dim]" if state["skipped"] else f"[green]{cached_msg}[/green]" if state["cached"]
               else f"[dim]{dt:.1f}s[/dim]")
        note = f" [dim]{state['note']}[/dim]" if state["note"] else ""
        console.print(f"  {tag}{note}")
        emit("stage_end", name=name, seconds=round(dt, 1), cached=state["cached"], skipped=state["skipped"],
             note=state["note"])


def fmt_ts(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    frac = seconds - int(seconds)
    base = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    return f"{base}.{int(frac * 10)}"


def fmt_usd(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.995 or x == 0 else f"${x:,.3f}"
