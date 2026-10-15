"""Top-moments scan: ingest → transcribe → signals → moments → ranking → raw export → report."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import top as T
from .cache import fingerprint, read_json
from .candidates import cached_proposals, candidates_stage
from .cli_common import BACKEND_LABEL, resolve_backend
from .config import Config
from .cost import CostTracker, estimate_tokens
from .export import (STASH, _render_cached, make_poster, prune_stale_clips, render_span, restore_all,
                     restore_exports, slug, stash_exports)
from .ui import console, emit, fmt_ts, stage

MOMENTS_DIR = "clips"


def moment_path(out_dir: Path, m: T.Moment) -> Path:
    t = int(m.start)
    return out_dir / MOMENTS_DIR / f"{m.rank:02d}-{slug(m.title)}-{t // 60}m{t % 60:02d}s.mp4"


def moment_fingerprint(video_id: str, m: T.Moment, cfg: Config) -> str:
    return fingerprint("render", "top", video_id, round(m.start, 3), round(m.end, 3),
                       cfg.section_hash("export"))


def ranking_estimate(cfg: Config, request: dict, regions: list[T.Region], llm) -> dict:
    """Size of the one ranking call, in the shape the approval flow uses (nothing to pay if it is cached)."""
    price = cfg.price(cfg.judging.model)
    text = request["system"][0]["text"] + request["messages"][0]["content"]
    usd = (estimate_tokens(text) * price.input + 2500 * price.output) / 1e6
    if llm.is_cached(**request):
        usd = 0.0
    return {"candidates": len(regions), "eligible": len(regions), "uncached": int(usd > 0),
            "model": cfg.judging.model, "effort": cfg.top.effort, "expected_usd": round(usd, 4),
            "worst_case_usd": round(usd * 2, 4), "billing": llm.billing, "expected_calls": 1, "worst_case_calls": 1}


def previous_was_ranked(out_dir: Path) -> bool:
    """The finished result this scan set aside (export.stash_exports) was ranked by Claude."""
    try:
        rep = read_json(out_dir / STASH / "report.json")
    except (OSError, ValueError):
        return False
    return rep.get("mode") == "top" and (rep.get("ranking") or {}).get("method") == "claude"


def run_top(args: argparse.Namespace, cfg: Config, confirm) -> dict:
    from .pipeline import front_stages, make_llm
    from .report import out_dir_for, print_top_report, write_in_progress, write_top_reports

    count = args.max_clips or cfg.top.count
    rerun = set(args.rerun)
    backend = resolve_backend(cfg, args.judge_backend)
    scripted = backend == "scripted"
    state: dict = {}

    def on_ingested(meta) -> None:
        state["out_dir"] = out_dir = out_dir_for(meta, cfg, scripted=scripted, mode="top")
        if not args.dry_run:
            stash_exports(out_dir)
            write_in_progress(meta, cfg, out_dir, scripted=scripted, backend=backend, mode="top")

    meta, ws, transcript, sig, s_fp = front_stages(args.source, cfg, rerun=rerun, on_ingested=on_ingested)
    out_dir = state["out_dir"]
    tracker = CostTracker(cfg)
    llm = make_llm(cfg, ws, tracker, args.judge_backend, args.scripted_judge, refresh="judging" in rerun)
    notes: list[str] = []
    recall_incomplete = False

    with stage("Moments") as st:
        segments = T.load_heatmap(ws)
        has_heat = T.heat_series(segments, meta.duration) is not None
        if not has_heat and cfg.top.use_claude and cfg.candidates.llm_proposer:
            # No audience data: the transcript proposer is the best stand-in for it (cached after the first
            # run; a run that missed part of the transcript is finished here).
            console.print(f"  [dim]no \"Most replayed\" data: Claude reads the transcript for moments "
                          f"({BACKEND_LABEL.get(backend, backend)})[/dim]")
            candidates_stage(ws, transcript, sig, s_fp, meta.duration, cfg, llm, console=console,
                             force="candidates" in rerun)
            if read_json(ws.path("candidates.json")).get("proposer_incomplete"):
                recall_incomplete = True
                notes.append("Claude couldn't read part of the transcript this run, so moments there may be "
                             "missed; re-run to finish it (the rest is cached).")
        # With audience data, a transcript read from an earlier scan is reused (free) but not started.
        proposals = cached_proposals(ws, s_fp, cfg, llm) or []
        score, parts, heat = T.curve(segments, sig, meta.duration, proposals, cfg)
        regions = T.find_regions(score, parts, heat, meta.duration, transcript, proposals, cfg,
                                 limit=max(cfg.top.candidates, count), need=count)
        st["note"] = (f"{len(regions)} candidate region(s) · \"Most replayed\" "
                      f"{'used' if heat is not None else 'not available'}"
                      + (f" · {len(proposals)} transcript flag(s)" if proposals else ""))
    for r in regions:
        console.print(f"  [dim]{fmt_ts(r.start)}–{fmt_ts(r.end)} peak {fmt_ts(r.peak)} score {r.score:.2f}"
                      f" · {', '.join(r.sources) or 'interest peak'}[/dim]")
    if args.dry_run:
        console.print("[dim]--dry-run: stopped before ranking.[/dim]")
        return {"code": 0, "out_dir": out_dir, "status": "dry_run"}

    with stage("Ranking") as st:
        use_claude = cfg.top.use_claude and bool(regions)
        declined = False
        if use_claude and llm.billing != "subscription":
            # Paid API (or the offline test judge, which follows the API rule): spend over the limit is approved
            # first, exactly as in great mode. A cached ranking costs nothing and is never asked about.
            est = ranking_estimate(cfg, T.ranking_request(regions, transcript, meta.title, meta.duration, cfg,
                                                          has_heat, count), regions, llm)
            emit("estimate", **est, threshold=cfg.judging.confirm_above_usd)
            if est["uncached"]:
                console.print(f"  ranking with {cfg.judging.model}: ~${est['expected_usd']:.2f} at API prices")
            declined = not confirm(est, cfg)
        if use_claude and not declined:
            console.print(f"  [dim]ranking on {BACKEND_LABEL.get(backend, backend)}[/dim]")
        result = T.rank(llm if use_claude and not declined else None, regions, transcript, meta.title,
                        meta.duration, cfg, has_heat, count, console=console)
        result.notes = notes + result.notes
        result.recall_incomplete = recall_incomplete
        picker = ("the scripted test ranking" if scripted else "Claude") if result.method == "claude" \
            else "measured signals"
        st["note"] = ("cancelled: not approved" if declined
                      else f"{len(result.moments)} moment(s), picked by {picker}")

    # Claude was meant to rank but didn't. Never replace a finished Claude-ranked result for this video with
    # a weaker one: put it back. A declined approval means cancel: nothing is ranked or exported.
    failed = use_claude and not declined and result.method != "claude"
    if declined or (failed and not result.error_final and previous_was_ranked(out_dir)):
        why = ("Ranking wasn't approved, so nothing was ranked or exported." if declined
               else f"Claude couldn't rank the moments this time ({result.error}). Your earlier results for this "
                    f"video were kept; re-run later to refresh them.")
        result.notes = [n for n in result.notes if not n.startswith("Claude couldn't rank")]
        # stash_exports only sets a FINISHED report aside. Without one, leave any stashed renders out of sight
        # (a later approved run reuses those that still match) so the folder matches this report.
        if (out_dir / STASH / "report.json").exists() and restore_all(out_dir):
            console.print(f"[yellow]{why} The previous report and clips were left in place.[/yellow]")
        else:
            result.moments, result.cancelled, result.method = [], True, "none"
            result.notes = [*notes, why]
            write_top_reports(meta, transcript, sig, result, cfg, out_dir, {}, tracker, llm, scripted)
            console.print(f"[yellow]{why}[/yellow]")
        return {"code": 3, "out_dir": out_dir, "status": "unavailable", "notes": [*notes, why]}

    keep = {str(moment_path(out_dir, m)): moment_fingerprint(meta.video_id, m, cfg) for m in result.moments}
    restore_exports(out_dir, keep)
    exported: dict[int, dict] = {}
    if result.moments and not args.no_cut:
        with stage("Export"):
            for m in result.moments:
                path = moment_path(out_dir, m)
                info: dict = {"file": None, "poster": None}
                try:
                    _render_cached(path, keep[str(path)],
                                   lambda m=m, path=path: render_span(meta.path, m.start, m.end, path, cfg),
                                   {"span": [m.start, m.end], "source": meta.source, "title": m.title}, "export" in rerun)
                    info["file"] = str(path)
                    poster = path.with_suffix(".jpg")
                    if not poster.exists() or poster.stat().st_mtime < path.stat().st_mtime:
                        make_poster(path, poster, max(0.0, min(m.focus - m.start, m.duration - 0.5)))
                    info["poster"] = str(poster) if poster.exists() else None
                except Exception as e:  # one failed render shouldn't lose the report
                    console.print(f"  [red]export failed for moment {m.rank}: {e}[/red]")
                exported[m.rank] = info
    else:
        with stage("Export") as st:
            st["skipped"] = True
            st["note"] = "export is off" if args.no_cut else "nothing to export"
    for p in prune_stale_clips(out_dir, keep):
        console.print(f"  [dim]removed an older export: {p.name}[/dim]")
    paths = write_top_reports(meta, transcript, sig, result, cfg, out_dir, exported, tracker, llm, scripted)
    print_top_report(meta, paths)
    status = paths["report"]["status"]
    return {"code": 0, "out_dir": out_dir, "status": status, "notes": result.notes}

