"""Terminal + report.md + report.json."""

from __future__ import annotations

import time
from pathlib import Path

from rich.panel import Panel
from rich.table import Table

from . import __version__
from .cache import write_json
from .config import Config
from .judging.panel import near_miss_key
from .ui import console, fmt_ts, fmt_usd

PERSONA_SHORT = {"jaded_scroller": "scroller", "joe_superfan": "superfan", "shorts_editor": "editor"}
STAGE_LABEL = {"deterministic_gates": "hard gate (measured)", "panel": "judging panel",
               "devils_advocate": "devil's advocate", "error": "judging error", "great": "GREAT"}


def out_dir_for(a, cfg: Config) -> Path:
    from .export import slug

    return cfg.output_dir / f"{slug(a.meta.title, 50)}-{a.meta.video_id[-10:]}"


def verdict_label(result) -> str:
    if result.great:
        n = len(result.great)
        return f"{n} GREAT CLIP{'S' if n > 1 else ''}"
    if result.status == "unavailable":
        return "JUDGING UNAVAILABLE"
    if result.status == "incomplete":
        return "INCOMPLETE (no GREAT clips among the candidates that could be judged)"
    return "NO GREAT CLIPS"


def _judge_rows(d) -> list[dict]:
    rows = []
    for a in d.assessments:
        rows.append({"judge": a["persona"], "overall": a["overall"], "rubric_mean": a["rubric_mean"],
                     "effective": a["effective"], "failed_gates": a["failed_gates"], "ok": a["ok"]})
    return rows


def _scores_line(d) -> str:
    parts = []
    for a in d.assessments:
        name = PERSONA_SHORT.get(a["persona"], a["persona"])
        if a["overall"] is None:
            parts.append(f"{name} —")
        else:
            parts.append(f"{name} {a['effective']:.1f}" + ("" if a["ok"] and not a["failed_gates"] else "✗"))
    return " · ".join(parts)


def _clip_json(rank, d, c, pkg, exported) -> dict:
    return {
        "rank": rank, "id": c.id, "start": round(c.start, 2), "end": round(c.end, 2),
        "start_ts": fmt_ts(c.start), "end_ts": fmt_ts(c.end), "duration": round(c.duration, 2),
        "segments": [list(s) for s in c.segments], "jump_cuts": c.jump_cuts,
        "score": d.min_effective, "mean_score": d.mean_effective, "judges": _judge_rows(d),
        "hook_line": pkg.get("hook_line", ""), "why_viral": pkg.get("why_viral", ""),
        "caption_title": pkg.get("caption_title", ""), "posting_hook_text": pkg.get("posting_hook_text", ""),
        "hashtags": pkg.get("hashtags", []),
        "devils_advocate": (d.devil or {}).get("strongest_flop_argument", ""),
        "arbiter": {k: (d.arbiter or {}).get(k) for k in ("survival_confidence", "decisive_issue", "analysis")},
        "file": exported.get(c.id),
    }


def _reject_json(d, c) -> dict:
    return {"id": c.id, "start": round(c.start, 2), "end": round(c.end, 2), "start_ts": fmt_ts(c.start),
            "end_ts": fmt_ts(c.end), "duration": round(c.duration, 2), "killed_at": d.stage,
            "reasons": d.reasons, "judges": _judge_rows(d), "origin": c.origin, "proposal_reason": c.reason,
            "devils_advocate": (d.devil or {}).get("strongest_flop_argument"),
            "arbiter_confidence": (d.arbiter or {}).get("survival_confidence")}


def build_report(a, result, cfg: Config, exported: dict | None = None) -> dict:
    exported = exported or {}
    by_id = {c.id: c for c in a.candidates}
    great_ids = {d.candidate_id for d in result.great}
    rejected = [d for d in result.decisions if d.candidate_id not in great_ids]
    near = sorted([d for d in rejected if d.stage != "error"], key=near_miss_key)[:3]
    j = cfg.judging
    return {
        "tool": f"clip-scanner {__version__}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": {"id": a.meta.video_id, "title": a.meta.title, "source": a.meta.source, "url": a.meta.url,
                  "uploader": a.meta.uploader, "duration": round(a.meta.duration, 1)},
        "verdict": verdict_label(result),
        "status": result.status,
        "bar": {"judges": j.judges, "model": j.model, "effort": j.effort, "overall_min": j.overall_min,
                "rubric_mean_min": j.rubric_mean_min, "criterion_floors": j.criterion_floors,
                "gate_policy": j.gate_policy, "devils_advocate": cfg.devils_advocate.enabled,
                "min_survival_confidence": cfg.devils_advocate.min_survival_confidence},
        "great_clips": [_clip_json(i, d, by_id[d.candidate_id], result.packaging.get(d.candidate_id, {}), exported)
                        for i, d in enumerate(result.great, 1)],
        "near_misses": [_reject_json(d, by_id[d.candidate_id]) for d in (near if not result.great else [])],
        "diagnosis": result.diagnosis,
        "rejected": [_reject_json(d, by_id[d.candidate_id]) for d in rejected],
        "candidates": len(a.candidates),
        "transcript": {"backend": a.transcript.backend, "model": a.transcript.model, "words": len(a.transcript.words)},
        "signals_notes": a.signals.notes,
        "notes": result.notes,
        "cost": a.tracker.summary(),
    }


def render_markdown(rep: dict, explain_rejects: bool) -> str:
    v = rep["video"]
    b = rep["bar"]
    L = [f"# clip-scanner report: {v['title']}", "",
         f"- Source: {v['url'] or v['source']}",
         f"- Duration: {fmt_ts(v['duration'])} · candidates proposed: {rep['candidates']} · transcript: "
         f"{rep['transcript']['backend']}/{rep['transcript']['model']}",
         f"- Bar: every one of {len(b['judges'])} judges ≥ {b['overall_min']} (rubric mean ≥ {b['rubric_mean_min']}, "
         f"floors {b['criterion_floors']}), all gates pass ({b['gate_policy']}), devil's advocate "
         f"{'on' if b['devils_advocate'] else 'off'} · {b['model']} @ {b['effort']}",
         f"- Generated {rep['generated_at']} by {rep['tool']}", "",
         f"## Verdict: {'✅' if rep['great_clips'] else '❌'} {rep['verdict']}", ""]
    if rep["diagnosis"]:
        L += [f"> {rep['diagnosis']}", ""]
    for n in rep["notes"] + rep["signals_notes"]:
        L += [f"> ⚠️ {n}", ""]
    for c in rep["great_clips"]:
        judges = " · ".join(f"{PERSONA_SHORT.get(r['judge'], r['judge'])} {r['effective']:.1f}" for r in c["judges"])
        arb = c["arbiter"]
        jc = f", {c['jump_cuts']} jump cut(s)" if c["jump_cuts"] else ""
        L += [f"### #{c['rank']}: {c['start_ts']} → {c['end_ts']} ({c['duration']:.1f}s{jc}) · score {c['score']:.1f}",
              "",
              f"- **Hook:** {c['hook_line']}",
              f"- **Why it will go viral:** {c['why_viral']}",
              f"- **On-screen caption / title:** {c['caption_title']}",
              f"- **Posting hook text:** {c['posting_hook_text']}",
              f"- **Hashtags:** {' '.join(c['hashtags'])}",
              f"- **Panel (effective scores):** {judges}",
              f"- **Devil's advocate:** {c['devils_advocate'] or 'n/a'}",
              f"- **Arbiter:** survived (confidence {arb.get('survival_confidence')}): {arb.get('decisive_issue') or ''}",
              f"- **File:** {c['file'] or '(not exported)'}", ""]
    if not rep["great_clips"] and rep["near_misses"]:
        L += ["## Top near-misses (and exactly what killed them)", "",
              "| # | Window | Length | Killed at | Decisive reason |", "|---|---|---|---|---|"]
        for i, r in enumerate(rep["near_misses"], 1):
            L.append(f"| {i} | {r['start_ts']}–{r['end_ts']} | {r['duration']:.1f}s | "
                     f"{STAGE_LABEL.get(r['killed_at'], r['killed_at'])} | {_md(r['reasons'][0] if r['reasons'] else '')} |")
        L.append("")
    if explain_rejects and rep["rejected"]:
        L += ["## Every rejected candidate", "",
              "| id | Window | Length | Killed at | Reasons |", "|---|---|---|---|---|"]
        for r in rep["rejected"]:
            L.append(f"| {r['id']} | {r['start_ts']}–{r['end_ts']} | {r['duration']:.1f}s | "
                     f"{STAGE_LABEL.get(r['killed_at'], r['killed_at'])} | {_md('; '.join(r['reasons']))} |")
        L.append("")
    cost = rep["cost"]
    L += ["## Cost", "", f"Actual API spend this run: **{fmt_usd(cost['total_usd'])}** (cached calls are free).", ""]
    for purpose, s in cost["by_purpose"].items():
        L.append(f"- {purpose}: {s['calls']} call(s), {s['cached_calls']} cached, {fmt_usd(s['usd'])}")
    return "\n".join(L) + "\n"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def write_reports(a, result, cfg: Config, *, explain_rejects: bool = False, exported: dict | None = None) -> dict:
    out = out_dir_for(a, cfg)
    out.mkdir(parents=True, exist_ok=True)
    rep = build_report(a, result, cfg, exported)
    write_json(out / "report.json", rep)
    (out / "report.md").write_text(render_markdown(rep, explain_rejects), encoding="utf-8")
    return {"dir": out, "md": out / "report.md", "json": out / "report.json", "report": rep}


def print_report(a, result, cfg: Config, paths: dict, *, explain_rejects: bool = False,
                 exported: dict | None = None) -> None:
    rep = paths["report"]
    good = bool(rep["great_clips"])
    console.print()
    console.print(Panel(f"[bold]{rep['verdict']}[/bold]" + (f"\n{rep['diagnosis']}" if rep["diagnosis"] else ""),
                        title=f"clip-scanner · {a.meta.title[:60]}", border_style="green" if good else "red"))
    for n in rep["notes"]:
        console.print(f"[yellow]⚠ {n}[/yellow]")
    if good:
        t = Table(show_lines=True)
        for col in ("#", "Window", "Len", "Score", "Hook / caption / why"):
            t.add_column(col, overflow="fold")
        for c in rep["great_clips"]:
            t.add_row(str(c["rank"]), f"{c['start_ts']}–{c['end_ts']}", f"{c['duration']:.1f}s", f"{c['score']:.1f}",
                      f"[bold]{c['caption_title']}[/bold]\nHook: {c['hook_line']}\nWhy: {c['why_viral']}\n"
                      f"Post: {c['posting_hook_text']}" + (f"\n[dim]{c['file']}[/dim]" if c["file"] else ""))
        console.print(t)
    elif rep["near_misses"]:
        t = Table(title="Top near-misses", show_lines=True)
        for col in ("Window", "Len", "Killed at", "Reason"):
            t.add_column(col, overflow="fold")
        for r in rep["near_misses"]:
            t.add_row(f"{r['start_ts']}–{r['end_ts']}", f"{r['duration']:.1f}s",
                      STAGE_LABEL.get(r["killed_at"], r["killed_at"]), r["reasons"][0] if r["reasons"] else "")
        console.print(t)
    if explain_rejects and rep["rejected"]:
        t = Table(title="Every rejected candidate", show_lines=False)
        for col in ("id", "Window", "Len", "Killed at", "Reasons"):
            t.add_column(col, overflow="fold")
        for r in rep["rejected"]:
            t.add_row(r["id"], f"{r['start_ts']}–{r['end_ts']}", f"{r['duration']:.1f}s",
                      STAGE_LABEL.get(r["killed_at"], r["killed_at"]), "; ".join(r["reasons"]))
        console.print(t)
    console.print(f"Actual API cost this run: [bold]{fmt_usd(rep['cost']['total_usd'])}[/bold]")
    console.print(f"[dim]report: {paths['md']}\njson:   {paths['json']}[/dim]")
