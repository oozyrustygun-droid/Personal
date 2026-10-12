"""`clip-scanner calibrate`: measure the judges on known hits and flops, then tune the bar.

Every calibration clip is judged as ONE candidate spanning the whole file, with the FULL panel
(no short-circuit) plus the devil's advocate and arbiter. Raw verdicts are cached, so the
threshold sweep afterwards is free. It re-applies the consensus rule under many candidate
thresholds, keeps the combinations that reject every known flop, and among those with the best
recall recommends the smallest change from the current bar (ties go to the stricter option,
because a false positive costs more than a false negative).
"""

from __future__ import annotations

import argparse
import itertools
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cache import fingerprint, write_json
from .candidates import Candidate
from .cli_common import resolve_backend
from .config import REPO_ROOT, Config, load_config
from .judging.gates import GateResult
from .judging.panel import Decision, JudgeOutcome, decide
from .ui import console, fmt_usd, stage

MIN_PER_LABEL_TO_WRITE = 10  # --write-config needs at least this many hits AND flops

SWEEP = {
    "overall_min": [7.0, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5],
    "rubric_mean_min": [6.5, 7.0, 7.5, 8.0, 8.5],
    "min_survival_confidence": [5.0, 6.0, 7.0, 8.0],
}


@dataclass
class Entry:
    id: str
    label: str            # hit | flop | ambiguous
    views: int | None
    file: Path | None
    url: str
    notes: str = ""


def whole_clip_candidate(transcript, signals, duration: float, cfg: Config, signals_fp: str):
    """A calibration clip is already a finished short: judge it exactly as posted."""
    peak_i = int(np.argmax(signals.series["interest"])) if signals.series["interest"].size else 0
    c = Candidate(id="c01", segments=[(0.0, round(duration, 3))], peak=round(peak_i * signals.hop, 2),
                  origin="calibration", score=0.0, reason="whole calibration clip")
    return [c], fingerprint("calibration-candidate", signals_fp, duration)


def load_manifest(path: Path, cfg: Config) -> list[Entry]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    base = path.parent / "clips"
    out = []
    for i, d in enumerate(data.get("clip", []), 1):
        views = d.get("views")
        label = (d.get("label") or "").lower()
        if label not in ("hit", "flop"):
            if views is None:
                label = "ambiguous"
            elif views >= cfg.calibration.hit_min_views:
                label = "hit"
            elif views <= cfg.calibration.flop_max_views:
                label = "flop"
            else:
                label = "ambiguous"
        f = base / d["file"] if d.get("file") else None
        out.append(Entry(id=d.get("id") or f"clip{i:02d}", label=label, views=views, file=f,
                         url=d.get("url", ""), notes=d.get("notes", "")))
    return out


def fetch(entry: Entry, cfg: Config) -> Path | None:
    from yt_dlp import YoutubeDL

    target = entry.file or (Path("calibration/clips") / f"{entry.label}s" / f"{entry.id}.mp4")
    target.parent.mkdir(parents=True, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "format": cfg.ingest.format,
            "merge_output_format": "mp4", "outtmpl": str(target.with_suffix(".%(ext)s"))}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(entry.url, download=True)
        got = Path(ydl.prepare_filename(info))
    final = got if got.exists() else next(target.parent.glob(f"{target.stem}.*"), None)
    return final


def redecide(d: Decision, cfg: Config) -> Decision:
    """Re-apply the consensus rule to cached raw verdicts under a different config."""
    det = [GateResult(**g) for g in d.deterministic]
    outcomes = [JudgeOutcome(**{k: v for k, v in j.items()}) for j in d.judges]
    return decide(d.candidate_id, det, outcomes or None, d.arbiter, d.devil, cfg)


def variant(cfg: Config, overall: float, rubric: float, survival: float) -> Config:
    data = cfg.model_dump(exclude={"source_path"})
    data["judging"]["overall_min"] = overall
    data["judging"]["rubric_mean_min"] = rubric
    data["devils_advocate"]["min_survival_confidence"] = survival
    return Config.model_validate(data)


def metrics(rows: list[tuple[str, bool]]) -> dict:
    tp = sum(1 for lab, g in rows if lab == "hit" and g)
    fp = sum(1 for lab, g in rows if lab == "flop" and g)
    fn = sum(1 for lab, g in rows if lab == "hit" and not g)
    tn = sum(1 for lab, g in rows if lab == "flop" and not g)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "flop_pass_rate": fp / (fp + tn) if fp + tn else None}


def sweep(labeled: list[tuple[str, Decision]], cfg: Config) -> dict:
    results = []
    for o, r, s in itertools.product(SWEEP["overall_min"], SWEEP["rubric_mean_min"], SWEEP["min_survival_confidence"]):
        v = variant(cfg, o, r, s)
        rows = [(lab, redecide(d, v).great) for lab, d in labeled]
        results.append({"overall_min": o, "rubric_mean_min": r, "min_survival_confidence": s, **metrics(rows)})
    safe = [x for x in results if x["fp"] == 0]
    if not safe:
        return {"grid": results, "recommended": None,
                "note": "No threshold combination rejects every known flop. The judges cannot separate your "
                        "flops from your hits; add more examples or revise the style guide."}
    best_tp = max(x["tp"] for x in safe)
    tied = [x for x in safe if x["tp"] == best_tp]
    # Smallest move from the current bar that rejects every known flop and keeps the most hits.
    # Small calibration sets overfit easily, so don't move thresholds the data doesn't demand.
    # Ties go to the stricter option.
    cur = (cfg.judging.overall_min, cfg.judging.rubric_mean_min, cfg.devils_advocate.min_survival_confidence)

    def distance(x):
        return (abs(x["overall_min"] - cur[0]) / 0.25 + abs(x["rubric_mean_min"] - cur[1]) / 0.5
                + abs(x["min_survival_confidence"] - cur[2]) / 1.0)

    rec = min(tied, key=lambda x: (distance(x), -x["overall_min"], -x["rubric_mean_min"],
                                   -x["min_survival_confidence"]))
    return {"grid": results, "recommended": rec, "note": ""}


def loosened(rec: dict, cfg: Config) -> list[str]:
    """Which recommended thresholds are LOOSER than the current bar."""
    out = []
    if rec["overall_min"] < cfg.judging.overall_min:
        out.append(f"overall_min {cfg.judging.overall_min} -> {rec['overall_min']}")
    if rec["rubric_mean_min"] < cfg.judging.rubric_mean_min:
        out.append(f"rubric_mean_min {cfg.judging.rubric_mean_min} -> {rec['rubric_mean_min']}")
    if rec["min_survival_confidence"] < cfg.devils_advocate.min_survival_confidence:
        out.append(f"min_survival_confidence {cfg.devils_advocate.min_survival_confidence} -> "
                   f"{rec['min_survival_confidence']}")
    return out


def fully_judged(d: Decision, cfg: Config) -> bool:
    """Every call for this clip produced a usable answer from the configured model. A clip with a missing
    judge, devil or arbiter is rejected under every threshold, so it would silently stop constraining
    the sweep (an unjudged flop can never 'pass')."""
    if d.stage == "deterministic_gates":
        return True  # a hard-gate kill is the same at every threshold
    if d.stage == "error" or d.errors or any(a.get("unavailable") for a in d.assessments):
        return False
    return not cfg.devils_advocate.enabled or d.arbiter is not None


def write_guard(rec: dict | None, cfg: Config, hits: int, flops: int, scripted: bool, allow_loosen: bool,
                unjudged: int = 0, stopped: str = "") -> str | None:
    """Reason to refuse --write-config, or None if writing is safe.

    unjudged: labelled clips that ended in an error. An unjudged flop never passes under any threshold,
    so it would silently stop constraining the sweep; a bar is only written from a complete run."""
    if rec is None:
        return "no safe recommendation to write"
    if scripted:
        return "verdicts came from the scripted test judge, not Claude"
    if stopped:
        return f"judging stopped early ({stopped}); re-run calibrate to finish (finished verdicts are cached)"
    if unjudged:
        return (f"{unjudged} labelled clip(s) were not fully judged; re-run calibrate to finish (finished verdicts "
                "are cached). If the same clips fail again, add --rerun-judging to ask again instead of replaying "
                "a cached refusal, or leave them out with --only")
    if hits < MIN_PER_LABEL_TO_WRITE or flops < MIN_PER_LABEL_TO_WRITE:
        return (f"need at least {MIN_PER_LABEL_TO_WRITE} hits AND {MIN_PER_LABEL_TO_WRITE} flops that the panel "
                f"actually scored (have {hits} hits, {flops} flops; clips killed by a hard gate or an error "
                f"don't count); a bar tuned on less overfits")
    loose = loosened(rec, cfg)
    if loose and not allow_loosen:
        return ("the recommendation LOOSENS the bar (" + "; ".join(loose) + "). Re-run with --allow-loosen "
                "if you have checked the per-clip results and really want that")
    return None


def write_thresholds(path: Path, rec: dict) -> None:
    import tomlkit

    doc = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    doc.setdefault("judging", tomlkit.table())
    doc.setdefault("devils_advocate", tomlkit.table())
    doc["judging"]["overall_min"] = rec["overall_min"]
    doc["judging"]["rubric_mean_min"] = rec["rubric_mean_min"]
    doc["devils_advocate"]["min_survival_confidence"] = rec["min_survival_confidence"]
    path.write_text(tomlkit.dumps(doc), encoding="utf-8")


def run_calibrate(args: argparse.Namespace) -> int:
    from .judging.runner import build_context, judge_candidate
    from .judging.gates import clip_measurements, deterministic_gates
    from .pipeline import analyze

    cfg = load_config(args.config)
    manifest = Path(args.manifest or cfg.calibration.manifest)
    if not manifest.is_absolute() and not manifest.exists():
        manifest = REPO_ROOT / manifest
    if not manifest.exists():
        console.print(f"[red]manifest not found: {manifest}[/red]")
        return 2
    entries = load_manifest(manifest, cfg)
    if args.only:
        entries = [e for e in entries if e.id in set(args.only)]
    ready: list[tuple[Entry, Path]] = []
    for e in entries:
        path = e.file if e.file and e.file.exists() else None
        if path is None and e.url and args.fetch:
            try:
                path = fetch(e, cfg)
            except Exception as ex:  # one bad URL shouldn't stop the run
                console.print(f"[yellow]{e.id}: download failed ({ex})[/yellow]")
        if path is None:
            console.print(f"[yellow]{e.id}: no local file{' (pass --fetch to download it)' if e.url else ''}; skipped[/yellow]")
            continue
        ready.append((e, path))
    hits = sum(1 for e, _ in ready if e.label == "hit")
    flops = sum(1 for e, _ in ready if e.label == "flop")
    console.print(f"[bold]Calibration set:[/bold] {hits} hits, {flops} flops, "
                  f"{len(ready) - hits - flops} ambiguous (excluded from metrics)")
    if hits < 5 or flops < 5:
        console.print("[yellow]Fewer than 5 hits or 5 flops: the threshold suggestion will be noisy. "
                      "Add more labelled examples to calibration/manifest.toml.[/yellow]")
    if not ready:
        return 2

    analyses = []
    for e, path in ready:
        console.rule(f"{e.id} ({e.label}{f', {e.views:,} views' if e.views else ''})")
        analyses.append((e, analyze(str(path), cfg, judge_backend=args.judge_backend,
                                    scripted_judge=args.scripted_judge, single_window=True, quiet=True,
                                    rerun={"judging"} if getattr(args, "rerun_judging", False) else frozenset())))

    calls_per_clip = len(cfg.judging.judges) + (2 if cfg.devils_advocate.enabled else 0)
    price = cfg.price(cfg.judging.model)
    est = len(analyses) * calls_per_clip * (8000 * price.input + 4000 * price.output) / 1e6
    backend = resolve_backend(cfg, args.judge_backend)
    subscription = backend == "claude-code"
    if subscription:
        console.print(f"Judging {len(analyses)} clips with the full panel: up to {len(analyses) * calls_per_clip} "
                      f"calls on your Claude plan through Claude Code (no API charges; cached clips are free).")
    else:
        console.print(f"Judging {len(analyses)} clips with the full panel: up to {len(analyses) * calls_per_clip} "
                      f"calls, estimated {fmt_usd(est)} (cached clips are free).")
    if not subscription and est > cfg.judging.confirm_above_usd and not args.yes:
        if console.input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
            return 1

    labeled: list[tuple[str, Decision]] = []
    table = []
    total_cost = 0.0
    unjudged_ids: list[str] = []
    with stage("Judging calibration clips"):
        stop = None
        for e, a in analyses:
            # One stop (login, plan limit, extra usage) covers every clip: later clips still resolve
            # from the cache but start no new requests.
            stop = stop or next((x.llm.fatal for _, x in analyses if x.llm.fatal is not None), None)
            if stop is not None and a.llm.fatal is None:
                a.llm.fatal = stop
            ctx = build_context(cfg, a.llm, a.transcript, a.signals, a.frames, a.meta.title, a.meta.uploader,
                                a.meta.duration)
            c = a.candidates[0]
            det = deterministic_gates(c, a.transcript, a.signals, cfg)
            d = judge_candidate(c, det, clip_measurements(c, a.transcript, a.signals, cfg.gates.music_prob), ctx, full_panel=True)
            total_cost += a.tracker.total()
            if e.label in ("hit", "flop"):
                labeled.append((e.label, d))
                if not fully_judged(d, cfg):
                    unjudged_ids.append(e.id)
            correct = (e.label == "hit") == d.great if e.label in ("hit", "flop") else None
            table.append({"id": e.id, "label": e.label, "views": e.views, "great": d.great, "stage": d.stage,
                          "min_effective": d.min_effective, "correct": correct,
                          "reason": d.reasons[0] if d.reasons else "", "decision": d.to_json()})
            mark = "" if correct is None else ("[green]✓[/green]" if correct else "[red]✗[/red]")
            console.print(f"  {mark} {e.id:<24} {e.label:<9} → {'GREAT' if d.great else 'not great':<9} "
                          f"[dim]{d.min_effective if d.min_effective is not None else '–'} "
                          f"{d.reasons[0] if d.reasons else ''}[/dim]")

    # Only clips the panel actually scored constrain the thresholds being swept. A flop killed by a hard
    # gate or an error says nothing about overall_min or survival confidence.
    informative = {"hit": 0, "flop": 0}
    for lab, d in labeled:
        if d.stage in ("panel", "devils_advocate", "great") and fully_judged(d, cfg):
            informative[lab] += 1
    current = metrics([(lab, d.great) for lab, d in labeled])
    sw = sweep(labeled, cfg)
    hit_scores = [r["min_effective"] for r in table if r["label"] == "hit" and r["min_effective"] is not None]
    flop_scores = [r["min_effective"] for r in table if r["label"] == "flop" and r["min_effective"] is not None]
    console.print()
    def pct(x):
        return "n/a" if x is None else f"{x:.0%}"

    console.print(f"[bold]At the current bar[/bold] (overall ≥ {cfg.judging.overall_min}, rubric ≥ "
                  f"{cfg.judging.rubric_mean_min}, survival ≥ {cfg.devils_advocate.min_survival_confidence}): "
                  f"precision {pct(current['precision'])} · recall {pct(current['recall'])} · "
                  f"flops passed {current['fp']}/{current['fp'] + current['tn']}")
    if hit_scores and flop_scores:
        console.print(f"Score separation: lowest hit {min(hit_scores):.2f} vs highest flop {max(flop_scores):.2f}")
    rec = sw["recommended"]
    if rec:
        console.print(f"[bold]Recommended bar[/bold]: overall_min={rec['overall_min']}, "
                      f"rubric_mean_min={rec['rubric_mean_min']}, "
                      f"min_survival_confidence={rec['min_survival_confidence']} → recall {pct(rec['recall'])} "
                      f"with 0 known flops passing")
    else:
        console.print(f"[red]{sw['note']}[/red]")

    out_dir = manifest.parent / "results" / time.strftime("%Y%m%d-%H%M%S")
    stopped = sorted({str(a.llm.fatal).split(": ", 1)[-1] for _, a in analyses if a.llm.fatal is not None})
    for msg in stopped:
        console.print(f"[yellow]Judging stopped early: {msg}[/yellow]")
    unjudged = len(unjudged_ids)
    if unjudged:
        console.print(f"[yellow]{unjudged} labelled clip(s) were not fully judged ({', '.join(unjudged_ids)}); the "
                      f"precision figures above treat them as rejected. Re-run to finish.[/yellow]")
    write_json(out_dir / "calibration.json", {"judge_backend": backend, "informative": informative,
                                              "current": current, "recommended": rec, "note": sw["note"],
                                              "clips": table, "grid": sw["grid"], "cost_usd": round(total_cost, 4)})
    spent = "no API charges (Claude plan)" if subscription else f"API cost this run {fmt_usd(total_cost)}"
    console.print(f"[dim]details: {out_dir / 'calibration.json'} · {spent}[/dim]")
    if rec:
        loose = loosened(rec, cfg)
        if loose:
            console.print(f"[yellow]Note: this would LOOSEN the bar: {'; '.join(loose)}[/yellow]")
    if args.write_config:
        refusal = write_guard(rec, cfg, informative["hit"], informative["flop"], backend == "scripted",
                              args.allow_loosen, unjudged=unjudged, stopped="; ".join(stopped))
        if refusal:
            console.print(f"[red]Not writing config.toml: {refusal}.[/red]")
            return 1
        path = Path(cfg.source_path) if cfg.source_path else Path("config.toml")
        write_thresholds(path, rec)
        console.print(f"[green]wrote recommended thresholds to {path}[/green]")
    return 0
