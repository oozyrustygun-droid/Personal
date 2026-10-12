"""Export GREAT clips as 1080x1920 MP4s: face-tracked crop, pre-roll, jump cuts, burned-in captions."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .cache import fingerprint, read_json, write_json
from .captions import TimedWord, build_ass
from .config import Config
from .facetrack import shared_crop_path
from .media import hwaccel_args, probe, require_binary
from .ui import console


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def timed_words(cand, transcript, segs: list[tuple[float, float]]) -> list[TimedWord]:
    out, offset = [], 0.0
    for s, e in segs:
        for w in transcript.words:
            if s - 1e-3 <= w.start and w.end <= e + 1e-3:
                out.append(TimedWord(offset + w.start - s, offset + w.end - s, w.text))
        offset += e - s
    return out


def crop_geometry(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int, int]:
    """(crop_w, crop_h, y) of the largest out_w:out_h window that fits the source."""
    target = out_w / out_h
    if src_w / src_h > target:  # wider than 9:16: crop horizontally, track x
        return _even(src_h * target), src_h - src_h % 2, 0
    crop_h = _even(src_w / target)
    return src_w - src_w % 2, crop_h, max(0, (src_h - crop_h) // 2)


def render_clip(a, cand, out_path: Path, cfg: Config) -> Path:
    require_binary("ffmpeg")
    src = a.meta.path
    info = probe(src)
    sw, sh = info.display_size
    ex = cfg.export
    # The candidate's segments already include the (clamped) pre-roll: export exactly what was judged.
    segs = list(cand.segments)
    first_word = min((w.start for w in a.transcript.words if segs[0][0] <= w.start <= segs[0][1]), default=segs[0][0])
    pre = max(0.0, first_word - segs[0][0])
    total = sum(e - s for s, e in segs)
    crop_w, crop_h, crop_y = crop_geometry(sw, sh, ex.width, ex.height)
    horizontal = crop_w < sw
    center = (sw - crop_w) // 2
    path = shared_crop_path(a.ws.root, src, segs, sw, sh, crop_w, cfg) if horizontal else []
    if horizontal and not path:
        path = [(0.0, center)]  # same fallback as the judges' keyframes: a centred crop
    x0 = path[0][1] if path else 0

    with tempfile.TemporaryDirectory(prefix="clipscan-") as td:
        tdp = Path(td)
        cmds = "\n".join(f"{t:.3f} crop@c x {x};" for t, x in path) + "\n"
        (tdp / "crop.cmd").write_text(cmds)
        if cfg.captions.enabled:
            ass = build_ass(timed_words(cand, a.transcript, segs), cfg.captions, ex.width, ex.height, total)
            (tdp / "captions.ass").write_text(ass, encoding="utf-8")
        inputs: list[str] = []
        chains: list[str] = []
        labels = ""
        for k, (s, e) in enumerate(segs):
            d = e - s
            inputs += [*hwaccel_args(cfg.signals.hwaccel), "-ss", f"{s:.3f}", "-t", f"{d:.3f}", "-i", str(src)]
            fade = min(0.012, d / 4)
            chains.append(f"[{k}:v]setpts=PTS-STARTPTS[v{k}]")
            chains.append(f"[{k}:a]asetpts=PTS-STARTPTS,afade=t=in:st=0:d={fade:.3f},"
                          f"afade=t=out:st={max(d - fade, 0):.3f}:d={fade:.3f}[a{k}]")
            labels += f"[v{k}][a{k}]"
        vf = [f"sendcmd=f=crop.cmd,crop@c=w={crop_w}:h={crop_h}:x={x0}:y={crop_y}" if horizontal
              else f"crop=w={crop_w}:h={crop_h}:x=0:y={crop_y}",
              f"scale={ex.width}:{ex.height}:flags=lanczos", "setsar=1", "format=yuv420p"]
        if cfg.captions.enabled:
            vf.append("ass=captions.ass")
        fade_in = min(0.12, pre) if pre > 0 else 0.05
        graph = ";".join(chains + [
            f"{labels}concat=n={len(segs)}:v=1:a=1[vc][ac]",
            f"[vc]{','.join(vf)}[vout]",
            f"[ac]afade=t=in:st=0:d={fade_in:.3f},aresample=48000[aout]",
        ])
        tmp_out = tdp / "out.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[vout]", "-map", "[aout]",
               "-c:v", "libx264", "-preset", ex.preset, "-crf", str(ex.crf), "-profile:v", "high",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", ex.audio_bitrate, "-movflags", "+faststart",
               str(tmp_out)]
        p = subprocess.run(cmd, cwd=str(tdp), capture_output=True)  # relative paths keep the filter graph unescaped
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg export failed:\n{p.stderr.decode(errors='replace')[-2000:]}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_out.replace(out_path)
    return out_path


def slug(text: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "clip"


def clip_path(out_dir: Path, rank: int, cand_id: str, packaging: dict, raw: bool = False) -> Path:
    title = packaging.get(cand_id, {}).get("caption_title", "") or cand_id
    return out_dir / "clips" / f"{rank:02d}-{slug(title)}-{cand_id}{'-raw' if raw else ''}.mp4"


def render_fingerprint(a, cand, cfg: Config, kind: str = "vertical") -> str:
    return fingerprint("render", kind, a.meta.video_id, [list(s) for s in cand.segments],
                       cfg.section_hash("export", "captions"))


def raw_window(a, cand, cfg: Config) -> tuple[float, float]:
    """Contiguous landscape source span with editing handles on both sides (jump cuts not applied)."""
    h = cfg.export.raw_handles_s
    return max(0.0, cand.start - h), min(a.meta.duration, cand.end + h)


def render_raw(a, cand, out_path: Path, cfg: Config) -> Path:
    """Untouched original framing and audio, no captions: footage for re-editing by hand."""
    s, e = raw_window(a, cand, cfg)
    tmp = out_path.with_suffix(".tmp.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", *hwaccel_args(cfg.signals.hwaccel), "-ss", f"{s:.3f}",
                        "-t", f"{e - s:.3f}", "-i", str(a.meta.path), "-map", "0:v:0", "-map", "0:a:0",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(cfg.export.crf), "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", cfg.export.audio_bitrate, "-movflags", "+faststart", str(tmp)],
                       capture_output=True)
    if p.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg raw export failed:\n{p.stderr.decode(errors='replace')[-2000:]}")
    tmp.replace(out_path)
    return out_path


def expected_files(out_dir: Path, a, result, cfg: Config) -> dict[str, str]:
    """{path: render fingerprint} of every export the CURRENT verdict justifies."""
    by_id = {c.id: c for c in a.candidates}
    keep: dict[str, str] = {}
    for rank, d in enumerate(result.great, 1):
        c = by_id[d.candidate_id]
        keep[str(clip_path(out_dir, rank, c.id, result.packaging))] = render_fingerprint(a, c, cfg)
        if cfg.export.raw_copy:
            keep[str(clip_path(out_dir, rank, c.id, result.packaging, raw=True))] = render_fingerprint(a, c, cfg, "raw")
    return keep


def _sidecar_fp(path: Path) -> str | None:
    side = path.with_suffix(".json")
    try:
        return read_json(side).get("fingerprint") if side.exists() else None
    except (OSError, ValueError):
        return None


def prune_stale_clips(out_dir: Path, keep: dict[str, str]) -> list[Path]:
    """Delete every export that the latest verdict does not justify: clips no longer GREAT, and files
    whose render fingerprint doesn't match the window that was judged (e.g. after boundaries changed)."""
    removed = []
    for p in (out_dir / "clips").glob("*.mp4"):
        if str(p) not in keep or _sidecar_fp(p) != keep[str(p)]:
            p.unlink(missing_ok=True)
            p.with_suffix(".json").unlink(missing_ok=True)
            removed.append(p)
    return removed


STASH = ".previous-run"
REPORTS = ("report.json", "report.md")


def _is_in_progress(report_json: Path) -> bool:
    try:
        return read_json(report_json).get("status") == "in_progress"
    except (OSError, ValueError):
        return True


def stash_exports(out_dir: Path) -> None:
    """Before a scan, move the previous report and exports out of sight. If this run is interrupted, the
    folder must not keep offering a verdict or clips that a tightened bar might now reject."""
    stash = out_dir / STASH
    (stash / "clips").mkdir(parents=True, exist_ok=True)
    clips = out_dir / "clips"
    if clips.exists():
        for p in list(clips.glob("*.mp4")) + list(clips.glob("*.json")):
            p.replace(stash / "clips" / p.name)
    if (out_dir / "report.json").exists() and not _is_in_progress(out_dir / "report.json"):
        for name in REPORTS:  # keep the last COMPLETED report; never stash an in-progress marker over it
            if (out_dir / name).exists():
                (out_dir / name).replace(stash / name)


def restore_exports(out_dir: Path, keep: dict[str, str]) -> None:
    """After judging, bring back stashed renders that are still justified (saves re-encoding); drop the rest."""
    stash = out_dir / STASH
    if not stash.exists():
        return
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    for p in (stash / "clips").glob("*.mp4"):
        target = out_dir / "clips" / p.name
        if str(target) in keep and _sidecar_fp(p) == keep[str(target)]:
            p.with_suffix(".json").replace(target.with_suffix(".json"))
            p.replace(target)
    shutil.rmtree(stash, ignore_errors=True)


def restore_all(out_dir: Path) -> bool:
    """Judging never ran (cancelled at the cost prompt, no credentials, --dry-run): put the previous
    report and clips back exactly as they were. Returns True if there was something to restore."""
    stash = out_dir / STASH
    if not stash.exists():
        return False
    restored = False
    for name in REPORTS:
        if (stash / name).exists():
            (stash / name).replace(out_dir / name)
            restored = True
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)
    for p in (stash / "clips").iterdir():
        p.replace(out_dir / "clips" / p.name)
        restored = True
    shutil.rmtree(stash, ignore_errors=True)
    return restored


def _render_cached(path: Path, fp: str, render, meta: dict, force: bool) -> bool:
    sidecar = path.with_suffix(".json")
    if path.exists() and not force and sidecar.exists() and read_json(sidecar).get("fingerprint") == fp:
        return True
    render()
    write_json(sidecar, {"fingerprint": fp, **meta})
    console.print(f"  [green]✓[/green] {path}")
    return True


def export_clips(a, result, cfg: Config, out_dir: Path, force: bool = False) -> dict[str, dict]:
    """Render each GREAT clip. Returns {candidate_id: {"vertical", "raw", "raw_in", "raw_out"}}."""
    by_id = {c.id: c for c in a.candidates}
    exported: dict[str, dict] = {}
    for rank, d in enumerate(result.great, 1):
        c = by_id[d.candidate_id]
        info: dict = {"vertical": None, "raw": None}
        meta = {"segments": [list(s) for s in c.segments], "source": a.meta.source, "candidate": c.id}
        path = clip_path(out_dir, rank, c.id, result.packaging)
        try:
            _render_cached(path, render_fingerprint(a, c, cfg), lambda: render_clip(a, c, path, cfg), meta, force)
            info["vertical"] = str(path)
        except Exception as e:  # one failed render shouldn't lose the report
            console.print(f"  [red]vertical export failed for {c.id}: {e}[/red]")
        if cfg.export.raw_copy:
            raw = clip_path(out_dir, rank, c.id, result.packaging, raw=True)
            s, e = raw_window(a, c, cfg)
            try:
                _render_cached(raw, render_fingerprint(a, c, cfg, "raw"), lambda: render_raw(a, c, raw, cfg),
                               {**meta, "raw_window": [s, e]}, force)
                info.update(raw=str(raw), raw_in=round(c.start - s, 2), raw_out=round(c.end - s, 2))
            except Exception as ex:
                console.print(f"  [red]raw export failed for {c.id}: {ex}[/red]")
        exported[c.id] = info
    return exported
