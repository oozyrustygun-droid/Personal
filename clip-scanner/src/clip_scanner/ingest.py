"""Stage 1: get the video onto disk and describe it.

URLs are downloaded with yt-dlp (best stream at or below 1080p) into the per-video
workspace. Local files are referenced in place, without a copy, and identified by a
content fingerprint, so renaming a file still hits the cache.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse

from .cache import Workspace, fingerprint, read_json, write_json
from .config import Config
from .media import extract_audio, probe

INGEST_VERSION = 1


@dataclass
class VideoMeta:
    video_id: str
    source: str
    title: str
    media_path: str
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    uploader: str = ""
    url: str = ""

    @property
    def path(self) -> Path:
        return Path(self.media_path)

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:80] or "video"


def local_video_id(path: Path) -> str:
    """Content fingerprint: size + first and last 4 MB (fast even for multi-GB files)."""
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        if size > 8 << 20:
            f.seek(-(4 << 20), 2)
            h.update(f.read())
    return f"local-{_slug(path.stem)[:40]}-{h.hexdigest()[:12]}"


def _ydl_opts(cfg: Config, outtmpl: str | None = None) -> dict:
    opts: dict = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": cfg.ingest.format,
        "merge_output_format": "mp4",
    }
    if outtmpl:
        opts["outtmpl"] = outtmpl
    if cfg.ingest.cookies_from_browser:
        opts["cookiesfrombrowser"] = (cfg.ingest.cookies_from_browser,)
    return opts


def resolve_url(url: str, cfg: Config) -> tuple[str, dict]:
    from yt_dlp import YoutubeDL

    with YoutubeDL(_ydl_opts(cfg)) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise RuntimeError(f"yt-dlp could not resolve {url}")
    if info.get("_type") == "playlist":
        raise RuntimeError("playlists are not supported; pass a single video URL")
    extractor = (info.get("extractor_key") or info.get("extractor") or "web").lower()
    return f"{_slug(extractor)}-{_slug(str(info['id']))}", info


def download(url: str, ws: Workspace, cfg: Config) -> Path:
    from yt_dlp import YoutubeDL

    existing = sorted(ws.root.glob("source.*"))
    existing = [p for p in existing if p.suffix not in (".part", ".ytdl") and ".part" not in p.name]
    if existing:
        return existing[0]
    with YoutubeDL(_ydl_opts(cfg, str(ws.path("source.%(ext)s")))) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = Path(ydl.prepare_filename(info))
    if not filename.exists():
        matches = [p for p in ws.root.glob("source.*") if ".part" not in p.name]
        if not matches:
            raise RuntimeError("download finished but no source file was found")
        filename = matches[0]
    return filename


def ingest(source: str, cfg: Config, *, console=None) -> tuple[VideoMeta, Workspace]:
    """Resolve, download if needed, probe, and extract 16 kHz mono audio. Resumable."""
    info: dict = {}
    if is_url(source):
        video_id, info = resolve_url(source, cfg)
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no such file: {source}")
        video_id = local_video_id(path)

    ws = Workspace(cfg.cache_dir / video_id)
    if is_url(source):
        # YouTube's "Most replayed" graph: which parts viewers jump back to (absent on many videos).
        heat = info.get("heatmap")
        write_json(ws.path("heatmap.json"), {"segments": heat if isinstance(heat, list) else []})
    meta_path = ws.path("meta.json")
    fp = fingerprint("ingest", INGEST_VERSION, source if is_url(source) else video_id)

    if ws.is_done("ingest", fp, [meta_path, ws.path("audio.wav")]):
        meta = VideoMeta(**read_json(meta_path))
        if Path(meta.media_path).exists():
            return meta, ws

    if is_url(source):
        if console:
            console.print(f"[dim]Downloading {source} (≤1080p) …[/dim]")
        media = download(source, ws, cfg)
        title = info.get("title") or video_id
        uploader = info.get("uploader") or info.get("channel") or ""
        url = info.get("webpage_url") or source
    else:
        media = Path(source).expanduser().resolve()
        title, uploader, url = media.stem, "", ""

    pi = probe(media)
    if not pi.has_audio:
        raise RuntimeError("the video has no audio track; clip-scanner needs speech to judge moments")
    w, h = pi.display_size
    meta = VideoMeta(video_id=video_id, source=source, title=title, media_path=str(media), duration=pi.duration,
                     width=w, height=h, fps=pi.fps, has_audio=pi.has_audio, uploader=uploader, url=url)
    if console:
        console.print(f"[dim]Extracting audio ({pi.duration / 60:.1f} min) …[/dim]")
    extract_audio(media, ws.path("audio.wav"))
    write_json(meta_path, asdict(meta))
    ws.mark_done("ingest", fp)
    return meta, ws
