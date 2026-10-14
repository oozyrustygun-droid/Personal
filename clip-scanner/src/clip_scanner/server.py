"""`clip-scanner ui`: a local web front end.

Paste or drop YouTube/TikTok links, watch each scan move through the stages, approve the judging
spend, then review GREAT clips (play, copy captions, download) or the near-misses. Scans run one
at a time on this machine, with the same pipeline as the CLI.

Security: the server listens on 127.0.0.1 only, checks the Host header (against DNS rebinding), and
requires a per-launch token on every state-changing request, so no other website can start scans
that spend money.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import ui
from .cli_common import BACKEND_LABEL, JUDGE_BACKENDS, resolve_backend
from .config import load_config
from .ui import console

WEB_DIR = Path(__file__).parent / "web"
STAGES = ["Ingest", "Transcribe", "Signals", "Candidates", "Keyframes", "Judging", "Export"]
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class Job:
    def __init__(self, source: str, max_clips: int | None, cut: bool):
        self.id = uuid.uuid4().hex[:10]
        self.source = source
        self.max_clips = max_clips
        self.cut = cut
        self.status = "queued"          # queued | running | awaiting_approval | done | failed | cancelled
        self.title = ""
        self.stages = {s: {"state": "pending", "seconds": None, "note": ""} for s in STAGES}
        self.current = ""
        self.log: deque[str] = deque(maxlen=500)
        self.estimate: dict | None = None
        self.judging = {"total": 0, "eligible": 0, "done": 0, "errors": 0, "great": 0}
        self.out_dir: str | None = None
        self.result_status = ""
        self.notes: list[str] = []
        self.error = ""
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self._decision = threading.Event()
        self._approved = False

    def to_json(self) -> dict:
        return {"id": self.id, "source": self.source, "title": self.title, "status": self.status,
                "current": self.current, "stages": [{"name": k, **v} for k, v in self.stages.items()],
                "log": list(self.log)[-80:], "estimate": self.estimate, "judging": self.judging,
                "report_dir": Path(self.out_dir).name if self.out_dir else None,
                "result_status": self.result_status, "notes": self.notes, "error": self.error, "max_clips": self.max_clips,
                "cut": self.cut, "created": self.created, "started": self.started, "finished": self.finished}

    # -- events from the pipeline ------------------------------------------------------------------
    def on_event(self, kind: str, data: dict) -> None:
        if kind == "stage_start" and data["name"] in self.stages:
            self.current = data["name"]
            self.stages[data["name"]]["state"] = "active"
        elif kind == "stage_end" and data["name"] in self.stages:
            st = self.stages[data["name"]]
            st.update(state="cached" if data.get("cached") else "done", seconds=data.get("seconds"),
                      note=data.get("note", ""))
            if data["name"] == "Ingest" and data.get("note"):
                self.title = data["note"].split(" · ")[0]
        elif kind == "estimate":
            self.estimate = data
        elif kind == "judging":
            self.judging.update(total=data["total"], eligible=data["eligible"])
        elif kind == "judging_done":
            self.judging["great"] = data.get("great", self.judging["great"])  # after duplicates are dropped
        elif kind == "candidate":
            if data.get("stage") == "error":  # not judged (login, plan limit, a failed call)
                self.judging["errors"] += 1
            else:
                self.judging["done"] += 1
            self.judging["great"] += int(bool(data.get("great")))


class _LogWriter:
    """File-like sink for the rich console while a job runs: lines go into the job's log."""

    def __init__(self, job: Job):
        self.job = job
        self.buf = ""

    def write(self, text: str) -> int:
        self.buf += _ANSI.sub("", text)
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.job.log.append(line.rstrip())
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


class Scanner:
    """Runs queued jobs one at a time on a background thread."""

    def __init__(self, config_path: str | None, judge_backend: str | None = None, scripted_judge: str | None = None):
        self.config_path = config_path
        self.judge_backend = judge_backend
        self.scripted_judge = scripted_judge
        self.jobs: OrderedDict[str, Job] = OrderedDict()
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="clip-scanner-worker")
        self.thread.start()

    def add(self, sources: list[str], max_clips: int | None, cut: bool) -> list[Job]:
        new = []
        with self.lock:
            for s in sources:
                job = Job(s, max_clips, cut)
                self.jobs[job.id] = job
                new.append(job)
        self.wake.set()
        return new

    def decide(self, job_id: str, approve: bool) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status == "queued" and not approve:
            job.status = "cancelled"
            return True
        if job.status == "awaiting_approval":
            job._approved = approve
            job._decision.set()
            return True
        return False

    def snapshot(self) -> list[dict]:
        with self.lock:
            return [j.to_json() for j in reversed(self.jobs.values())]

    def _next(self) -> Job | None:
        with self.lock:
            return next((j for j in self.jobs.values() if j.status == "queued"), None)

    def _loop(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self.wake.wait(timeout=1.0)
                self.wake.clear()
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        from .judging.runner import needs_approval
        from .pipeline import run_scan

        job.status, job.started = "running", time.time()
        old_file = console.file
        console.file = _LogWriter(job)
        ui.listeners.append(job.on_event)

        def confirm(est: dict, cfg) -> bool:
            if not needs_approval(est, cfg):
                return True
            job.status = "awaiting_approval"
            job._decision.wait()
            job.status = "running" if job._approved else "cancelled"
            return job._approved

        args = argparse.Namespace(source=job.source, config=self.config_path, max_clips=job.max_clips,
                                  no_cut=not job.cut, explain_rejects=True, dry_run=False, rerun=[], yes=False,
                                  judge_backend=self.judge_backend, scripted_judge=self.scripted_judge)
        try:
            res = run_scan(args, confirm)
            job.out_dir = str(res.get("out_dir")) if res.get("out_dir") else None
            job.result_status = res.get("status", "")
            job.notes = list(res.get("notes") or [])
            if job.status != "cancelled":
                job.status = "done"
        except Exception as e:  # show the failure in the UI instead of killing the worker
            job.status = "failed"
            job.error = friendly_error(e)
            for line in traceback.format_exc().splitlines()[-12:]:
                job.log.append(line)
        finally:
            ui.listeners.remove(job.on_event)
            console.file = old_file
            job.current = ""
            job.finished = time.time()


def friendly_error(e: Exception) -> str:
    """One readable line for the UI; the full traceback goes to the job log."""
    text = str(e)
    low = text.lower()
    if type(e).__name__ == "DownloadError" or "yt_dlp" in type(e).__module__:
        if "private video" in low:
            return "Couldn't download: the video is private."
        if re.search(r"sign in to confirm|confirm your age|age[- ]restricted|login required|members[- ]only", low):
            return ("Couldn't download: YouTube wants a signed-in browser for this video. Set "
                    "[ingest] cookies_from_browser = \"chrome\" in config.toml.")
        if "403" in low or "forbidden" in low or "unable to download" in low or "timed out" in low:
            return ("Couldn't download the video (YouTube refused the connection). Check your internet "
                    "connection, then update the downloader: uv lock --upgrade-package yt-dlp && uv sync.")
        if "unsupported url" in low:
            return "That link isn't a video page yt-dlp understands. Use a single YouTube/TikTok video URL."
        return "Couldn't download the video. Details in the log."
    if isinstance(e, FileNotFoundError):
        return f"File not found: {e.filename or text}"
    if "no audio track" in low:
        return "The video has no audio track, so there's nothing to judge."
    first = text.strip().splitlines()[0] if text.strip() else type(e).__name__
    return f"{type(e).__name__}: {first[:300]}"


def list_reports(output_dir: Path) -> list[dict]:
    out = []
    for rep in output_dir.glob("*/report.json"):
        try:
            data = json.loads(rep.read_text())
        except (OSError, ValueError):
            continue
        out.append({"dir": rep.parent.name, "title": data.get("video", {}).get("title", rep.parent.name),
                    "verdict": data.get("verdict", ""), "status": data.get("status", ""),
                    "generated_at": data.get("generated_at") or data.get("started_at", ""),
                    "great": len(data.get("great_clips", []) or []), "mtime": rep.stat().st_mtime,
                    "scripted": data.get("judge_backend") == "scripted"})
    return sorted(out, key=lambda r: -r["mtime"])


def load_report(output_dir: Path, name: str) -> dict | None:
    d = (output_dir / name).resolve()
    if not d.is_relative_to(output_dir.resolve()) or not (d / "report.json").exists():
        return None
    rep = json.loads((d / "report.json").read_text())

    def url_for(path: str | None) -> str | None:
        if not path:
            return None
        p = Path(path)
        p = p if p.is_absolute() else (Path.cwd() / p)
        try:
            rel = p.resolve().relative_to(output_dir.resolve())
        except ValueError:
            return None
        return "/files/" + "/".join(rel.parts) if p.exists() else None

    for c in rep.get("great_clips", []) or []:
        c["file_url"] = url_for(c.get("file"))
        c["raw_url"] = url_for(c.get("raw_file"))
        c["poster_url"] = url_for(c.get("poster"))
    rep["dir"] = name
    rep["md_url"] = f"/files/{name}/report.md" if (d / "report.md").exists() else None
    return rep


def make_handler(scanner: Scanner, output_dir: Path, token: str, port: int):
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8").replace("__CLIP_SCANNER_TOKEN__", token)
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "clip-scanner"

        def log_message(self, *a) -> None:  # keep the terminal quiet
            pass

        # -- helpers -------------------------------------------------------------------------------
        def _json(self, data, status: int = 200) -> None:
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _token_ok(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Clip-Scanner-Token", ""), token)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return {}

        # -- routes --------------------------------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._json({"error": "bad host"}, HTTPStatus.FORBIDDEN)
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = index_html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return None
            if path == "/api/jobs":
                return self._json({"jobs": scanner.snapshot()})
            if path == "/api/reports":
                return self._json({"reports": list_reports(output_dir)})
            if path.startswith("/api/reports/"):
                rep = load_report(output_dir, unquote(path[len("/api/reports/"):]))
                return self._json(rep) if rep else self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            if path.startswith("/files/"):
                return self._file(unquote(path[len("/files/"):]))
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok() or not self._token_ok():
                return self._json({"error": "forbidden"}, HTTPStatus.FORBIDDEN)
            path = urlparse(self.path).path
            body = self._body()
            if path == "/api/jobs":
                sources = parse_sources(body.get("text", ""))
                if not sources:
                    return self._json({"error": "no YouTube/TikTok links or video files found"},
                                      HTTPStatus.BAD_REQUEST)
                max_clips = body.get("max_clips")
                max_clips = int(max_clips) if isinstance(max_clips, (int, float, str)) and str(max_clips).isdigit() \
                    and int(max_clips) >= 1 else None
                jobs = scanner.add(sources, max_clips, bool(body.get("cut", True)))
                return self._json({"jobs": [j.to_json() for j in jobs]})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/(approve|cancel)", path)
            if m:
                ok = scanner.decide(m.group(1), m.group(2) == "approve")
                return self._json({"ok": ok}, 200 if ok else HTTPStatus.CONFLICT)
            return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _file(self, rel: str) -> None:
            target = (output_dir / rel).resolve()
            if not target.is_relative_to(output_dir.resolve()) or not target.is_file():
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            size = target.stat().st_size
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if target.suffix == ".md":
                ctype = "text/markdown; charset=utf-8"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            status = 200
            if rng:  # browsers need byte ranges to seek inside a video
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start = max(0, size - int(m.group(2)))
                    end = min(end, size - 1)
                    if start > end:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return None
                    status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if self.headers.get("X-Download") or "download=1" in (urlparse(self.path).query or ""):
                self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
            self.end_headers()
            with open(target, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return None
                    remaining -= len(chunk)
            return None

    return Handler


_URL = re.compile(r"https?://[^\s<>\"']+")


def parse_sources(text: str) -> list[str]:
    """Links from pasted/dropped text (any mix of lines, spaces, commas), plus existing local file paths.
    De-duplicated, in order."""
    found: list[str] = []
    for m in _URL.finditer(text or ""):
        url = m.group(0).rstrip(".,);]")
        if url not in found:
            found.append(url)
    for line in (text or "").splitlines():
        line = line.strip().strip('"').strip("'")
        if line and not line.startswith("http") and Path(line).expanduser().is_file() and line not in found:
            found.append(str(Path(line).expanduser()))
    return found


def serve(config_path: str | None = None, port: int = 8765, open_browser: bool = True,
          judge_backend: str | None = None, scripted_judge: str | None = None) -> tuple[ThreadingHTTPServer, str]:
    cfg = load_config(config_path)
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scanner = Scanner(config_path, judge_backend, scripted_judge)
    token = secrets.token_urlsafe(24)
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", p), make_handler(scanner, output_dir, token, p))
            break
        except OSError:
            continue
    if httpd is None:
        raise OSError(f"no free port in {port}-{port + 19}")
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    httpd.scanner = scanner  # type: ignore[attr-defined]
    httpd.token = token  # type: ignore[attr-defined]
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    return httpd, url


def run_ui(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="clip-scanner ui", description="Local web front end for clip-scanner.")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--judge-backend", choices=JUDGE_BACKENDS, default=None, metavar="{claude-code,api}",
                   help="claude-code = your Claude subscription (default, from config.toml); api = ANTHROPIC_API_KEY")
    p.add_argument("--scripted-judge", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    httpd, url = serve(args.config, args.port, not args.no_browser, args.judge_backend, args.scripted_judge)
    console.print(f"[bold green]clip-scanner UI running at {url}[/bold green]  (Ctrl-C to stop)")
    backend = resolve_backend(load_config(args.config), args.judge_backend)
    console.print(f"Judging runs on {BACKEND_LABEL.get(backend, backend)}.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        # The scan runs on a background thread whose worker pool would otherwise keep judging on the way
        # out. Everything finished is already cached on disk, so stop the model calls and exit now.
        from .claude_code import stop_all_runs

        stop_all_runs()
        console.print("\nStopped. Finished scans are cached; re-add a link to resume an interrupted one.")
        sys.stdout.flush()
        os._exit(130)
    return 0
