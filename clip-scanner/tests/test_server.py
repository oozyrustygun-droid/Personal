"""The local web UI: a real scan driven through the HTTP API (scripted judge, sample video)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from clip_scanner.server import parse_sources, serve
from conftest import pick_asr

pytestmark = pytest.mark.e2e
FIX = Path(__file__).parent / "fixtures" / "sample.mp4"


def req(base, path, method="GET", body=None, token=None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["X-Clip-Scanner-Token"] = token
    r = urllib.request.Request(base + path, method=method, headers=h,
                               data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    backend, model = pick_asr()
    cfg = tmp / "config.toml"
    cfg.write_text(f'[general]\ncache_dir = "{tmp / "cache"}"\noutput_dir = "{tmp / "out"}"\n'
                   f'[transcribe]\nbackend = "{backend}"\nmodel = "{model}"\n'
                   '[judging]\nconfirm_above_usd = 0.0\n')  # force the approval step
    rules = tmp / "rules.json"
    rules.write_text(json.dumps({"great_phrases": ["oh my god", "never eating"], "great_score": 9.2,
                                 "default_score": 6.2, "hook_fail_openers": ["so anyway"]}))
    httpd, url = serve(str(cfg), port=18765, open_browser=False, judge_backend="scripted",
                       scripted_judge=str(rules))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"base": url.rstrip("/"), "token": httpd.token, "tmp": tmp}
    httpd.shutdown()


def wait_for(base, pred, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        jobs = json.loads(req(base, "/api/jobs")[2])["jobs"]
        if jobs and pred(jobs[0]):
            return jobs[0]
        time.sleep(0.5)
    raise AssertionError(f"timed out; last job: {jobs[0] if jobs else None}")


def test_parse_sources():
    text = "check https://www.youtube.com/watch?v=abc123, and\nhttps://youtu.be/xyz).\nhttps://www.youtube.com/watch?v=abc123"
    assert parse_sources(text) == ["https://www.youtube.com/watch?v=abc123", "https://youtu.be/xyz"]
    assert parse_sources(f"  {FIX}  ") == [str(FIX)]
    assert parse_sources("no links here") == []


def test_page_and_auth(server):
    status, _, body = req(server["base"], "/")
    assert status == 200 and server["token"].encode() in body and b"clip-scanner" in body
    assert req(server["base"], "/api/jobs", "POST", {"text": str(FIX)})[0] == 403             # no token
    assert req(server["base"], "/api/jobs", "POST", {"text": str(FIX)}, token="wrong")[0] == 403
    assert req(server["base"], "/api/jobs", headers={"Host": "evil.example:18765"})[0] == 403  # DNS rebinding
    assert req(server["base"], "/api/jobs", "POST", {"text": "nothing"}, token=server["token"])[0] == 400


def test_scan_with_approval_and_results(server):
    base, token = server["base"], server["token"]
    status, _, body = req(base, "/api/jobs", "POST", {"text": str(FIX), "max_clips": 3, "cut": True, "mode": "great"},
                          token=token)
    assert status == 200
    job_id = json.loads(body)["jobs"][0]["id"]
    job = wait_for(base, lambda j: j["status"] in ("awaiting_approval", "failed", "done"))
    assert job["status"] == "awaiting_approval", job.get("error") or job["log"][-10:]
    assert job["estimate"]["expected_usd"] > 0 and job["stages"][0]["state"] in ("done", "cached")
    assert req(base, f"/api/jobs/{job_id}/approve", "POST", token=token)[0] == 200
    job = wait_for(base, lambda j: j["status"] in ("done", "failed"))
    assert job["status"] == "done", job.get("error") or job["log"][-15:]
    assert job["judging"]["great"] >= 1 and job["report_dir"].endswith("-SCRIPTED-TEST")

    reports = json.loads(req(base, "/api/reports")[2])["reports"]
    assert reports and reports[0]["dir"] == job["report_dir"] and reports[0]["scripted"]
    rep = json.loads(req(base, f"/api/reports/{job['report_dir']}")[2])
    clip = rep["great_clips"][0]
    assert clip["file_url"] and clip["raw_url"]
    status, headers, data = req(base, clip["file_url"], headers={"Range": "bytes=0-99"})
    assert status == 206 and len(data) == 100 and headers["Content-Range"].startswith("bytes 0-99/")
    status, headers, _ = req(base, clip["raw_url"] + "?download=1")
    assert status == 200 and "attachment" in headers.get("Content-Disposition", "")


def test_files_cannot_escape_output_dir(server):
    assert req(server["base"], "/files/../config.toml")[0] == 404
    assert req(server["base"], "/files/%2e%2e/config.toml")[0] == 404
    assert req(server["base"], "/api/reports/..%2F..")[0] == 404


def test_cancel_at_approval(server):
    """A video nothing has been judged for yet, so judging needs approval; cancelling spends nothing."""
    import subprocess

    base, token = server["base"], server["token"]
    other = server["tmp"] / "other.mp4"  # different content hash -> fresh workspace, uncached judging
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(FIX), "-t", "39", "-c", "copy", str(other)], check=True)
    job_id = json.loads(req(base, "/api/jobs", "POST", {"text": str(other), "mode": "great"},
                            token=token)[2])["jobs"][0]["id"]
    job = wait_for(base, lambda j: j["id"] == job_id and j["status"] in ("awaiting_approval", "done", "failed"))
    assert job["status"] == "awaiting_approval", job["log"][-10:]
    assert req(base, f"/api/jobs/{job_id}/cancel", "POST", token=token)[0] == 200
    job = wait_for(base, lambda j: j["id"] == job_id and j["finished"])
    assert job["status"] == "cancelled" and job["result_status"] == "unavailable"
    assert job["judging"]["done"] == 0  # nothing was judged


def test_top_mode_scan(server):
    """The default mode: the best moments as raw landscape clips, served to the page. The test judge follows the
    API rule, so with a $0 limit the one ranking call is approved first (on the subscription it just runs)."""
    base, token = server["base"], server["token"]
    status, _, body = req(base, "/api/jobs", "POST", {"text": str(FIX), "cut": True}, token=token)  # config default
    assert status == 200
    job = json.loads(body)["jobs"][0]
    assert job["mode"] == "top"
    assert [s["name"] for s in job["stages"]] == ["Ingest", "Transcribe", "Signals", "Moments", "Ranking", "Export"]
    job_id = job["id"]
    job = wait_for(base, lambda j: j["id"] == job_id and j["status"] in ("done", "failed", "awaiting_approval"))
    assert job["status"] == "awaiting_approval", job.get("error") or job["log"][-15:]
    assert job["estimate"]["uncached"] == 1 and job["estimate"]["expected_calls"] == 1
    assert req(base, f"/api/jobs/{job_id}/approve", "POST", token=token)[0] == 200
    job = wait_for(base, lambda j: j["id"] == job_id and j["status"] in ("done", "failed"))
    assert job["status"] == "done", job.get("error") or job["log"][-15:]
    assert job["report_dir"].endswith("-top-SCRIPTED-TEST") and job["result_status"] == "complete"
    assert all(s["state"] in ("done", "cached") for s in job["stages"]), job["stages"]
    reports = {r["dir"]: r for r in json.loads(req(base, "/api/reports")[2])["reports"]}
    assert reports[job["report_dir"]]["mode"] == "top" and reports[job["report_dir"]]["great"] == 1
    rep = json.loads(req(base, f"/api/reports/{job['report_dir']}")[2])
    m = rep["moments"][0]
    assert rep["mode"] == "top" and m["file_url"] and m["poster_url"]
    status, headers, data = req(base, m["file_url"], headers={"Range": "bytes=0-99"})
    assert status == 206 and len(data) == 100
    page = req(base, "/")[2]
    assert b'DEFAULT_MODE = "top"' in page

    # Re-scan: the ranking is cached and free, so it runs straight through.
    job_id = json.loads(req(base, "/api/jobs", "POST", {"text": str(FIX)}, token=token)[2])["jobs"][0]["id"]
    job = wait_for(base, lambda j: j["id"] == job_id and j["status"] in ("done", "failed", "awaiting_approval"))
    assert job["status"] == "done" and job["result_status"] == "complete", job["log"][-10:]


def test_top_mode_cancel_exports_nothing(server):
    """Cancel at the approval card really cancels: no ranking, no clips, the earlier results untouched."""
    import subprocess

    base, token = server["base"], server["token"]
    other = server["tmp"] / "other-top.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(FIX), "-t", "38", "-c", "copy", str(other)], check=True)
    job_id = json.loads(req(base, "/api/jobs", "POST", {"text": str(other), "mode": "top"},
                            token=token)[2])["jobs"][0]["id"]
    job = wait_for(base, lambda j: j["id"] == job_id and j["status"] in ("awaiting_approval", "done", "failed"))
    assert job["status"] == "awaiting_approval", job["log"][-10:]
    assert req(base, f"/api/jobs/{job_id}/cancel", "POST", token=token)[0] == 200
    job = wait_for(base, lambda j: j["id"] == job_id and j["finished"])
    assert job["status"] == "cancelled" and job["result_status"] == "unavailable"
    assert job["stages"][-1]["state"] == "pending"  # never exported
    out = server["tmp"] / "out" / job["report_dir"]
    assert not list(out.glob("clips/*.mp4"))
    rep = json.loads((out / "report.json").read_text())
    assert rep["status"] == "unavailable" and not rep["moments"]


def test_friendly_errors():
    from clip_scanner.server import friendly_error

    class DownloadError(Exception):
        pass

    msg = friendly_error(DownloadError("ERROR: [youtube] x: Unable to download API page: <urlopen error Tunnel "
                                       "connection failed: 403 Forbidden>; Confirm you are on the latest version"))
    assert msg.startswith("Couldn't download the video") and "403" not in msg  # "page" is not "age"
    assert "signed-in" in friendly_error(DownloadError("ERROR: Sign in to confirm your age"))
    assert "private" in friendly_error(DownloadError("ERROR: Private video"))
    assert friendly_error(FileNotFoundError(2, "No such file", "/x.mp4")).startswith("File not found")


def test_job_card_great_count_drops_duplicates():
    from clip_scanner.server import Job

    job = Job("https://youtu.be/x", None, True)
    job.on_event("candidate", {"id": "c01", "stage": "great", "great": True})
    job.on_event("candidate", {"id": "c02", "stage": "great", "great": True})   # same moment, other opening
    job.on_event("candidate", {"id": "c03", "stage": "error", "great": False})
    job.on_event("judging_done", {"great": 1, "duplicates": ["c02"]})
    assert job.judging["great"] == 1 and job.judging["errors"] == 1 and job.judging["done"] == 2
