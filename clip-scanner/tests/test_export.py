import os
import pytest

from clip_scanner.candidates import Candidate
from clip_scanner.captions import TimedWord, ass_time, build_ass, caption_events, clean_word
from clip_scanner.config import CaptionsCfg, Config
from clip_scanner.boundaries import apply_preroll
from clip_scanner.export import crop_geometry, timed_words
from clip_scanner.facetrack import smooth_path
from clip_scanner.transcribe import Transcript, Word


def test_ass_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(61.234) == "0:01:01.23"
    assert ass_time(3725.5) == "1:02:05.50"


def test_clean_word():
    assert clean_word('"what?"', True) == "WHAT?"
    assert clean_word("chat,", True) == "CHAT"
    assert clean_word("{\\b1}x", False) == "(b1)x"


def test_caption_events_word_by_word_timing():
    cfg = CaptionsCfg()
    words = [TimedWord(0.3, 0.6, "Wait,"), TimedWord(0.7, 1.0, "what"), TimedWord(3.0, 3.4, "is"),
             TimedWord(3.5, 3.9, "that?")]
    ev = caption_events(words, cfg, clip_end=4.5)
    assert [e[2] for e in ev] == ["WAIT", "WHAT", "IS", "THAT?"]
    assert ev[0] == (0.3, 0.7, "WAIT")                   # held until the next word starts
    assert ev[1][1] == pytest.approx(1.0 + cfg.max_word_hold_s)  # long pause: capped hold
    assert ev[-1][1] <= 4.5                              # never past the clip end
    for (s1, e1, _), (s2, _, _) in zip(ev, ev[1:]):
        assert e1 <= s2                                  # one word on screen at a time


def test_caption_grouping():
    cfg = CaptionsCfg(words_per_caption=2)
    words = [TimedWord(i * 0.3, i * 0.3 + 0.25, w) for i, w in enumerate(["a", "b", "c"])]
    assert [e[2] for e in caption_events(words, cfg, 2.0)] == ["A B", "C"]


def test_build_ass_has_style_and_events():
    cfg = CaptionsCfg()
    ass = build_ass([TimedWord(0.1, 0.4, "yo")], cfg, 1080, 1920, 2.0)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert f"Style: Word,{cfg.font},{cfg.font_size}" in ass
    assert ",2,60,60,480," in ass  # bottom-centre alignment, lower-third margin
    assert "Dialogue: 0,0:00:00.10," in ass and "YO" in ass


def test_preroll_clamped_by_previous_word_and_scene_cut():
    words = [Word(9.0, 9.9, "before"), Word(10.0, 10.4, "start")]
    segs = apply_preroll([(9.95, 20.0)], words, [], 0.3)
    assert segs[0][0] >= 9.9  # never includes the tail of "before"
    segs2 = apply_preroll([(12.0, 20.0), (25.0, 30.0)], words, [11.9], 0.3)
    assert segs2[0][0] == pytest.approx(11.9) and segs2[1] == (25.0, 30.0)  # stops at the scene cut
    assert apply_preroll([(12.0, 20.0)], words, [], 0.3)[0][0] == pytest.approx(11.7)


def test_timed_words_follow_jump_cuts():
    tr = Transcript(words=[Word(10.0, 10.3, "a"), Word(15.0, 15.3, "b")])
    c = Candidate("c", [(10.0, 11.0), (14.8, 16.0)], 15.0, "signal", 1.0, "")
    tw = timed_words(c, tr, c.segments)
    assert [(round(w.start, 2), w.text) for w in tw] == [(0.0, "a"), (1.2, "b")]


def test_crop_geometry():
    assert crop_geometry(1920, 1080, 1080, 1920) == (608, 1080, 0)       # landscape: track horizontally
    w, h, y = crop_geometry(1080, 1920, 1080, 1920)
    assert (w, h, y) == (1080, 1920, 0)                                  # already vertical
    w, h, y = crop_geometry(1080, 1350, 1080, 1920)                      # 4:5 (taller than 9:16? no)
    assert w <= 1080 and h <= 1350


def test_smooth_path_ignores_jitter_inside_deadzone():
    cfg = Config()
    times = [i / 6 for i in range(60)]
    xs = [0.5 + (0.01 if i % 2 else -0.01) for i in range(60)]
    path = smooth_path(times, xs, crop_frac=0.316, cfg=cfg)
    assert max(path) - min(path) < 0.005


def test_smooth_path_follows_big_move_and_clamps():
    cfg = Config()
    times = [i / 6 for i in range(120)]
    xs = [0.3] * 30 + [0.95] * 90
    path = smooth_path(times, xs, crop_frac=0.316, cfg=cfg)
    assert path[25] == pytest.approx(0.3, abs=0.05)
    assert path[-1] == pytest.approx(1 - 0.316 / 2, abs=0.01)  # clamped to the frame edge


def test_smooth_path_returns_to_center_when_face_lost():
    cfg = Config()
    times = [i / 6 for i in range(120)]
    xs = [0.25] * 20 + [None] * 100
    path = smooth_path(times, xs, crop_frac=0.316, cfg=cfg)
    assert path[25] == pytest.approx(path[19], abs=0.03)  # held briefly
    assert path[-1] == pytest.approx(0.5, abs=0.06)       # then eased back to centre


def test_stash_restore_cycle(tmp_path):
    """Interrupted runs hide old results; cancelled runs restore them; completed runs keep only what's justified."""
    import json

    from clip_scanner.export import STASH, restore_all, restore_exports, stash_exports

    out = tmp_path / "out"
    (out / "clips").mkdir(parents=True)
    (out / "report.json").write_text(json.dumps({"status": "complete", "verdict": "1 GREAT CLIP"}))
    (out / "report.md").write_text("old")
    for name, fp in (("01-a-c01.mp4", "fpA"), ("02-b-c02.mp4", "fpB")):
        (out / "clips" / name).write_text("video")
        (out / "clips" / name).with_suffix(".json").write_text(json.dumps({"fingerprint": fp}))

    stash_exports(out)
    assert not list((out / "clips").glob("*.mp4")) and not (out / "report.json").exists()
    (out / "report.json").write_text(json.dumps({"status": "in_progress"}))  # the marker a run writes
    assert restore_all(out)                                                   # e.g. cancelled at the cost prompt
    assert json.loads((out / "report.json").read_text())["verdict"] == "1 GREAT CLIP"
    assert len(list((out / "clips").glob("*.mp4"))) == 2 and not (out / STASH).exists()

    stash_exports(out)
    keep = {str(out / "clips" / "01-a-c01.mp4"): "fpA", str(out / "clips" / "02-b-c02.mp4"): "fpCHANGED"}
    restore_exports(out, keep)                     # completed run: only the still-justified render returns
    assert [p.name for p in (out / "clips").glob("*.mp4")] == ["01-a-c01.mp4"]
    assert not (out / STASH).exists()


def test_stash_never_overwrites_last_completed_report_with_a_marker(tmp_path):
    import json

    from clip_scanner.export import restore_all, stash_exports

    out = tmp_path / "out"
    out.mkdir()
    (out / "report.json").write_text(json.dumps({"status": "complete", "verdict": "NO GREAT CLIPS"}))
    stash_exports(out)
    (out / "report.json").write_text(json.dumps({"status": "in_progress"}))   # run crashed here
    stash_exports(out)                                                         # next run starts
    assert restore_all(out)
    assert json.loads((out / "report.json").read_text())["verdict"] == "NO GREAT CLIPS"


def _fake_ffmpeg(dirpath, filters):
    import stat

    dirpath.mkdir(parents=True, exist_ok=True)
    exe = dirpath / "ffmpeg"
    lines = ["Filters:", "  T.. = Timeline support", *(f" ..C {f:<16} V->V       test filter" for f in filters)]
    # echo is a shell builtin: the test PATH holds nothing but the fake ffmpeg
    exe.write_text("#!/bin/sh\n" + "".join(f"echo '{line}'\n" for line in lines))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return exe


def test_keg_only_ffmpeg_full_goes_first_on_path(tmp_path, monkeypatch):
    from clip_scanner.media import has_filter, prefer_full_ffmpeg

    plain = _fake_ffmpeg(tmp_path / "plain", ["sendcmd", "scale"])          # Homebrew's slim ffmpeg
    full = _fake_ffmpeg(tmp_path / "ffmpeg-full" / "bin", ["ass", "sendcmd", "scale"])
    monkeypatch.setenv("PATH", str(plain.parent))
    assert not has_filter("ass") and has_filter("sendcmd")
    assert prefer_full_ffmpeg((str(tmp_path / "missing"), str(full.parent))) == str(full.parent)
    assert os.environ["PATH"].split(os.pathsep)[0] == str(full.parent)
    assert has_filter("ass")
    prefer_full_ffmpeg((str(full.parent),))  # idempotent
    assert os.environ["PATH"].split(os.pathsep).count(str(full.parent)) == 1


def test_text_hook_is_drawn_at_the_top_for_the_whole_clip():
    from clip_scanner.captions import TimedWord, build_ass, clean_title
    from clip_scanner.config import Config

    cfg = Config().captions
    ass = build_ass([TimedWord(0.1, 0.4, "wait")], cfg, 1080, 1920, 15.0, title="He lost his wallet 💀 {\\b1}")
    hook = [ln for ln in ass.splitlines() if ln.startswith("Dialogue: 1,") and ",Hook," in ln]
    assert len(hook) == 1 and "0:00:00.00,0:00:15.00" in hook[0]
    assert "HE LOST HIS WALLET (\\B1)" in hook[0] or "HE LOST HIS WALLET (B1)" in hook[0]
    assert "💀" not in ass and "Style: Hook," in ass
    assert clean_title("  a‍ b  ") == "a b"
    assert "Dialogue: 1," not in build_ass([], cfg, 1080, 1920, 15.0)          # no hook, no title line


def test_burned_hook_only_in_text_hook_mode():
    from clip_scanner.config import Config
    from clip_scanner.export import burned_hook

    pkg = {"c01": {"caption_title": "Joe's flight from hell"}}
    assert burned_hook(Config.model_validate({"judging": {"text_hook": False}}), pkg, "c01") == ""
    on = Config.model_validate({"judging": {"text_hook": True}})
    assert burned_hook(on, pkg, "c01") == "Joe's flight from hell"
    off = Config.model_validate({"judging": {"text_hook": True}, "captions": {"burn_text_hook": False}})
    assert burned_hook(off, pkg, "c01") == ""


def test_text_hook_clip_is_never_exported_without_its_hook(monkeypatch):
    from types import SimpleNamespace

    from clip_scanner import export
    from clip_scanner.media import MediaError

    monkeypatch.setattr(export, "has_filter", lambda name: False)  # Homebrew's plain ffmpeg: no libass
    monkeypatch.setattr(export, "probe", lambda p: SimpleNamespace(display_size=(1920, 1080)))
    monkeypatch.setattr(export, "require_binary", lambda n: n)
    a = SimpleNamespace(meta=SimpleNamespace(path="x.mp4"), transcript=SimpleNamespace(words=[]),
                        ws=SimpleNamespace(root="."))
    cand = SimpleNamespace(segments=[(10.0, 25.0)])
    cfg = Config.model_validate({"judging": {"text_hook": True}, "export": {"crop_mode": "center"}})
    with pytest.raises(MediaError, match="text hook"):
        export.render_clip(a, cand, __import__("pathlib").Path("out.mp4"), cfg, title="He lost his wallet")
