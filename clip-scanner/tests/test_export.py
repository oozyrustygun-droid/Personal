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
