import numpy as np
import pytest

from clip_scanner.boundaries import straddles
from clip_scanner.candidates import Candidate, find_peaks, generate, iou
from clip_scanner.config import Config
from clip_scanner.signals import HOP, Signals
from clip_scanner.transcribe import Transcript, Word


def synthetic(duration=300.0, peaks=(60.0, 150.0, 240.0), dead=((100.0, 104.0),)):
    """Talking every ~0.35 s with sentence ends every ~4 s; loud reactions at `peaks`."""
    words, t, k = [], 0.5, 0
    while t < duration - 1:
        if any(s <= t < e for s, e in dead):
            t = next(e for s, e in dead if s <= t < e)
            continue
        k += 1
        words.append(Word(round(t, 2), round(t + 0.28, 2), "word." if k % 11 == 0 else "word"))
        t += 0.35
    n = int(duration / HOP)
    zeros = lambda: np.zeros(n, dtype=np.float32)  # noqa: E731
    s = {name: zeros() for name in ("rms_db", "loud_spike", "laughter", "scream", "crowd", "gasp", "music",
                                     "speech", "chewing", "reaction", "speech_rate", "speech_rate_z",
                                     "word_active", "visual_change", "visual_z", "cut_near", "interest")}
    s["rms_db"][:] = -20
    for w in words:
        s["word_active"][int((w.start - 0.15) / HOP): int((w.end + 0.15) / HOP) + 1] = 1
    for p in peaks:
        i = int(p / HOP)
        s["interest"][i - 20: i + 20] = 3.0
        s["laughter"][i: i + 25] = 0.9  # 2.5 s of laughter right after the payoff
    s["interest"] += 0.1
    sig = Signals(duration=duration, series=s, scene_cuts=np.array([], dtype=np.float32),
                  silences=np.zeros((0, 2), dtype=np.float32))
    return Transcript(words=words), sig


def test_find_peaks_respects_separation():
    curve = np.zeros(1000)
    curve[[100, 105, 400, 900]] = [5, 4, 3, 2]
    assert find_peaks(curve, 0.1, min_sep_s=2.0, quantile=0.5) == [100, 400, 900]


def test_generate_windows_are_clean_and_cover_payoffs():
    cfg = Config()
    tr, sig = synthetic()
    cands = generate(tr, sig, sig.duration, cfg)
    assert cands, "expected candidates around the planted peaks"
    for c in cands:
        assert cfg.candidates.min_len_s <= c.duration <= cfg.candidates.max_len_s
        assert not straddles(tr.words, c.start) and not straddles(tr.words, c.end)
        assert c.start < c.peak <= c.end + 1e-6
        for (a, b), (x, y) in zip(c.segments, c.segments[1:]):
            assert b <= x  # segments ordered, non-overlapping
    covered = {p for p in (60.0, 150.0, 240.0) if any(c.start <= p <= c.end for c in cands)}
    assert covered == {60.0, 150.0, 240.0}


def test_generate_dedups_overlaps():
    cfg = Config()
    tr, sig = synthetic()
    cands = generate(tr, sig, sig.duration, cfg)
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            assert iou((a.start, a.end), (b.start, b.end)) <= cfg.candidates.nms_iou + 1e-9


def test_dead_air_inside_window_is_jump_cut():
    cfg = Config()
    tr, sig = synthetic(peaks=(110.0,), dead=((100.0, 104.0),))
    cands = generate(tr, sig, sig.duration, cfg)
    spanning = [c for c in cands if c.start < 100.0 and c.end > 104.0]
    assert spanning, "expected a window spanning the dead air"
    for c in spanning:
        assert c.jump_cuts >= 1
        assert all(not (a < 101.0 and b > 103.0) for a, b in c.segments)


def test_candidate_time_mapping_roundtrip():
    c = Candidate(id="c01", segments=[(10.0, 15.0), (20.0, 30.0)], peak=25.0, origin="signal", score=1.0, reason="")
    assert c.duration == pytest.approx(15.0)
    assert c.to_content(12.0) == pytest.approx(2.0)
    assert c.to_content(25.0) == pytest.approx(10.0)
    assert c.to_source(10.0) == pytest.approx(25.0)
    assert c.to_source(c.to_content(27.5)) == pytest.approx(27.5)
    assert Candidate.from_json(c.to_json()) == c
