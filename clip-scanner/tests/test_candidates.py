import numpy as np
import pytest

from clip_scanner.boundaries import straddles
from clip_scanner.candidates import Candidate, find_peaks, generate, hook_first_start, iou, opens_on_moment
from clip_scanner.config import Config
from clip_scanner.signals import HOP, Signals
from clip_scanner.transcribe import Transcript, Word


def synthetic(duration=300.0, peaks=(60.0, 150.0, 240.0), dead=((100.0, 104.0),), react_at_peak=False,
              strength=None):
    """Talking every ~0.35 s with sentence ends every ~4 s; loud reactions at `peaks`.
    react_at_peak: interest rises exactly where the laughter starts (a real reaction's shape), so the
    reaction itself is the peak; strength: {peak: interest level}."""
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
        level = (strength or {}).get(p, 3.0)
        if react_at_peak:
            s["interest"][i: i + 25] = level
        else:
            s["interest"][i - 20: i + 20] = level
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
    """Overlapping windows are de-duplicated, except two versions of one moment that open differently."""
    cfg = Config()
    tr, sig = synthetic()
    cands = generate(tr, sig, sig.duration, cfg)
    for i, a in enumerate(cands):
        for b in cands[i + 1:]:
            if iou((a.start, a.end), (b.start, b.end)) > cfg.candidates.nms_iou + 1e-9:
                assert a.peak == b.peak
                assert opens_on_moment(a.start, a.peak, cfg) != opens_on_moment(b.start, b.peak, cfg)


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


def test_payoff_after_a_breath_is_not_cut_off():
    """A short dip inside a moment ("what is that thing?" … breath … "OH MY GOD") must not end the window."""
    cfg = Config()
    tr, sig = synthetic(peaks=())
    x = sig.series["interest"]
    x[int(100 / HOP): int(103 / HOP)] = 3.2      # build-up (the found peak)
    x[int(103 / HOP): int(103.8 / HOP)] = 0.3    # breath
    x[int(103.8 / HOP): int(106 / HOP)] = 2.8    # the actual payoff
    cands = generate(tr, sig, sig.duration, cfg)
    around = [c for c in cands if c.start < 100 < c.end]
    assert around and all(c.end >= 106.0 for c in around), [(c.start, c.end) for c in around]


def test_window_does_not_span_two_moments():
    cfg = Config()
    tr, sig = synthetic(peaks=(100.0, 130.0))
    cands = generate(tr, sig, sig.duration, cfg)
    for c in cands:
        assert not (c.start < 101.0 and c.end > 129.0), (c.start, c.end)
    second = [c for c in cands if c.start < 130.0 <= c.end]
    assert second and all(c.start >= 101.5 for c in second), [(c.start, c.end) for c in second]


def _on_reaction(c, t, cfg):
    """The clip opens on the reaction at source time t: it lands inside the hook window."""
    return 0.0 <= t - c.start <= cfg.gates.hook_window_s + 0.25


def test_every_reaction_gets_a_version_that_opens_on_it():
    """A hook in the first 2 s is a hard gate, so each reaction must be offered opening on the reaction."""
    from clip_scanner.judging.gates import deterministic_gates

    cfg = Config()
    tr, sig = synthetic(react_at_peak=True)
    cands = generate(tr, sig, sig.duration, cfg)
    for p in (60.0, 150.0, 240.0):
        on = [c for c in cands if _on_reaction(c, p, cfg)]
        assert on, (p, [(c.start, c.end, c.peak) for c in cands])
        for c in on:
            assert cfg.candidates.min_len_s <= c.duration <= cfg.candidates.max_len_s
            assert not straddles(tr.words, c.start) and not straddles(tr.words, c.end)
            hook = next(g for g in deterministic_gates(c, tr, sig, cfg) if g.name == "hook_onset")
            assert hook.passed, hook.detail


def test_dead_air_after_the_reaction_still_leaves_a_version_that_opens_on_it():
    cfg = Config()
    for dead in (((113.0, 118.0),), ((114.0, 120.0),), ((112.0, 116.0), (117.0, 121.0))):
        tr, sig = synthetic(peaks=(110.0,), dead=dead, react_at_peak=True)
        cands = generate(tr, sig, sig.duration, cfg)
        assert any(_on_reaction(c, 110.0, cfg) for c in cands), (dead, [(c.start, c.end) for c in cands])


def test_a_reaction_first_clip_never_stops_as_the_next_reaction_lands():
    cfg = Config()
    for b in (110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 125.0):
        tr, sig = synthetic(peaks=(100.0, b), dead=(), react_at_peak=True)
        for c in generate(tr, sig, sig.duration, cfg):
            # it either ends before B's own reaction-first opening, or takes all of B's laughter
            assert not (b - 1.0 < c.end < b + 2.5), (b, c.start, c.end)


def test_transcript_moment_keeps_its_own_window_and_respects_the_floor():
    from clip_scanner.proposer import Moment

    cfg = Config()
    tr, sig = synthetic(peaks=(150.0,), dead=(), react_at_peak=True)
    m = Moment(start_s=185.0, end_s=198.0, payoff_s=196.0, kind="punchline", why="deadpan line", strength=3)
    cands = [c for c in generate(tr, sig, sig.duration, cfg, [m]) if c.start <= 196.0 <= c.end]
    assert cands and all(c.start >= 152.0 for c in cands), [(c.start, c.end) for c in cands]
    assert any(abs(c.start - 185.0) < 1.0 for c in cands)  # the proposer's own window is judged


def test_another_moments_window_does_not_remove_the_reaction_first_version():
    from clip_scanner.proposer import Moment

    cfg = Config()
    tr, sig = synthetic(peaks=(100.0,), dead=(), react_at_peak=True)
    m = Moment(start_s=84.0, end_s=110.0, payoff_s=107.0, kind="story", why="follow-up line", strength=3)
    cands = generate(tr, sig, sig.duration, cfg, [m])
    assert any(_on_reaction(c, 100.0, cfg) for c in cands), [(c.start, c.end, c.origin) for c in cands]


def test_reaction_also_flagged_by_the_proposer_still_leads_with_its_reaction():
    from clip_scanner.proposer import Moment

    cfg = Config.model_validate({"candidates": {"alt_openings": False}})
    tr, sig = synthetic(dead=(), react_at_peak=True)
    m = Moment(start_s=138.0, end_s=153.0, payoff_s=150.0, kind="reaction", why="the bite", strength=4)
    around = [c for c in generate(tr, sig, sig.duration, cfg, [m]) if c.start <= 150.0 <= c.end]
    assert around and any(_on_reaction(c, 150.0, cfg) for c in around), [(c.start, c.end) for c in around]


def test_wordless_reaction_opens_on_the_scene_cut():
    words = [Word(96.0, 97.0, "okay")]
    assert hook_first_start(100.3, words, [99.9], 1.6) == pytest.approx(99.9)


def test_window_scores_only_its_own_moment():
    from clip_scanner.candidates import heuristic_score

    tr, sig = synthetic(peaks=(100.0, 111.0), dead=(), react_at_peak=True, strength={100.0: 1.2, 111.0: 6.0})
    segs = [(99.0, 116.0)]
    assert heuristic_score(sig, segs, 100.0, own_end=103.0) < heuristic_score(sig, segs, 100.0) - 3.0


def test_fuzz_invariants():
    """Random streams: every window is clean, bounded, contains its peak, and the budget holds."""
    import random

    from clip_scanner.proposer import Moment

    cfg = Config()
    for seed in range(60):
        rng = random.Random(seed)
        n = rng.randint(1, 6)
        peaks = sorted({round(rng.uniform(20, 280), 1) for _ in range(n)})
        dead = tuple((t, t + rng.uniform(0.5, 5.0)) for t in sorted(rng.uniform(10, 290) for _ in range(rng.randint(0, 4))))
        tr, sig = synthetic(peaks=tuple(peaks), dead=dead, react_at_peak=rng.random() < 0.5)
        props = [Moment(start_s=(a := rng.uniform(10, 260)), end_s=a + rng.uniform(12, 30),
                        payoff_s=a + rng.uniform(8, 12), kind="other", why="x", strength=rng.randint(1, 5))
                 for _ in range(rng.randint(0, 2))]
        cands = generate(tr, sig, sig.duration, cfg, props)
        assert len(cands) <= cfg.candidates.max_candidates
        for c in cands:
            assert cfg.candidates.min_len_s - 1e-6 <= c.duration <= cfg.candidates.max_len_s + 1e-6, (seed, c)
            assert not straddles(tr.words, c.start) and not straddles(tr.words, c.end), (seed, c)
            assert c.start <= c.peak <= c.end + 1e-6, (seed, c.start, c.peak, c.end)
            if "(opens on the moment)" in c.reason:
                assert opens_on_moment(c.start, c.peak, cfg), (seed, c)


def test_reaction_first_version_wins_when_slots_are_tight():
    cfg = Config.model_validate({"candidates": {"min_candidates": 3, "max_candidates": 3}})
    tr, sig = synthetic()
    cands = generate(tr, sig, sig.duration, cfg)
    assert len(cands) == 3 and all(opens_on_moment(c.start, c.peak, cfg) for c in cands)
    only = generate(tr, sig, sig.duration, Config.model_validate({"candidates": {"alt_openings": False}}))
    assert len({c.peak for c in only}) == len(only)  # one version per moment


def test_hook_first_start_never_cuts_a_word():
    # continuous talk ("...and so then") with no pause, a 0.6 s pause, then the reaction "OH NO"
    words = [Word(9.5, 9.9, "and"), Word(9.95, 10.3, "so"), Word(10.35, 10.6, "then"), Word(11.2, 11.5, "OH"),
             Word(11.55, 11.9, "NO")]
    # the earliest phrase start (a word after a pause) inside the window before the peak: "OH"
    assert abs(hook_first_start(11.6, words, [], 1.6) - (11.2 - 0.08)) < 1e-6
    # no phrase start in range: the earliest word in range ("then"), still a clean word edge
    t = hook_first_start(10.5, words, [], 0.3)
    assert abs(t - (10.35 - 0.05)) < 1e-6 and not straddles(words, t)   # 0.05 s after "so" ends
    # wordless reaction (a scream) long after the last word: just before the peak
    assert abs(hook_first_start(20.0, words, [], 1.6) - 19.4) < 1e-6
    # target inside a long word: move to that word's start
    long_word = [Word(11.0, 11.6, "aaaah")]
    t = hook_first_start(11.5, long_word, [], 0.2)      # 11.3 would cut into "aaaah"
    assert t <= 11.0 and not straddles(long_word, t)
