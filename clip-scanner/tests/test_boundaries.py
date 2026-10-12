import random

import pytest

from clip_scanner.boundaries import (
    end_point,
    snap_end,
    snap_start,
    split_sentences,
    start_point,
    straddles,
    tighten,
)
from clip_scanner.transcribe import Word


def mk(spec):
    """spec: list of (start, end, text)."""
    return [Word(s, e, t) for s, e, t in spec]


WORDS = mk([
    (0.50, 0.80, "So"), (0.85, 1.20, "anyway"), (1.25, 1.60, "chat."),          # sentence 0
    (3.00, 3.30, "Wait,"), (3.35, 3.60, "what"), (3.62, 3.80, "is"), (3.82, 4.20, "that?"),  # sentence 1
    (4.50, 4.90, "That's"), (4.92, 5.40, "disgusting!"),                         # sentence 2
    (8.00, 8.40, "Anyway"), (8.45, 8.90, "moving"), (8.95, 9.40, "on."),         # sentence 3 (after laugh gap)
])


def test_split_sentences_on_punctuation_and_pauses():
    sents = split_sentences(WORDS, pause_break_s=0.7)
    assert [(s.i0, s.i1) for s in sents] == [(0, 2), (3, 6), (7, 8), (9, 11)]
    # no punctuation at all: long pauses still split
    bare = mk([(0, 0.3, "a"), (0.35, 0.6, "b"), (2.0, 2.3, "c")])
    assert [(s.i0, s.i1) for s in split_sentences(bare, 0.7)] == [(0, 1), (2, 2)]


def test_speaker_change_splits():
    ws = [Word(0, 0.3, "yo", speaker="A"), Word(0.35, 0.6, "hey", speaker="B")]
    assert len(split_sentences(ws)) == 2


def test_snap_start_lands_on_sentence_start_without_cutting_previous_word():
    sents = split_sentences(WORDS)
    start, i = snap_start(2.6, WORDS, sents)
    assert i == 3                       # "Wait," starts the sentence nearest 2.6
    assert WORDS[2].end <= start <= WORDS[3].start
    assert not straddles(WORDS, start)


def test_snap_start_prefers_scene_cut_in_gap():
    sents = split_sentences(WORDS)
    start, i = snap_start(3.0, WORDS, sents, cuts=[2.7])
    assert i == 3 and start == pytest.approx(2.7)


def test_snap_start_ignores_cut_far_before_word():
    # A cut 1.2 s before the first word would add dead air, so it is not used.
    start = start_point(WORDS, 3, cuts=[1.8])
    assert start == pytest.approx(3.0 - 0.08)


def test_snap_end_never_before_payoff_and_finishes_sentence():
    sents = split_sentences(WORDS)
    end, j = snap_end(3.7, WORDS, sents, duration=12.0)  # payoff inside "is" → finish "that?"
    assert j == 6
    assert end >= 3.7 and WORDS[6].end <= end <= WORDS[7].start
    assert not straddles(WORDS, end)


def test_snap_end_payoff_in_wordless_gap_extends_over_reaction():
    sents = split_sentences(WORDS)
    # Payoff is a laugh at 6.5 s, after "disgusting!" and before "Anyway" (8.0 s).
    end, j = snap_end(6.5, WORDS, sents, duration=12.0, extend_to=7.4)
    assert j == 8
    assert 7.4 <= end <= WORDS[9].start
    assert end >= 6.5


def test_snap_end_payoff_in_mid_sentence_pause_still_reached():
    # 0.6 s pause inside a sentence (shorter than pause_break_s), dense speech after it.
    ws = mk([(0.0, 0.4, "you"), (0.45, 0.9, "know"), (1.5, 1.8, "what"), (1.82, 2.1, "I"), (2.12, 2.4, "mean")])
    sents = split_sentences(ws, 0.7)
    end, j = snap_end(1.4, ws, sents, duration=3.0, search_s=0.2)
    assert end >= 1.4 and not straddles(ws, end)


def test_end_point_does_not_run_into_next_word():
    end = end_point(WORDS, 2, 12.0, tail=5.0)  # huge tail would hit "Wait," at 3.0
    assert end == pytest.approx(3.0)


def test_end_uses_scene_cut_in_tail():
    end = end_point(WORDS, 8, 12.0, cuts=[5.7], tail=0.35)
    assert end == pytest.approx(5.7)


def test_tighten_removes_long_dead_air_and_trims_edges():
    segs = tighten(0.0, 30.0, [(0.0, 2.0), (10.0, 13.0), (20.0, 21.0), (28.0, 30.0)], max_dead=1.5, keep=0.4)
    # leading 2 s trimmed, 3 s gap jump-cut (keeping 0.4 s), 1 s gap kept, trailing 2 s trimmed
    assert segs[0][0] == pytest.approx(1.8)
    assert segs[0][1] == pytest.approx(10.2) and segs[1][0] == pytest.approx(12.8)
    assert segs[-1][1] == pytest.approx(28.2)
    assert len(segs) == 2


def test_tighten_noop_without_long_gaps():
    assert tighten(5.0, 20.0, [(8.0, 9.0)], max_dead=1.5, keep=0.4) == [(5.0, 20.0)]


@pytest.mark.parametrize("seed", range(40))
def test_randomized_invariants(seed):
    rng = random.Random(seed)
    words, t = [], rng.uniform(0, 1)
    for k in range(rng.randint(20, 120)):
        dur = rng.uniform(0.08, 0.6)
        text = rng.choice(["yo", "bro", "what", "is", "that", "no", "way", "chat"])
        if rng.random() < 0.15:
            text += rng.choice([".", "!", "?"])
        words.append(Word(round(t, 3), round(t + dur, 3), text))
        t += dur + (rng.uniform(0.0, 0.15) if rng.random() < 0.8 else rng.uniform(0.3, 2.5))
    duration = t + 1.0
    sents = split_sentences(words, 0.7)
    cuts = sorted(rng.uniform(0, duration) for _ in range(rng.randint(0, 8)))
    for _ in range(10):
        a = rng.uniform(0, duration)
        payoff = rng.uniform(a, duration)
        s = snap_start(a, words, sents, cuts)
        e = snap_end(payoff, words, sents, duration, cuts, extend_to=payoff + rng.uniform(0, 2))
        assert s is not None and e is not None
        start, i = s
        end, j = e
        assert not straddles(words, start), (start, i)
        assert not straddles(words, end), (end, j)
        assert start <= words[i].start + 1e-9
        assert i == 0 or start >= words[i - 1].end - 1e-9
        assert end >= words[j].end - 1e-9
        assert j + 1 >= len(words) or end <= words[j + 1].start + 1e-9
        # never end before the payoff
        assert end >= payoff - 1e-6, (payoff, end, j)
