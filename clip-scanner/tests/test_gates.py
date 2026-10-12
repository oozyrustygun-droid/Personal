import numpy as np
import pytest

from clip_scanner.candidates import Candidate
from clip_scanner.config import Config
from clip_scanner.judging.gates import clip_measurements, deterministic_gates
from clip_scanner.signals import HOP, Signals
from clip_scanner.transcribe import Transcript, Word

CFG = Config()


def make_signals(duration=120.0, events=True):
    n = int(duration / HOP)
    names = ("rms_db", "loud_spike", "laughter", "scream", "crowd", "gasp", "music", "speech", "chewing",
             "reaction", "speech_rate", "speech_rate_z", "word_active", "visual_change", "visual_z", "cut_near",
             "interest")
    s = {k: np.zeros(n, dtype=np.float32) for k in names}
    return Signals(duration=duration, series=s, scene_cuts=np.array([], dtype=np.float32),
                   silences=np.zeros((0, 2), dtype=np.float32), events_available=events)


def talk(sig, words_spec):
    """Words every 0.4 s between (start, end) spans; marks word_active like signals.reaction_series."""
    words = []
    for a, b in words_spec:
        t = a
        while t + 0.3 <= b:
            words.append(Word(round(t, 2), round(t + 0.3, 2), "yo"))
            t += 0.4
    from clip_scanner.signals import reaction_series

    _, _, active = reaction_series(Transcript(words=words), len(sig.series["word_active"]))
    sig.series["word_active"][:] = active
    return Transcript(words=words)


def gate(results, name):
    return next(g for g in results if g.name == name)


def cand(start, end, segments=None):
    return Candidate(id="c", segments=segments or [(start, end)], peak=(start + end) / 2, origin="signal",
                     score=1, reason="")


@pytest.mark.parametrize("dur, ok", [(11.9, False), (12.0, True), (60.0, True), (60.2, False)])
def test_length_gate(dur, ok):
    sig = make_signals()
    tr = talk(sig, [(10.0, 10.0 + dur)])
    res = deterministic_gates(cand(10.0, 10.0 + dur), tr, sig, CFG)
    assert gate(res, "length").passed is ok


def test_hook_onset_requires_early_activity():
    sig = make_signals()
    tr = talk(sig, [(11.3, 40.0)])  # first word 1.3 s into a clip starting at 10.0
    assert not gate(deterministic_gates(cand(10.0, 40.0), tr, sig, CFG), "hook_onset").passed
    tr2 = talk(make_signals(), [(10.4, 40.0)])
    sig2 = make_signals()
    talk(sig2, [(10.4, 40.0)])
    assert gate(deterministic_gates(cand(10.0, 40.0), tr2, sig2, CFG), "hook_onset").passed


def test_hook_onset_counts_wordless_reaction():
    sig = make_signals()
    tr = talk(sig, [(11.6, 40.0)])
    sig.series["laughter"][int(10.3 / HOP): int(11.6 / HOP)] = 0.9  # laugh before the first word
    res = deterministic_gates(cand(10.0, 40.0), tr, sig, CFG)
    assert gate(res, "hook_onset").passed
    assert clip_measurements(cand(10.0, 40.0), tr, sig)["first_activity_at"] == pytest.approx(0.3, abs=0.11)


def test_dead_air_gate():
    sig = make_signals()
    tr = talk(sig, [(10.0, 20.0), (21.5, 40.0)])  # last word ends 19.9, next starts 21.5: 1.6 s of silence
    gap = 21.5 - max(w.end for w in tr.words if w.end <= 20.0)
    res = deterministic_gates(cand(10.0, 40.0), tr, sig, CFG)
    assert gap > 1.5 and not gate(res, "dead_air").passed, gap
    sig2 = make_signals()
    tr2 = talk(sig2, [(10.0, 20.0), (21.2, 40.0)])  # 1.3 s of silence
    assert gate(deterministic_gates(cand(10.0, 40.0), tr2, sig2, CFG), "dead_air").passed


def test_dead_air_is_not_understated():
    """Regression: word padding used to make ~1.9 s of real silence measure as 1.5 s."""
    sig = make_signals()
    tr = talk(sig, [(10.0, 20.0), (21.8, 40.0)])  # ~1.9 s of true silence
    m = clip_measurements(cand(10.0, 40.0), tr, sig)
    assert m["longest_dead_air"] >= 1.8


def test_laughter_is_not_dead_air():
    sig = make_signals()
    tr = talk(sig, [(10.0, 20.0), (24.0, 40.0)])
    sig.series["laughter"][int(19.9 / HOP): int(24.1 / HOP)] = 0.8
    assert gate(deterministic_gates(cand(10.0, 40.0), tr, sig, CFG), "dead_air").passed


def test_jump_cut_segments_exclude_removed_silence():
    sig = make_signals()
    tr = talk(sig, [(10.0, 20.0), (24.0, 40.0)])
    c = cand(10.0, 40.0, segments=[(10.0, 20.4), (23.6, 40.0)])
    assert gate(deterministic_gates(c, tr, sig, CFG), "dead_air").passed
    assert clip_measurements(c, tr, sig)["jump_cuts"] == 1


def test_music_bed_gate_and_unmeasured_fallback():
    sig = make_signals()
    tr = talk(sig, [(10.0, 40.0)])
    sig.series["music"][int(10 / HOP): int(30 / HOP)] = 0.9  # 2/3 of the clip
    assert not gate(deterministic_gates(cand(10.0, 40.0), tr, sig, CFG), "music_bed").passed
    sig.series["music"][:] = 0
    sig.series["music"][int(10 / HOP): int(15 / HOP)] = 0.9
    assert gate(deterministic_gates(cand(10.0, 40.0), tr, sig, CFG), "music_bed").passed
    sig_off = make_signals(events=False)
    tr_off = talk(sig_off, [(10.0, 40.0)])
    g = gate(deterministic_gates(cand(10.0, 40.0), tr_off, sig_off, CFG), "music_bed")
    assert not g.passed and "not measured" in g.detail  # judges can't hear: unmeasured music fails closed
    lax = Config.model_validate({"gates": {"require_music_check": False}})
    assert gate(deterministic_gates(cand(10.0, 40.0), tr_off, sig_off, lax), "music_bed").passed


def test_music_gate_uses_configured_threshold():
    """Review finding #8: gates.music_prob used to be ignored (hard-coded 0.5)."""
    sig = make_signals()
    tr = talk(sig, [(10.0, 40.0)])
    sig.series["music"][int(10 / HOP): int(40 / HOP)] = 0.45  # quiet music bed under the whole clip
    assert gate(deterministic_gates(cand(10.0, 40.0), tr, sig, CFG), "music_bed").passed
    strict = Config.model_validate({"gates": {"music_prob": 0.3}})
    assert not gate(deterministic_gates(cand(10.0, 40.0), tr, sig, strict), "music_bed").passed


def test_length_gate_compares_unrounded():
    sig = make_signals()
    tr = talk(sig, [(10.0, 70.1)])
    assert not gate(deterministic_gates(cand(10.0, 70.004), tr, sig, CFG), "length").passed


def test_no_speech_fails():
    sig = make_signals()
    sig.series["loud_spike"][:] = 1.0  # something loud, but no words
    res = deterministic_gates(cand(10.0, 40.0), Transcript(words=[]), sig, CFG)
    assert not gate(res, "has_speech").passed
