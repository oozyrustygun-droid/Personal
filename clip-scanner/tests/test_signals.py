import numpy as np

from clip_scanner.signals import HOP, loudness_db, reaction_series, silence_runs, speech_baseline
from clip_scanner.transcribe import Transcript, Word, _clean_words


def test_loudness_and_silence_runs():
    sr = 16000
    t = np.arange(sr * 3) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    tone[sr: 2 * sr] = 0.0  # one silent second in the middle
    db = loudness_db(tone.astype(np.float32), sr)
    assert len(db) == 30
    runs = silence_runs(db, -42.0)
    assert len(runs) == 1
    assert abs(runs[0][0] - 1.0) < 0.11 and abs(runs[0][1] - 2.0) < 0.11


def test_speech_baseline_ignores_silence():
    # 60 s: alternating 1 s of speech at -20 dB and 1 s of silence at -60 dB.
    per = int(1 / HOP)
    sec = np.concatenate([np.full(per, -20.0), np.full(per, -60.0)] * 30).astype(np.float32)
    base = speech_baseline(sec, -42.0)
    assert np.allclose(base, -20.0, atol=0.5)


def test_reaction_series_marks_catchphrases_and_exclamations():
    words = [Word(0.0, 0.3, "okay"), Word(0.4, 0.6, "type"), Word(0.6, 0.7, "in"), Word(0.7, 1.0, "chat!"),
             Word(2.0, 2.3, "moving"), Word(2.3, 2.6, "on.")]
    n = 40
    reaction, rate, active = reaction_series(Transcript(words=words), n)
    assert reaction[int(0.5 / HOP)] == 1.0      # "type in chat"
    assert reaction[int(2.1 / HOP)] == 0.0      # "moving on"
    assert active[int(0.2 / HOP)] == 1.0 and active[int(1.6 / HOP)] == 0.0
    assert rate.max() > 0


def test_clean_words_removes_overlap_and_empties():
    raw = [Word(0.0, 0.6, " hello"), Word(0.5, 0.9, "world"), Word(1.0, 1.0, "  "), Word(1.2, 1.2, "x")]
    out = _clean_words(raw)
    assert [w.text for w in out] == ["hello", "world", "x"]
    assert out[0].end <= out[1].start
    assert out[2].end > out[2].start
