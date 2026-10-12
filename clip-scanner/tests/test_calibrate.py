from pathlib import Path

from clip_scanner.calibrate import load_manifest, metrics, redecide, sweep, write_thresholds
from clip_scanner.config import GATE_NAMES, RUBRIC_CRITERIA, Config, load_config
from clip_scanner.judging.gates import GateResult
from clip_scanner.judging.panel import JudgeOutcome, decide

CFG = Config()
DET = [GateResult("length", True, "ok")]
RULING = {"argument_convincing": False, "survival_confidence": 8.0, "decisive_issue": ""}


def judged(score: float):
    v = {"first_two_seconds": "", "biggest_weakness": "", "overall_justification": "", "hook_line": "",
         "overall": score, "verdict": "GREAT" if score >= 8.5 else "NOT_GREAT",
         "gates": {g: {"evidence": "", "passed": True} for g in GATE_NAMES},
         "scores": {c: {"justification": "", "score": score} for c in RUBRIC_CRITERIA}}
    outs = [JudgeOutcome(p, verdict=v) for p in CFG.judging.judges]
    return decide("c", DET, outs, RULING, {}, CFG)


def test_redecide_matches_original_decision():
    d = judged(9.0)
    assert redecide(d, CFG).great == d.great


def test_sweep_tightens_when_a_flop_passes():
    # A flop scored 8.6 slips through the 8.5 bar; the hits score 9.0+.
    labeled = [("hit", judged(9.0)), ("hit", judged(9.3)), ("flop", judged(8.6)), ("flop", judged(7.0))]
    assert metrics([(lab, d.great) for lab, d in labeled])["fp"] == 1
    rec = sweep(labeled, CFG)["recommended"]
    assert rec["fp"] == 0 and rec["tp"] == 2
    assert 8.6 < rec["overall_min"] <= 9.0


def test_sweep_keeps_current_bar_when_it_already_separates():
    labeled = [("hit", judged(9.0)), ("flop", judged(7.0))]
    rec = sweep(labeled, CFG)["recommended"]
    assert (rec["overall_min"], rec["rubric_mean_min"], rec["min_survival_confidence"]) == (8.5, 7.5, 7.0)


def test_sweep_never_recommends_a_bar_that_passes_a_flop():
    labeled = [("hit", judged(8.0)), ("flop", judged(8.0))]  # indistinguishable
    out = sweep(labeled, CFG)
    rec = out["recommended"]
    assert rec is None or rec["fp"] == 0


def test_manifest_labels_from_views(tmp_path):
    (tmp_path / "manifest.toml").write_text(
        '[[clip]]\nid="a"\nfile="hits/a.mp4"\nviews=5000000\n'
        '[[clip]]\nid="b"\nfile="flops/b.mp4"\nviews=900\n'
        '[[clip]]\nid="c"\nurl="https://example.com/c"\nviews=300000\n'
        '[[clip]]\nid="d"\nfile="x.mp4"\nlabel="flop"\nviews=9999999\n')
    entries = {e.id: e for e in load_manifest(tmp_path / "manifest.toml", CFG)}
    assert entries["a"].label == "hit" and entries["b"].label == "flop"
    assert entries["c"].label == "ambiguous" and entries["d"].label == "flop"


def test_write_thresholds_preserves_comments(tmp_path):
    src = Path(__file__).resolve().parents[1] / "config.toml"
    dst = tmp_path / "config.toml"
    dst.write_text(src.read_text())
    write_thresholds(dst, {"overall_min": 8.75, "rubric_mean_min": 8.0, "min_survival_confidence": 6.0})
    text = dst.read_text()
    assert "EVERY judge's overall rating" in text
    cfg = load_config(str(dst))
    assert cfg.judging.overall_min == 8.75 and cfg.judging.rubric_mean_min == 8.0
    assert cfg.devils_advocate.min_survival_confidence == 6.0


def test_write_guard_refuses_small_sets_loosening_and_scripted():
    """Review finding #2: --write-config used to lower the bar from 1 hit and 0 flops."""
    from clip_scanner.calibrate import loosened, write_guard

    tighter = {"overall_min": 8.75, "rubric_mean_min": 7.5, "min_survival_confidence": 7.0}
    looser = {"overall_min": 8.5, "rubric_mean_min": 7.5, "min_survival_confidence": 5.0}
    assert loosened(tighter, CFG) == [] and loosened(looser, CFG)
    assert write_guard(tighter, CFG, hits=1, flops=0, scripted=False, allow_loosen=False)       # too little data
    assert write_guard(tighter, CFG, hits=12, flops=12, scripted=True, allow_loosen=False)      # fake verdicts
    assert write_guard(looser, CFG, hits=12, flops=12, scripted=False, allow_loosen=False)      # loosening
    assert write_guard(looser, CFG, hits=12, flops=12, scripted=False, allow_loosen=True) is None
    assert write_guard(tighter, CFG, hits=12, flops=12, scripted=False, allow_loosen=False) is None
    assert write_guard(None, CFG, hits=12, flops=12, scripted=False, allow_loosen=True)
    # A run cut short (plan limit, errors) never writes: unjudged flops would stop constraining the bar.
    assert write_guard(tighter, CFG, hits=12, flops=12, scripted=False, allow_loosen=False, unjudged=1)
    assert write_guard(tighter, CFG, hits=12, flops=12, scripted=False, allow_loosen=False, stopped="usage limit")


def test_partly_judged_clips_never_count():
    """A clip missing a judge, the devil or the arbiter is rejected at every threshold, so it can't
    constrain the sweep; it must not count as informative and must block --write-config."""
    from clip_scanner.calibrate import fully_judged
    from clip_scanner.judging.panel import Decision

    full = Decision("c", False, "panel", ["low"], assessments=[{"unavailable": False}] * 3,
                    arbiter={"survival_confidence": 5.0}, devil={"x": 1})
    assert fully_judged(full, CFG)
    assert not fully_judged(Decision("c", False, "panel", ["low"], errors=["jaded_scroller: timeout"],
                                     arbiter={"survival_confidence": 5.0}), CFG)
    assert not fully_judged(Decision("c", False, "panel", ["low"], assessments=[{"unavailable": True}],
                                     arbiter={"survival_confidence": 5.0}), CFG)   # substitute model answered
    assert not fully_judged(Decision("c", False, "panel", ["low"], arbiter=None), CFG)  # arbiter never ran
    assert not fully_judged(Decision("c", False, "error", ["x"]), CFG)
    assert fully_judged(Decision("c", False, "deterministic_gates", ["gate"]), CFG)
