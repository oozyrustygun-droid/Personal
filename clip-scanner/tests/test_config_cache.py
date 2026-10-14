import tomllib

import pytest
from pydantic import ValidationError

from clip_scanner.cache import Workspace, fingerprint
from clip_scanner.config import REPO_ROOT, Config, load_config, load_style_guide


def test_repo_config_matches_defaults():
    """config.toml and the code defaults must agree, so the documented file is the real bar."""
    with open(REPO_ROOT / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert Config.model_validate(data).model_dump(exclude={"source_path"}) == Config().model_dump(
        exclude={"source_path"})


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("[judging]\noverall_minn = 5.0\n")
    with pytest.raises(ValidationError):
        load_config(str(bad))


def test_weights_must_cover_rubric(tmp_path):
    bad = tmp_path / "config.toml"
    bad.write_text("[judging.weights]\nhook_strength = 1.0\n")
    with pytest.raises(ValidationError):
        load_config(str(bad))


def test_style_guide_prompt_section_excludes_sources():
    text = load_style_guide()
    assert "type in chat" in text
    assert "## Sources" not in text


def test_workspace_stage_fingerprints(tmp_path):
    ws = Workspace(tmp_path / "vid")
    out = ws.path("x.json")
    fp = fingerprint("stage", 1, {"a": 1})
    assert not ws.is_done("stage", fp, [out])
    out.write_text("{}")
    ws.mark_done("stage", fp)
    assert ws.is_done("stage", fp, [out])
    assert not ws.is_done("stage", fingerprint("stage", 2, {"a": 1}), [out])
    out.unlink()
    assert not ws.is_done("stage", fp, [out])


def test_setting_one_floor_keeps_the_other(tmp_path):
    """Review finding #5: tightening hook_strength must not silently drop the payoff floor."""
    f = tmp_path / "config.toml"
    f.write_text("[judging.criterion_floors]\nhook_strength = 8.5\n")
    cfg = load_config(str(f))
    assert cfg.judging.criterion_floors == {"hook_strength": 8.5, "payoff": 8.0}


@pytest.mark.parametrize("snippet", ["[judging]\noverall_min = nan\n", "[judging]\nrubric_mean_min = inf\n",
                                     "[devils_advocate]\nmin_survival_confidence = nan\n",
                                     "[judging.criterion_floors]\npayoff = nan\n", "[judging]\noverall_min = 11.0\n",
                                     "[gates]\nmax_dead_air_s = nan\n", "[gates]\nmusic_prob = 50\n",
                                     "[gates]\nmusic_bed_max_fraction = 50\n"])
def test_non_finite_or_out_of_range_thresholds_rejected(tmp_path, snippet):
    f = tmp_path / "config.toml"
    f.write_text(snippet)
    with pytest.raises(ValidationError):
        load_config(str(f))


def test_missing_explicit_config_is_an_error(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.toml"))
    monkeypatch.setenv("CLIP_SCANNER_CONFIG", str(tmp_path / "also-nope.toml"))
    with pytest.raises(FileNotFoundError):
        load_config()


def test_local_overrides_merge_over_config(tmp_path):
    from clip_scanner.config import load_config

    (tmp_path / "config.toml").write_text('[judging]\neffort = "high"\noverall_min = 8.5\n')
    (tmp_path / "config.local.toml").write_text('[judging]\neffort = "medium"\n')
    cfg = load_config(str(tmp_path / "config.toml"))
    assert cfg.judging.effort == "medium" and cfg.judging.overall_min == 8.5
