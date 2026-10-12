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
