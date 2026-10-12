"""Per-video workspace with resumable, fingerprinted stages.

Each stage records a fingerprint built from its code version, the config it depends on and
its upstream fingerprints. A stage is skipped when its recorded fingerprint matches and
its outputs still exist, so re-runs only redo what actually changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def fingerprint(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()[:16]


def write_json(path: Path, data: Any) -> None:
    """Atomic JSON write (temp file + rename) so an interrupted run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Workspace:
    """Directory holding every artifact for one video."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._stages_path = self.root / "stages.json"

    def path(self, *names: str) -> Path:
        return self.root.joinpath(*names)

    def _stages(self) -> dict:
        if self._stages_path.exists():
            try:
                return read_json(self._stages_path)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def stage_fingerprint(self, stage: str) -> str | None:
        return self._stages().get(stage, {}).get("fingerprint")

    def is_done(self, stage: str, fp: str, outputs: list[Path] | tuple[Path, ...] = ()) -> bool:
        if self.stage_fingerprint(stage) != fp:
            return False
        return all(Path(p).exists() for p in outputs)

    def mark_done(self, stage: str, fp: str, **info: Any) -> None:
        stages = self._stages()
        stages[stage] = {"fingerprint": fp, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), **info}
        write_json(self._stages_path, stages)

    def invalidate(self, *stages: str) -> None:
        current = self._stages()
        for s in stages:
            current.pop(s, None)
        write_json(self._stages_path, current)
