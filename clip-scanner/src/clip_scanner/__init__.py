"""clip-scanner: find only the GREAT short-form clips in a long-form video."""

__version__ = "0.1.0"


def build_id() -> str:
    """Version plus the git commit it was run from, so a report says exactly which code produced it."""
    import subprocess
    from pathlib import Path

    try:
        rev = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "log", "-1",
                              "--format=%h %cs"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        rev = ""
    return f"{__version__} ({rev})" if rev else __version__
