__all__ = ["__version__"]

from pathlib import Path
from subprocess import DEVNULL, CalledProcessError, check_output

REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_latest_tag() -> str:
    try:
        return (
            check_output(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=REPO_ROOT,
                stderr=DEVNULL,
                text=True,
            )
            .strip()
        )
    except (CalledProcessError, OSError):
        return "0.0.0"

try:
    __version__ = _resolve_latest_tag()
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
