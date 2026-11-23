"""Locate the starter-kit repo root from anywhere this toolkit is unpacked.

The toolkit works both flattened at the repo root (the layout README_TOOLKIT.md
describes) and left in its own subdirectory, so nothing may assume that
`Path(__file__).parents[1]` is the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MARKER = Path("smart_buildings") / "training_set.yaml"


def find_repo_root(start: Path) -> Path:
    """Return the nearest ancestor of `start` that holds the simulator package."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / _MARKER).exists():
            return candidate
    raise RuntimeError(f"Could not find {_MARKER} above {start}; run from inside starter-kit.")


def bootstrap(toolkit_dir: Path) -> Path:
    """Make `smart_buildings` and `toolkit` importable; return the repo root."""
    toolkit_dir = toolkit_dir.resolve()
    root = find_repo_root(toolkit_dir)
    for path in (root, toolkit_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root
