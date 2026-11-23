"""Normalise Windows paths inside generated `config.gin` files.

`smart_control.utils.scenario_generator` assembles paths with `os.path.join`, so
on Windows a generated config contains `"C:\\...\\building_0500\\floor_plan.npy"`.
gin parses string literals with Python escape rules: `\\b` becomes a backspace
and `\\f` a form feed, and the building then fails to load its floor plan.

That package must stay byte-identical to the evaluation server, so the emitted
files are rewritten to forward slashes here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

# Any `*_filepath = "..."` / `*_path = "..."` line written by the generators.
_PATH_LINE = re.compile(r'^(\s*[A-Za-z0-9_]*_?filepath\s*=\s*")([^"]*)("\s*)$', re.MULTILINE)


def normalize_gin_paths(config_gin: Path) -> bool:
    """Rewrite backslashes inside filepath literals; return True if it changed."""
    if not config_gin.exists():
        return False
    text = config_gin.read_text()
    fixed = _PATH_LINE.sub(lambda m: m.group(1) + m.group(2).replace("\\", "/") + m.group(3), text)
    if fixed == text:
        return False
    config_gin.write_text(fixed)
    return True
