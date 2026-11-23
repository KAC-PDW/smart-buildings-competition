#!/usr/bin/env python3
"""Deprecated entry point: delegates to `smart_buildings_toolkit/scripts/run_experiment.py`.

Kept so existing Kaggle notebook cells keep working. The toolkit runner is the
single implementation; this file must stay argument-free logic.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve()
    runner = next(
        p / "scripts" / "run_experiment.py"
        for p in (here.parent, *here.parents)
        if (p / "scripts" / "run_experiment.py").exists()
    )
    print(f"[deprecated] delegating to {runner}", file=sys.stderr)
    sys.argv = [str(runner), *sys.argv[1:]]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
