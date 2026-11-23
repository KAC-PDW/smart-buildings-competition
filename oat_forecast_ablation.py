#!/usr/bin/env python3
"""Deprecated entry point: delegates to `smart_buildings_toolkit/scripts/run_experiment.py`.

The 4-arm ablation now lives in `smart_buildings_toolkit/toolkit/forecasters.py`;
this wrapper only translates the old CLI so prior notebook cells keep running.
`--shipped-mode all30` maps to `--mode all40` (30 shipped + 10 unseen = 40).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _translate(argv: list[str]) -> list[str]:
    """Map the legacy CLI onto the toolkit runner CLI."""
    out: list[str] = []
    steps: int | None = None
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--shipped-mode":
            mode = argv[i + 1] if i + 1 < len(argv) else "rep5"
            out += ["--mode", {"rep5": "rep5", "all30": "all40"}.get(mode, mode)]
            skip_next = True
        elif arg.startswith("--shipped-mode="):
            mode = arg.split("=", 1)[1]
            out += ["--mode", {"rep5": "rep5", "all30": "all40"}.get(mode, mode)]
        elif arg == "--steps":
            steps = int(argv[i + 1]) if i + 1 < len(argv) else None
            skip_next = True
        elif arg.startswith("--steps="):
            steps = int(arg.split("=", 1)[1])
        elif arg in ("--extra-start-seed", "--scenarios-dir", "--output-dir"):
            skip_next = True
            print(f"[deprecated] ignoring {arg}: the toolkit runner owns it", file=sys.stderr)
        elif arg.split("=", 1)[0] in ("--extra-start-seed", "--scenarios-dir", "--output-dir"):
            print(f"[deprecated] ignoring {arg}: the toolkit runner owns it", file=sys.stderr)
        else:
            out.append(arg)

    if steps is not None and "--mode" not in out:
        if steps == 288:
            out += ["--mode", "smoke"]
        else:
            print(f"[deprecated] ignoring --steps {steps}: use --shipped-mode rep5|all30", file=sys.stderr)
    return out


def main() -> None:
    here = Path(__file__).resolve()
    runner = next(
        p / "scripts" / "run_experiment.py"
        for p in (here.parent, *here.parents)
        if (p / "scripts" / "run_experiment.py").exists()
    )
    print(f"[deprecated] delegating to {runner}", file=sys.stderr)
    sys.argv = [str(runner), *_translate(sys.argv[1:])]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
