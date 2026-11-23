#!/usr/bin/env python3
"""Offline self-check for the toolkit plumbing: no simulator, no network.

    python smart_buildings_toolkit/scripts/selfcheck.py

Fails if repo-root discovery breaks, if `sync_to_transfer_policy.py` stops
emitting compilable Python, or if PolicyConfig stops round-tripping through JSON.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402


def main() -> None:
    repo_root = bootstrap(TOOLKIT_DIR)

    # 1. Repo root discovery must land on the checkout, not on the toolkit dir.
    for marker in ("pyproject.toml", "smart_buildings/training_set.yaml"):
        assert (repo_root / marker).exists(), f"{marker} missing under {repo_root}"

    # 2. The generated submission policy must be compilable Python.
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "transfer_policy.py"
        subprocess.run(
            [sys.executable, str(TOOLKIT_DIR / "scripts" / "sync_to_transfer_policy.py"), "--target", str(target)],
            check=True,
            cwd=repo_root,
        )
        code = target.read_text()
        compile(code, str(target), "exec")
        first = code.splitlines()[0]
        assert "Auto-generated TransferPolicy" in first, first
        assert "class TransferPolicy" in code

    # 3. PolicyConfig JSON round-trip keeps every field.
    from toolkit.policy_core import PolicyConfig

    cfg = PolicyConfig(kp=0.7, reset_full_k=4.0, use_sensor_band=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_policy_config.json"
        cfg.save(path)
        assert PolicyConfig.load(path) == cfg

    print("selfcheck OK")


if __name__ == "__main__":
    main()
