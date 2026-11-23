#!/usr/bin/env python3
"""Pre-submission audit for smart_buildings/baselines/transfer_policy.py."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402

# Works whether the toolkit sits at the repo root or in its own subdirectory.
REPO_ROOT = bootstrap(TOOLKIT_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TransferPolicy before Kaggle submission")
    parser.add_argument("--steps", type=int, default=288)
    args = parser.parse_args()

    from smart_buildings.baselines.transfer_policy import TransferPolicy
    from smart_buildings.baselines import transfer_policy as tp_module
    from smart_buildings.evaluate import (
        DEFAULT_SCENARIOS_DIR,
        create_building_meta_data,
        load_hybrid_environment,
    )
    from toolkit.evaluator import FULL_EPISODE_STEPS, KAGGLE_LIMIT_SECONDS, reset_env, step_env

    audited = Path(tp_module.__file__)
    print(f"[AUDIT] {audited} (mtime {audited.stat().st_mtime:.0f})")
    if "toolkit.policy_core" not in audited.read_text(errors="ignore"):
        print("[WARN] this file was not written by sync_to_transfer_policy.py; run the sync first")

    test_pairs = [
        ("building_train_1", "fairbanks_1"),
        ("building_train_2", "phoenix_6"),
    ]
    total_pol_s = 0.0
    total_steps = 0

    for b, w in test_pairs:
        cdir = REPO_ROOT / DEFAULT_SCENARIOS_DIR / b / w
        if not (cdir / "config.gin").exists():
            print(f"[SKIP] {cdir} not found")
            continue
        env = load_hybrid_environment(cdir / "config.gin")
        meta = create_building_meta_data(env, cdir)
        spec = env.action_spec()
        policy = TransferPolicy(meta, spec)
        obs = reset_env(env)

        zero_only = True
        for step in range(args.steps):
            t0 = time.perf_counter()
            act = policy.get_action(obs)
            total_pol_s += time.perf_counter() - t0
            total_steps += 1

            assert isinstance(act, dict), "Action must be a dict"
            cont = act["continuous_action"]
            disc = act["discrete_action"]
            assert cont.shape == spec["continuous_action"].shape, f"Continuous shape mismatch on {b}"
            assert disc.shape == spec["discrete_action"].shape, f"Discrete shape mismatch on {b}"
            assert np.all(np.isfinite(cont)), f"Non-finite continuous_action at step {step}"
            assert np.all((cont >= -1.0001) & (cont <= 1.0001)), f"Continuous action out of [-1, 1] at step {step}"
            assert np.all((disc == 0) | (disc == 1)), f"Discrete action not in {{0, 1}} at step {step}"
            if np.any(cont != 0.0) or np.any(disc != 0):
                zero_only = False

            obs, _, done = step_env(env, act)
            if done:
                break

        print(
            f"[PASS] {b}/{w}: dims=({spec['discrete_action'].shape[0]}d, "
            f"{spec['continuous_action'].shape[0]}c), zero_only={zero_only}"
        )
        if zero_only:
            print("  [WARN] Policy returned all-zero actions (middle of normalized range, not off!).")

    if total_steps > 0:
        sec_per_step = total_pol_s / total_steps
        proj_30_s = sec_per_step * FULL_EPISODE_STEPS * 30
        print(
            f"\nProjected 30-scenario policy inference time: {proj_30_s / 60:.1f} min "
            f"(Kaggle limit: {KAGGLE_LIMIT_SECONDS / 60:.0f} min)"
        )
        assert proj_30_s < KAGGLE_LIMIT_SECONDS * 0.85, "Projected inference time exceeds 85% Kaggle runtime budget!"
        print("ALL SUBMISSION CHECKS PASSED.")


if __name__ == "__main__":
    main()
