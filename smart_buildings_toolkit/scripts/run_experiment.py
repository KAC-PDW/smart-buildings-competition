#!/usr/bin/env python3
"""One-command Kaggle Notebook & Local Experiment Runner."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402

# Works whether the toolkit sits at the repo root or in its own subdirectory.
REPO_ROOT = bootstrap(TOOLKIT_DIR)


def is_kaggle_env() -> bool:
    return Path("/kaggle/working").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def ensure_editable_install(skip_install: bool = False) -> None:
    try:
        import smart_buildings  # noqa: F401
        return
    except ImportError:
        if skip_install:
            raise
    cmd = [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation", "-q"]
    print(f"[Setup] Installing starter-kit offline: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized Smart Buildings Experiment Runner")
    parser.add_argument("--mode", choices=["smoke", "rep5", "all40", "export3d"], default="smoke")
    parser.add_argument("--arms", nargs="+", default=["A", "B", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--extra-test-count", type=int, default=10)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--timesfm-model", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--export-building", type=str, default="building_train_1")
    parser.add_argument("--export-weather", type=str, default="fairbanks_1")
    parser.add_argument("--export-steps", type=int, default=288)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    ensure_editable_install(skip_install=args.skip_install)

    from smart_buildings.evaluate import DEFAULT_SCENARIOS_DIR
    from toolkit.evaluator import FULL_EPISODE_STEPS, print_summary, run_rollout, save_results
    from toolkit.exporter_3d import export_rollout_for_3d_viewer
    from toolkit.forecasters import build_forecaster
    from toolkit.policy_core import PolicyConfig
    from toolkit.scenarios import (
        DEFAULT_5_SCENARIOS,
        ensure_extra_test_scenarios,
        load_shipped_pairs,
    )

    out_dir = Path("/kaggle/working/ablation_results") if is_kaggle_env() else (REPO_ROOT / "ablation_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PolicyConfig.load(args.config_json) if args.config_json and args.config_json.exists() else PolicyConfig()

    print(f"=== Smart Buildings Toolkit Runner (env={'Kaggle' if is_kaggle_env() else 'Local'}, mode={args.mode}) ===")

    if args.mode in ("smoke", "rep5", "all40"):
        steps = 288 if args.mode == "smoke" else FULL_EPISODE_STEPS
        extra_n = 2 if args.mode == "smoke" else args.extra_test_count
        sc_dir = REPO_ROOT / DEFAULT_SCENARIOS_DIR

        queue: list[tuple[str, Path]] = []
        pairs = load_shipped_pairs(REPO_ROOT) if args.mode == "all40" else DEFAULT_5_SCENARIOS
        for b, w in pairs:
            queue.append(("shipped", sc_dir / b / w))

        if extra_n > 0:
            extra_dirs = ensure_extra_test_scenarios(
                REPO_ROOT, REPO_ROOT / "smart_buildings" / "extra_test_scenarios", count=extra_n
            )
            for d in extra_dirs:
                queue.append(("unseen_extra", d))

        timesfm_fc = (
            build_forecaster("C", timesfm_model=args.timesfm_model, device=args.device)
            if "C" in args.arms
            else None
        )

        results = []
        for split, cdir in queue:
            if not (cdir / "config.gin").exists():
                print(f"[SKIP] Missing {cdir}")
                continue
            print(f"\n>>> [{split}] {cdir.parent.name}/{cdir.name} (steps={steps})")
            oat_trace = None
            need_oat = "D" in args.arms
            if "A" in args.arms or need_oat:
                res_a, oat_trace = run_rollout(cdir, split, build_forecaster("A"), steps=steps, config=cfg, record_oat=need_oat)
                if "A" in args.arms:
                    results.append(res_a)
                    print(f"  {res_a.arm:18s}: MAE={res_a.mae_vs_zero:.5f}, policy={res_a.policy_wall_s:.2f}s")
            if "B" in args.arms:
                res_b, _ = run_rollout(cdir, split, build_forecaster("B"), steps=steps, config=cfg)
                results.append(res_b)
                print(f"  {res_b.arm:18s}: MAE={res_b.mae_vs_zero:.5f}, policy={res_b.policy_wall_s:.2f}s")
            if "C" in args.arms and timesfm_fc is not None:
                res_c, _ = run_rollout(cdir, split, timesfm_fc, steps=steps, config=cfg)
                results.append(res_c)
                print(f"  {res_c.arm:18s}: MAE={res_c.mae_vs_zero:.5f}, policy={res_c.policy_wall_s:.2f}s (fcst={res_c.forecast_wall_s:.2f}s)")
            if "D" in args.arms and oat_trace is not None:
                res_d, _ = run_rollout(cdir, split, build_forecaster("D", oracle_trace=oat_trace), steps=steps, config=cfg)
                results.append(res_d)
                print(f"  {res_d.arm:18s}: MAE={res_d.mae_vs_zero:.5f}, policy={res_d.policy_wall_s:.2f}s")

        b_path = REPO_ROOT / "baseline_returns.json"
        b_ret = json.loads(b_path.read_text()) if b_path.exists() else {}
        print_summary(results, b_ret)
        csv_p, json_p = save_results(results, out_dir)
        print(f"\nSaved results -> {csv_p}, {json_p}")

    if args.mode in ("smoke", "export3d"):
        cdir = REPO_ROOT / DEFAULT_SCENARIOS_DIR / args.export_building / args.export_weather
        out_json = out_dir / f"rollout_{args.export_building}_{args.export_weather}_ArmA.json"
        export_rollout_for_3d_viewer(cdir, out_json, steps=args.export_steps, config=cfg)


if __name__ == "__main__":
    main()
