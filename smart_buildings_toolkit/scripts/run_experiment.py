#!/usr/bin/env python3
"""One-command Kaggle Notebook & Local Experiment Runner."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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


def _run_single_job(job: dict[str, Any]) -> tuple[Any, Any]:
    """Worker entrypoint for sequential or ProcessPoolExecutor rollouts."""
    os.chdir(job["repo_root"])
    if job["arm_code"] == "TRANSFER":
        import time
        from smart_buildings.baselines.transfer_policy import TransferPolicy
        from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
        from toolkit.evaluator import ScenarioResult, reset_env, step_env

        cdir = Path(job["config_dir"])
        env = load_hybrid_environment(cdir / "config.gin")
        meta = create_building_meta_data(env, cdir)
        policy = TransferPolicy(meta, env.action_spec())
        obs = reset_env(env)
        total_reward = 0.0
        policy_wall_s = 0.0
        env_wall_s = 0.0
        actual_steps = 0
        for _ in range(job["steps"]):
            t0 = time.perf_counter()
            action = policy.get_action(obs)
            policy_wall_s += time.perf_counter() - t0
            t1 = time.perf_counter()
            obs, reward, done = step_env(env, action)
            env_wall_s += time.perf_counter() - t1
            total_reward += reward
            actual_steps += 1
            if done:
                break
        res = ScenarioResult(
            split=job["split"],
            arm=job["arm_label"] or "transfer",
            building=cdir.parent.name,
            weather=cdir.name,
            steps=actual_steps,
            total_reward=total_reward,
            mae_vs_zero=abs(total_reward) / max(1, actual_steps),
            policy_wall_s=policy_wall_s,
            forecast_wall_s=0.0,
            forecast_calls=0,
            env_wall_s=env_wall_s,
        )
        return res, None

    from toolkit.evaluator import run_rollout
    from toolkit.forecasters import build_forecaster
    from toolkit.policy_core import PolicyConfig

    cfg = PolicyConfig(**job["cfg_dict"])
    fc = build_forecaster(
        job["arm_code"],
        timesfm_model=job["timesfm_model"],
        device=job["device"],
        oracle_trace=job["oracle_trace"],
    )
    return run_rollout(
        Path(job["config_dir"]),
        job["split"],
        fc,
        steps=job["steps"],
        config=cfg,
        record_oat=job["record_oat"],
        arm_label=job["arm_label"],
    )


def _build_run2_specs() -> tuple[list[tuple[str, str, Any, bool]], list[tuple[str, str, Any, bool]]]:
    from toolkit.policy_core import PolicyConfig

    cfg_a = PolicyConfig(
        fix_hws_cooling_bug=False,
        fix_sensor_band_precond=False,
        fix_hws_pump_demand_check=False,
    )
    cfg_b = PolicyConfig(
        fix_hws_cooling_bug=True,
        fix_sensor_band_precond=True,
        fix_hws_pump_demand_check=True,
        worst_zone_guard_heat_k=999.0,
        enable_cold_hws_boost=True,
        reheat_full_k=1.5,
        reset_full_k=3.0,
        reset_effort_quantile=0.50,
    )
    cfg_c = PolicyConfig(
        fix_hws_cooling_bug=True,
        fix_sensor_band_precond=True,
        fix_hws_pump_demand_check=True,
        worst_zone_guard_heat_k=999.0,
        enable_cold_hws_boost=True,
        reheat_full_k=1.5,
        reset_full_k=3.0,
        reset_effort_quantile=0.50,
        use_continuous_precond=True,
        adaptive_eco_cap=True,
        pre_window_hours=3,
        precool_margin_k=1.0,
        preheat_margin_k=1.0,
    )
    shipped_specs = [
        ("R2_A_NoHWSFix", "A", cfg_a, True),
        ("R2_B_BugfixOnly", "A", cfg_b, False),
        ("R2_C_Ramp_Heuristic", "B", cfg_c, False),
        ("R2_D_Ramp_AdaptiveFcst", "E", cfg_c, False),
        ("R2_E_Ramp_Oracle", "D", cfg_c, False),
    ]
    extra_specs = [
        ("R2_B_BugfixOnly", "A", cfg_b, False),
        ("R2_C_Ramp_Heuristic", "B", cfg_c, False),
    ]
    return shipped_specs, extra_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalized Smart Buildings Experiment Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "rep5", "all40", "run2", "official30", "export3d"],
        default="smoke",
    )
    parser.add_argument("--arms", nargs="+", default=["A", "B", "D"], choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--extra-test-count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
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

    print(
        f"=== Smart Buildings Toolkit Runner (env={'Kaggle' if is_kaggle_env() else 'Local'}, "
        f"mode={args.mode}, workers={args.workers}) ==="
    )

    if args.mode in ("smoke", "rep5", "all40", "run2", "official30"):
        steps = 288 if args.mode == "smoke" else FULL_EPISODE_STEPS
        if args.mode == "smoke":
            extra_n = 2
        elif args.mode == "official30":
            extra_n = 0
        else:
            extra_n = args.extra_test_count
        sc_dir = REPO_ROOT / DEFAULT_SCENARIOS_DIR

        queue: list[tuple[str, Path]] = []
        pairs = load_shipped_pairs(REPO_ROOT) if args.mode in ("all40", "official30") else DEFAULT_5_SCENARIOS
        for b, w in pairs:
            queue.append(("shipped", sc_dir / b / w))

        if extra_n > 0:
            extra_dirs = ensure_extra_test_scenarios(
                REPO_ROOT, REPO_ROOT / "smart_buildings" / "extra_test_scenarios", count=extra_n
            )
            for d in extra_dirs:
                queue.append(("unseen_extra", d))

        valid_queue = [(split, cdir) for split, cdir in queue if (cdir / "config.gin").exists()]
        for split, cdir in queue:
            if not (cdir / "config.gin").exists():
                print(f"[SKIP] Missing {cdir}")

        results = []
        if args.mode == "official30":
            wave1_jobs = [
                {
                    "repo_root": str(REPO_ROOT),
                    "config_dir": str(cdir),
                    "split": split,
                    "arm_code": "TRANSFER",
                    "arm_label": "transfer",
                    "steps": steps,
                    "cfg_dict": asdict(cfg),
                    "record_oat": False,
                    "oracle_trace": None,
                    "timesfm_model": args.timesfm_model,
                    "device": args.device,
                    "keep_result": True,
                }
                for split, cdir in valid_queue
            ]
            oracle_templates = {}
        elif args.mode == "run2":
            shipped_specs, extra_specs = _build_run2_specs()
            wave1_jobs: list[dict[str, Any]] = []
            oracle_templates: dict[str, dict[str, Any]] = {}
            for split, cdir in valid_queue:
                specs = shipped_specs if split == "shipped" else extra_specs
                for arm_label, arm_code, arm_cfg, rec_oat in specs:
                    job = {
                        "repo_root": str(REPO_ROOT),
                        "config_dir": str(cdir),
                        "split": split,
                        "arm_code": arm_code,
                        "arm_label": arm_label,
                        "steps": steps,
                        "cfg_dict": asdict(arm_cfg),
                        "record_oat": rec_oat,
                        "oracle_trace": None,
                        "timesfm_model": args.timesfm_model,
                        "device": args.device,
                        "keep_result": True,
                    }
                    if arm_code == "D":
                        oracle_templates[str(cdir)] = job
                    else:
                        wave1_jobs.append(job)
        else:
            need_oat = "D" in args.arms
            wave1_jobs = []
            oracle_templates = {}
            for split, cdir in valid_queue:
                ran_tracer = False
                for arm_code in args.arms:
                    if arm_code == "D":
                        oracle_templates[str(cdir)] = {
                            "repo_root": str(REPO_ROOT),
                            "config_dir": str(cdir),
                            "split": split,
                            "arm_code": "D",
                            "arm_label": None,
                            "steps": steps,
                            "cfg_dict": asdict(cfg),
                            "record_oat": False,
                            "oracle_trace": None,
                            "timesfm_model": args.timesfm_model,
                            "device": args.device,
                            "keep_result": True,
                        }
                        continue
                    rec = need_oat and not ran_tracer
                    if rec:
                        ran_tracer = True
                    wave1_jobs.append(
                        {
                            "repo_root": str(REPO_ROOT),
                            "config_dir": str(cdir),
                            "split": split,
                            "arm_code": arm_code,
                            "arm_label": None,
                            "steps": steps,
                            "cfg_dict": asdict(cfg),
                            "record_oat": rec,
                            "oracle_trace": None,
                            "timesfm_model": args.timesfm_model,
                            "device": args.device,
                            "keep_result": True,
                        }
                    )
                if need_oat and not ran_tracer:
                    wave1_jobs.append(
                        {
                            "repo_root": str(REPO_ROOT),
                            "config_dir": str(cdir),
                            "split": split,
                            "arm_code": "A",
                            "arm_label": None,
                            "steps": steps,
                            "cfg_dict": asdict(cfg),
                            "record_oat": True,
                            "oracle_trace": None,
                            "timesfm_model": args.timesfm_model,
                            "device": args.device,
                            "keep_result": False,
                        }
                    )

        if args.workers > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                pending: dict[concurrent.futures.Future, dict[str, Any]] = {
                    pool.submit(_run_single_job, job): job for job in wave1_jobs
                }
                while pending:
                    done, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done:
                        job = pending.pop(fut)
                        res, oat_trace = fut.result()
                        if job["keep_result"]:
                            results.append(res)
                            print(
                                f"  [{res.split}] {res.building}/{res.weather} | {res.arm:24s}: "
                                f"MAE={res.mae_vs_zero:.5f}, policy={res.policy_wall_s:.2f}s "
                                f"(fcst={res.forecast_wall_s:.2f}s)",
                                flush=True,
                            )
                        if oat_trace is not None and job["config_dir"] in oracle_templates:
                            o_job = oracle_templates.pop(job["config_dir"])
                            o_job["oracle_trace"] = oat_trace
                            pending[pool.submit(_run_single_job, o_job)] = o_job
        else:
            timesfm_fc = (
                build_forecaster("C", timesfm_model=args.timesfm_model, device=args.device)
                if args.mode != "run2" and "C" in args.arms
                else None
            )
            oat_traces: dict[str, Any] = {}
            for job in wave1_jobs:
                if job["arm_code"] == "C" and timesfm_fc is not None:
                    res, oat_trace = run_rollout(
                        Path(job["config_dir"]),
                        job["split"],
                        timesfm_fc,
                        steps=job["steps"],
                        config=PolicyConfig(**job["cfg_dict"]),
                        record_oat=job["record_oat"],
                        arm_label=job["arm_label"],
                    )
                else:
                    res, oat_trace = _run_single_job(job)
                if job["keep_result"]:
                    results.append(res)
                    print(
                        f"  [{res.split}] {res.building}/{res.weather} | {res.arm:24s}: "
                        f"MAE={res.mae_vs_zero:.5f}, policy={res.policy_wall_s:.2f}s "
                        f"(fcst={res.forecast_wall_s:.2f}s)",
                        flush=True,
                    )
                if oat_trace is not None:
                    oat_traces[job["config_dir"]] = oat_trace
            for cdir_str, o_job in oracle_templates.items():
                if cdir_str not in oat_traces:
                    continue
                o_job["oracle_trace"] = oat_traces[cdir_str]
                res, _ = _run_single_job(o_job)
                results.append(res)
                print(
                    f"  [{res.split}] {res.building}/{res.weather} | {res.arm:24s}: "
                    f"MAE={res.mae_vs_zero:.5f}, policy={res.policy_wall_s:.2f}s",
                    flush=True,
                )

        results.sort(key=lambda r: (r.split, r.building, r.weather, r.arm))
        b_path = REPO_ROOT / "baseline_returns.json"
        b_ret = json.loads(b_path.read_text()) if b_path.exists() else {}
        print_summary(results, b_ret)
        csv_p, json_p = save_results(results, out_dir)
        print(f"\nSaved results -> {csv_p}, {json_p}")

        if args.mode == "official30":
            official_dir = out_dir / "official30_results"
            for r in results:
                s_dir = official_dir / r.building / r.weather / "transfer"
                s_dir.mkdir(parents=True, exist_ok=True)
                (s_dir / "summary.json").write_text(
                    json.dumps({"total_reward": r.total_reward, "total_steps": r.steps}, indent=2)
                )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "score_leaderboard.py"),
                    str(official_dir),
                    "--policy",
                    "transfer",
                    "--expect",
                    "30",
                ],
                cwd=str(REPO_ROOT),
                check=False,
            )

    if args.mode in ("smoke", "export3d"):
        cdir = REPO_ROOT / DEFAULT_SCENARIOS_DIR / args.export_building / args.export_weather
        out_json = out_dir / f"rollout_{args.export_building}_{args.export_weather}_ArmA.json"
        export_rollout_for_3d_viewer(cdir, out_json, steps=args.export_steps, config=cfg)


if __name__ == "__main__":
    main()
