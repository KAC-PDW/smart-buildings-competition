"""Unified evaluation engine for Smart Buildings Competition."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
from toolkit.forecasters import BaseOATForecaster
from toolkit.policy_core import GeneralizedHVACPolicy, PolicyConfig

FULL_EPISODE_STEPS = 8640
KAGGLE_LIMIT_SECONDS = 9 * 3600


@dataclass
class ScenarioResult:
    split: str
    arm: str
    building: str
    weather: str
    steps: int
    total_reward: float
    mae_vs_zero: float
    policy_wall_s: float
    forecast_wall_s: float
    forecast_calls: int
    env_wall_s: float


def reset_env(env: Any) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=np.float32)
    if hasattr(out, "observation"):
        return np.asarray(out.observation, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def step_env(env: Any, action: dict[str, Any]) -> tuple[np.ndarray, float, bool]:
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, term, trunc, _ = out
            return np.asarray(obs, dtype=np.float32), float(reward), bool(term or trunc)
        obs, reward, done, _ = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done)
    return np.asarray(out.observation, dtype=np.float32), float(out.reward), bool(out.is_last())


def run_rollout(
    config_dir: Path,
    split: str,
    forecaster: BaseOATForecaster,
    steps: int = FULL_EPISODE_STEPS,
    config: PolicyConfig | None = None,
    record_oat: bool = False,
    arm_label: str | None = None,
) -> tuple[ScenarioResult, np.ndarray | None]:
    env = load_hybrid_environment(config_dir / "config.gin")
    meta = create_building_meta_data(env, config_dir)
    action_spec = env.action_spec()

    forecaster.reset()
    policy = GeneralizedHVACPolicy(meta, action_spec, config=config, forecaster=forecaster)

    obs = reset_env(env)
    total_reward = 0.0
    policy_wall_s = 0.0
    env_wall_s = 0.0
    oat_trace: list[float] = []
    oat_idx = policy._primary_oat_idx

    actual_steps = 0
    for _ in range(steps):
        if record_oat and oat_idx is not None:
            oat_trace.append(float(obs[oat_idx]))

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
        split=split,
        arm=arm_label or forecaster.arm_name,
        building=config_dir.parent.name,
        weather=config_dir.name,
        steps=actual_steps,
        total_reward=total_reward,
        mae_vs_zero=abs(total_reward) / max(1, actual_steps),
        policy_wall_s=policy_wall_s,
        forecast_wall_s=forecaster.forecast_wall_s,
        forecast_calls=forecaster.forecast_calls,
        env_wall_s=env_wall_s,
    )
    return res, (np.asarray(oat_trace, dtype=np.float32) if record_oat else None)


def print_summary(results: list[ScenarioResult], baseline_returns: dict[str, Any]) -> None:
    for split_name in sorted({r.split for r in results}):
        sub_all = [r for r in results if r.split == split_name]
        arms = sorted({r.arm for r in sub_all})
        print(f"\n=== [{split_name}] Aggregate Summary & 30-Scenario Runtime Projection ===")
        print(
            f"{'Arm':24s} | {'Scen':>4s} | {'Mean MAE':>9s} | {'Delta vs Ref':>12s} | "
            f"{'Policy(s)':>9s} | {'Fcst(s)':>8s} | {'Proj 30x (min)':>14s}"
        )
        print("-" * 94)

        ref_mae = None
        rows = []
        for a in arms:
            sub = [r for r in sub_all if r.arm == a]
            mean_mae = float(np.mean([r.mae_vs_zero for r in sub]))
            if a.startswith(("Arm_A", "R2_A")):
                ref_mae = mean_mae
            tot_pol = sum(r.policy_wall_s for r in sub)
            tot_fcst = sum(r.forecast_wall_s for r in sub)
            proj_30_min = (tot_pol * (30.0 / max(1, len(sub)))) / 60.0
            rows.append((a, len(sub), mean_mae, tot_pol, tot_fcst, proj_30_min))

        if ref_mae is None and rows:
            ref_mae = rows[0][2]

        for a, n_sc, mean_mae, tot_pol, tot_fcst, proj_30_min in rows:
            delta = (mean_mae - ref_mae) if ref_mae is not None else 0.0
            flag = " (OK)" if proj_30_min * 60 < KAGGLE_LIMIT_SECONDS * 0.7 else " (TIGHT!)"
            print(
                f"{a:24s} | {n_sc:4d} | {mean_mae:9.5f} | {delta:+12.5f} | "
                f"{tot_pol:9.2f} | {tot_fcst:8.2f} | {proj_30_min:9.1f}{flag}"
            )


def save_results(results: list[ScenarioResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation_summary.json"
    csv_path = output_dir / "ablation_summary.csv"
    json_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    return csv_path, json_path
