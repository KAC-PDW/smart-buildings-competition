#!/usr/bin/env python3
"""Local 4-Arm Ablation + 10 Unseen Extra Test Scenarios for Smart Buildings.

Compares four arms on the shipped scenarios (5 representative climates or all 30)
PLUS 10 newly generated unseen test scenarios (procedural unseen buildings paired
with unused weather strips, including the held-out `minneapolis` station):

  Arm A (none)     : Improved pid_water baseline (no OAT forecast)
  Arm B (heuristic): Lightweight 24h-lag + damped 1h trend forecaster
  Arm C (timesfm)  : TimesFM 2.5 (200M) zero-shot hourly forecaster (p10/p50/p90)
  Arm D (oracle)   : Ground-truth future 6h OAT upper bound (local dev only)

Place this file at the root of your forked `starter-kit` repository and run:

    # Default: 5 representative shipped climates + 10 unseen extra test scenarios
    python oat_forecast_ablation.py --arms A B D

    # Full 30 shipped scenarios + 10 unseen extra test scenarios (40 total)
    python oat_forecast_ablation.py --arms A B D --shipped-mode all30

    # Include TimesFM 2.5 (requires `timesfm` or `transformers` + torch)
    python oat_forecast_ablation.py --arms A B C D --device cuda

    # Quick 1-day smoke test (NOT comparable to leaderboard baseline_returns.json)
    python oat_forecast_ablation.py --arms A B D --steps 288
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Ensure starter-kit root is importable when run from repo root or scripts/
REPO_ROOT = Path(__file__).resolve().parent
if not (REPO_ROOT / "smart_buildings").exists() and (REPO_ROOT.parent / "smart_buildings").exists():
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from smart_buildings.evaluate import (
    DEFAULT_BUILDINGS_DIR,
    DEFAULT_SCENARIOS_DIR,
    DEFAULT_WEATHER_DIR,
    create_building_meta_data,
    load_hybrid_environment,
)
from smart_buildings.smart_control.utils.scenario_generator import (
    generate_scenario_from_config,
    load_scenario_from_parts,
)

# 5 representative climate scenarios (one per shipped weather station)
DEFAULT_5_SCENARIOS: list[tuple[str, str]] = [
    ("building_train_1", "atlanta_2"),
    ("building_train_1", "denver_7"),
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "phoenix_5"),
    ("building_train_1", "singapore_8"),
]

STEPS_PER_HOUR = 12  # 5-minute simulator timestep
FULL_EPISODE_STEPS = 8640  # 30 days
KAGGLE_LIMIT_SECONDS = 9 * 3600  # 9-hour Kaggle Notebook runtime cap


@dataclass
class ForecastResult:
    """6-hour ahead Outside Air Temperature (OAT) forecast in Kelvin."""

    p50_k: np.ndarray  # shape (horizon_hours,)
    p10_k: np.ndarray  # shape (horizon_hours,)
    p90_k: np.ndarray  # shape (horizon_hours,)


class BaseOATForecaster:
    """Interface for pluggable hourly OAT forecasters."""

    arm_name: str = "base"

    def __init__(self, horizon_hours: int = 6) -> None:
        self.horizon_hours = horizon_hours
        self.step_count = 0
        self.forecast_calls = 0
        self.forecast_wall_s = 0.0
        self._cached: ForecastResult | None = None
        self._oat_history_5m: deque[float] = deque(maxlen=24 * 7 * STEPS_PER_HOUR)

    def reset(self) -> None:
        self.step_count = 0
        self.forecast_calls = 0
        self.forecast_wall_s = 0.0
        self._cached = None
        self._oat_history_5m.clear()

    def update_and_forecast(self, oat_k: float) -> ForecastResult | None:
        """Record current 5-min OAT and refresh forecast once per hour."""
        if np.isfinite(oat_k):
            self._oat_history_5m.append(float(oat_k))
        should_refresh = (self.step_count % STEPS_PER_HOUR == 0)
        self.step_count += 1
        if not should_refresh:
            return self._cached

        t0 = time.perf_counter()
        self._cached = self._predict_hourly(step_idx=self.step_count - 1)
        self.forecast_wall_s += time.perf_counter() - t0
        if self._cached is not None:
            self.forecast_calls += 1
        return self._cached

    def _hourly_series(self) -> np.ndarray:
        """Downsample 5-minute OAT buffer to hourly averages (oldest -> newest)."""
        arr = np.asarray(self._oat_history_5m, dtype=np.float32)
        n_full_hours = len(arr) // STEPS_PER_HOUR
        if n_full_hours == 0:
            return arr[-1:] if len(arr) else np.array([], dtype=np.float32)
        trimmed = arr[-n_full_hours * STEPS_PER_HOUR :]
        return trimmed.reshape(n_full_hours, STEPS_PER_HOUR).mean(axis=1)

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        raise NotImplementedError


class NoForecast(BaseOATForecaster):
    """Arm A: No weather forecast (reactive control only)."""

    arm_name = "Arm_A_Baseline"

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        return None


class PeriodicHeuristicForecaster(BaseOATForecaster):
    """Arm B: Lightweight 24h-lag + damped 1h trend forecaster."""

    arm_name = "Arm_B_Heuristic"

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        hourly = self._hourly_series()
        if len(hourly) < 2:
            return None

        curr = float(hourly[-1])
        slope = float(np.clip(hourly[-1] - hourly[-2], -3.0, 3.0))
        h = np.arange(1, self.horizon_hours + 1, dtype=np.float32)

        if len(hourly) >= 24 + self.horizon_hours:
            yesterday_now = float(hourly[-25])
            yesterday_future = hourly[-24 : -24 + self.horizon_hours]
            level_shift = curr - yesterday_now
            p50 = yesterday_future + level_shift * np.exp(-0.15 * h)
        else:
            p50 = curr + slope * (1.0 - np.exp(-0.35 * h)) / 0.35

        spread = 0.8 + 0.35 * h
        return ForecastResult(p50_k=p50, p10_k=p50 - spread, p90_k=p50 + spread)


class TimesFM25Forecaster(BaseOATForecaster):
    """Arm C: Zero-shot TimesFM 2.5 (200M) hourly forecaster with fallback."""

    arm_name = "Arm_C_TimesFM25"

    def __init__(
        self,
        horizon_hours: int = 6,
        model_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "cpu",
        min_context_hours: int = 24,
    ) -> None:
        super().__init__(horizon_hours=horizon_hours)
        self.model_id = model_id
        self.device = device
        self.min_context_hours = min_context_hours
        self._fallback = PeriodicHeuristicForecaster(horizon_hours=horizon_hours)
        self._model = None
        self._backend = None
        self._load_model()

    def _load_model(self) -> None:
        """Load TimesFM 2.5 via `timesfm` or Hugging Face `transformers`."""
        try:
            import timesfm  # type: ignore

            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.model_id)
            model.compile(
                timesfm.ForecastConfig(
                    max_context=168,
                    max_horizon=max(12, self.horizon_hours),
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                )
            )
            self._model = model
            self._backend = "timesfm"
            return
        except Exception:
            pass

        try:
            import torch
            from transformers import TimesFm2_5ModelForPrediction  # type: ignore

            self._model = TimesFm2_5ModelForPrediction.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(self.device)
            self._model.eval()
            self._backend = "transformers"
        except Exception as exc:
            raise RuntimeError(
                f"TimesFM 2.5 ({self.model_id}) could not be loaded. "
                "Install `timesfm` or `transformers` + `torch`, or omit Arm C (`--arms A B D`)."
            ) from exc

    def reset(self) -> None:
        super().reset()
        self._fallback.reset()

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        hourly = self._hourly_series()
        if len(hourly) < self.min_context_hours:
            self._fallback._oat_history_5m = self._oat_history_5m
            return self._fallback._predict_hourly(step_idx)

        context = hourly[-168:].astype(np.float32)

        if self._backend == "timesfm":
            point_fc, quantile_fc = self._model.forecast(
                horizon=self.horizon_hours,
                inputs=[context],
            )
            p50 = np.asarray(point_fc[0][: self.horizon_hours], dtype=np.float32)
            q = np.asarray(quantile_fc[0][: self.horizon_hours], dtype=np.float32)
            p10 = q[:, 1] if q.ndim == 2 and q.shape[1] >= 10 else p50 - 1.5
            p90 = q[:, 9] if q.ndim == 2 and q.shape[1] >= 10 else p50 + 1.5
            return ForecastResult(p50_k=p50, p10_k=p10, p90_k=p90)

        import torch

        with torch.no_grad():
            inputs = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self._model(past_values=inputs)
            preds = out.mean_predictions[0, : self.horizon_hours].cpu().numpy().astype(np.float32)
            if hasattr(out, "full_predictions") and out.full_predictions is not None:
                full = out.full_predictions[0, : self.horizon_hours].cpu().numpy().astype(np.float32)
                p10 = full[:, 1] if full.ndim == 2 and full.shape[1] >= 10 else preds - 1.5
                p90 = full[:, 9] if full.ndim == 2 and full.shape[1] >= 10 else preds + 1.5
            else:
                p10, p90 = preds - 1.5, preds + 1.5
            return ForecastResult(p50_k=preds, p10_k=p10, p90_k=p90)


class OracleForecaster(BaseOATForecaster):
    """Arm D: Ground-truth future OAT series (theoretical upper bound)."""

    arm_name = "Arm_D_Oracle"

    def __init__(self, future_oat_5m: np.ndarray, horizon_hours: int = 6) -> None:
        super().__init__(horizon_hours=horizon_hours)
        self._future_oat_5m = np.asarray(future_oat_5m, dtype=np.float32)

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        horizon_vals = []
        n = len(self._future_oat_5m)
        for h in range(1, self.horizon_hours + 1):
            start = min(step_idx + (h - 1) * STEPS_PER_HOUR, n - 1)
            end = min(step_idx + h * STEPS_PER_HOUR, n)
            val = float(np.mean(self._future_oat_5m[start:end])) if end > start else float(self._future_oat_5m[-1])
            horizon_vals.append(val)
        p50 = np.asarray(horizon_vals, dtype=np.float32)
        return ForecastResult(p50_k=p50, p10_k=p50, p90_k=p50)


class ForecastAwarePIDPolicy(PIDWaterResetPolicy):
    """Improved pid_water baseline + pluggable OAT forecast anticipation."""

    USE_SENSOR_BAND = True
    WORST_ZONE_GUARD_K = 3.0

    def __init__(
        self,
        building_meta_data: dict[str, Any],
        action_spec: Any,
        forecaster: BaseOATForecaster,
    ) -> None:
        super().__init__(building_meta_data, action_spec)
        self.forecaster = forecaster
        self._primary_oat_idx = next(
            (c["oat_idx"] for c in self.ahu_controllers if c.get("oat_idx") is not None),
            None,
        )
        names = list(self.field_names)
        self._comfort_soon_idx = next(
            (i for i, n in enumerate(names) if "comfort" in n and "soon" in n),
            None,
        )

    def _ahu_error(self, obs: np.ndarray, ctrl: dict[str, Any], low: float, high: float) -> float:
        worst = self._worst_zone_error(obs, ctrl, low, high)
        if abs(worst) >= self.WORST_ZONE_GUARD_K:
            return worst
        temps = [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]
        return self._deadband_error(float(np.mean(temps)), low, high)

    def _forecast_band_shift(
        self,
        obs: np.ndarray,
        hour: int,
        day: int,
        low: float,
        high: float,
        forecast: ForecastResult | None,
    ) -> tuple[float, float]:
        if forecast is None or day >= 5:
            return low, high

        comfort_soon = (
            self._comfort_soon_idx is not None and float(obs[self._comfort_soon_idx]) > 0.5
        )
        hours_to_start = self.morning_start_hour - hour
        pre_window = comfort_soon or (0 < hours_to_start <= 2)
        if not pre_window:
            return low, high

        comfort_lo, comfort_hi = self.comfort_temp_window
        max_p90 = float(np.max(forecast.p90_k))
        min_p10 = float(np.min(forecast.p10_k))

        adj_low, adj_high = low, high
        if max_p90 > comfort_hi + 6.0:
            adj_high = min(adj_high, comfort_hi - 0.5)
        if min_p10 < comfort_lo - 15.0:
            adj_low = max(adj_low, comfort_lo + 0.5)
        return adj_low, adj_high

    def get_action(self, obs: np.ndarray) -> dict[str, Any]:
        oat_k = float(obs[self._primary_oat_idx]) if self._primary_oat_idx is not None else np.nan
        forecast = self.forecaster.update_and_forecast(oat_k)

        hour, day = self._get_hour_and_day(obs)
        is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        base_low, base_high = self.comfort_temp_window if is_work else self.eco_temp_window
        adj_low, adj_high = self._forecast_band_shift(obs, hour, day, base_low, base_high, forecast)

        orig_eco = self.eco_temp_window
        if not is_work:
            self.eco_temp_window = (adj_low, adj_high)
        try:
            action = super().get_action(obs)
        finally:
            self.eco_temp_window = orig_eco

        # Fix pid_water bug: do not run hot-water pump/pressure when no AHU is heating
        cont = action["continuous_action"]
        disc = action["discrete_action"]
        is_heating = self.water_temp_idx is not None and cont[self.water_temp_idx] > -0.99
        if not is_heating:
            if self.water_pressure_idx is not None:
                cont[self.water_pressure_idx] = -1.0
            if 0 <= self.water_discrete_idx < self.n_discrete:
                disc[self.water_discrete_idx] = 0

        return action


@dataclass
class ScenarioResult:
    split: str  # "shipped" or "unseen_extra10"
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


def load_shipped_pairs(repo_root: Path) -> list[tuple[str, str]]:
    """Read the 30 official pairs from smart_buildings/training_set.yaml."""
    ts_path = repo_root / "smart_buildings" / "training_set.yaml"
    data = yaml.safe_load(ts_path.read_text()) or {}
    return [(s["building"], s["weather"]) for s in data.get("scenarios", [])]


def select_unused_weather_strips(repo_root: Path, count: int) -> list[str]:
    """Pick `count` weather strips NOT used in the 30 shipped training scenarios.

    Prioritizes the held-out `minneapolis` station (5 strips) plus 1 unused strip
    from each of the 5 shipped stations so all 6 climates are covered.
    """
    used = {w for _, w in load_shipped_pairs(repo_root)}
    weather_dir = repo_root / DEFAULT_WEATHER_DIR
    all_strips = sorted(p.stem for p in weather_dir.glob("*.yaml"))
    unused = [w for w in all_strips if w not in used]

    # Round-robin across stations: minneapolis first, then the other 5 stations
    by_station: dict[str, list[str]] = {}
    for w in unused:
        st = w.rsplit("_", 1)[0]
        by_station.setdefault(st, []).append(w)

    picked: list[str] = []
    # 1) Take up to 5 minneapolis strips (completely unseen station)
    minn = by_station.pop("minneapolis", [])
    picked.extend(minn[: min(5, count)])
    # 2) Fill remaining slots round-robin across unused strips of the other 5 stations
    stations = sorted(by_station.keys())
    idx = 0
    while len(picked) < count and stations:
        st = stations[idx % len(stations)]
        if by_station[st]:
            picked.append(by_station[st].pop(0))
        else:
            stations.remove(st)
            continue
        idx += 1
    # 3) If still short, use any leftover minneapolis strips
    for w in minn[5:]:
        if len(picked) >= count:
            break
        picked.append(w)
    return picked[:count]


def ensure_extra_test_scenarios(
    repo_root: Path,
    extra_scenarios_dir: Path,
    count: int = 10,
    start_seed: int = 500,
) -> list[Path]:
    """Generate `count` unseen procedural buildings × unused weather strips.

    Reuses already-generated scenarios in `extra_scenarios_dir` when present so
    repeated ablation runs stay fast and deterministic.
    """
    extra_scenarios_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        p.parent for p in extra_scenarios_dir.glob("*/*/config.gin")
    )
    if len(existing) >= count:
        return existing[:count]

    buildings_dir = repo_root / DEFAULT_BUILDINGS_DIR
    weather_dir = repo_root / DEFAULT_WEATHER_DIR

    def _find_procedural_buildings() -> list[Path]:
        return sorted(
            p
            for p in buildings_dir.glob("building_*.yaml")
            if not p.stem.startswith("building_train_")
            and int(p.stem.split("_")[-1]) >= start_seed
        )

    proc_buildings = _find_procedural_buildings()
    if len(proc_buildings) < count:
        # Over-sample slightly because run_floor_generator skips seeds with zones < 20 m2
        sample_n = max(count + 6, int(count * 1.6))
        cmd = [
            sys.executable,
            "-m",
            "smart_buildings.run_floor_generator",
            "--start-seed",
            str(start_seed),
            "--count",
            str(sample_n),
        ]
        print(f"[Extra Test Setup] Generating procedural buildings: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        proc_buildings = _find_procedural_buildings()

    if len(proc_buildings) < count:
        raise RuntimeError(
            f"Requested {count} unseen procedural buildings, but only found {len(proc_buildings)} "
            f"under {buildings_dir} with seed >= {start_seed}."
        )

    unused_weathers = select_unused_weather_strips(repo_root, count)
    generated_dirs: list[Path] = []
    for b_yaml, w_stem in zip(proc_buildings[:count], unused_weathers):
        w_yaml = weather_dir / f"{w_stem}.yaml"
        target_dir = extra_scenarios_dir / b_yaml.stem / w_stem
        if not (target_dir / "config.gin").exists():
            print(f"[Extra Test Setup] Building scenario: {b_yaml.stem} × {w_stem}")
            sc_cfg = load_scenario_from_parts(
                str(b_yaml), str(w_yaml), output_base_dir=str(extra_scenarios_dir)
            )
            res = generate_scenario_from_config(sc_cfg)
            target_dir = Path(res["output_dir"])
        generated_dirs.append(target_dir)

    return generated_dirs


def _step_env(env: Any, action: dict[str, Any]) -> tuple[np.ndarray, float, bool]:
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, _ = out
            return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated or truncated)
        obs, reward, done, _ = out
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done)
    return np.asarray(out.observation, dtype=np.float32), float(out.reward), bool(out.is_last())


def _reset_env(env: Any) -> np.ndarray:
    out = env.reset()
    if isinstance(out, tuple):
        return np.asarray(out[0], dtype=np.float32)
    if hasattr(out, "observation"):
        return np.asarray(out.observation, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def run_one_rollout(
    config_dir: Path,
    split: str,
    forecaster: BaseOATForecaster,
    steps: int,
    record_oat: bool = False,
) -> tuple[ScenarioResult, np.ndarray | None]:
    env = load_hybrid_environment(config_dir / "config.gin")
    meta = create_building_meta_data(env, config_dir)
    action_spec = env.action_spec()

    forecaster.reset()
    policy = ForecastAwarePIDPolicy(meta, action_spec, forecaster=forecaster)

    obs = _reset_env(env)
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
        obs, reward, done = _step_env(env, action)
        env_wall_s += time.perf_counter() - t1

        total_reward += reward
        actual_steps += 1
        if done:
            break

    res = ScenarioResult(
        split=split,
        arm=forecaster.arm_name,
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


def print_split_summary(
    split_name: str, results: list[ScenarioResult], baseline_returns: dict[str, Any]
) -> None:
    sub_all = [r for r in results if r.split == split_name]
    if not sub_all:
        return

    arms = sorted({r.arm for r in sub_all})
    print(f"\n=== [{split_name}] Per-Scenario MAE vs 0 (lower is better) ===")
    header = f"{'Scenario':32s} | {'pid_water':>9s} | " + " | ".join(f"{a:>15s}" for a in arms)
    print(header)
    print("-" * len(header))

    scenarios = sorted({(r.building, r.weather) for r in sub_all})
    by_key = {(r.arm, r.building, r.weather): r for r in sub_all}

    for b, w in scenarios:
        key = f"{b}/{w}"
        ref = baseline_returns.get(key, {}).get("pid_water")
        ref_str = f"{abs(ref)/FULL_EPISODE_STEPS:9.5f}" if ref is not None else f"{'N/A':>9s}"
        cols = [
            f"{by_key[(a, b, w)].mae_vs_zero:15.5f}" if (a, b, w) in by_key else f"{'N/A':>15s}"
            for a in arms
        ]
        print(f"{key:32s} | {ref_str} | " + " | ".join(cols))

    print(f"\n=== [{split_name}] Aggregate Summary & 30-Scenario Runtime Projection ===")
    print(
        f"{'Arm':18s} | {'Mean MAE':>9s} | {'Delta vs A':>10s} | "
        f"{'Policy(s)':>9s} | {'Fcst(s)':>8s} | {'Proj 30x (min)':>14s}"
    )
    print("-" * 82)

    arm_a_mae = None
    arm_rows = []
    for a in arms:
        sub = [r for r in sub_all if r.arm == a]
        mean_mae = float(np.mean([r.mae_vs_zero for r in sub]))
        if a.startswith("Arm_A"):
            arm_a_mae = mean_mae
        tot_pol = sum(r.policy_wall_s for r in sub)
        tot_fcst = sum(r.forecast_wall_s for r in sub)
        scale = 30.0 / max(1, len(sub))
        proj_30_min = (tot_pol * scale) / 60.0
        arm_rows.append((a, mean_mae, tot_pol, tot_fcst, proj_30_min))

    for a, mean_mae, tot_pol, tot_fcst, proj_30_min in arm_rows:
        delta = (mean_mae - arm_a_mae) if arm_a_mae is not None else 0.0
        flag = " (OK)" if proj_30_min * 60 < KAGGLE_LIMIT_SECONDS * 0.7 else " (TIGHT!)"
        print(
            f"{a:18s} | {mean_mae:9.5f} | {delta:+10.5f} | {tot_pol:9.2f} | "
            f"{tot_fcst:8.2f} | {proj_30_min:9.1f}{flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-Arm OAT Forecast Ablation + 10 Unseen Extra Test Scenarios"
    )
    parser.add_argument("--arms", nargs="+", default=["A", "B", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument(
        "--shipped-mode",
        choices=["rep5", "all30", "none"],
        default="rep5",
        help="Shipped scenarios to include: 5 representative climates, all 30, or none.",
    )
    parser.add_argument(
        "--extra-test-count",
        type=int,
        default=10,
        help="Number of unseen procedural test scenarios to generate & evaluate (0 to disable).",
    )
    parser.add_argument("--extra-start-seed", type=int, default=500)
    parser.add_argument("--steps", type=int, default=FULL_EPISODE_STEPS)
    parser.add_argument("--scenarios-dir", type=Path, default=REPO_ROOT / DEFAULT_SCENARIOS_DIR)
    parser.add_argument(
        "--extra-scenarios-dir",
        type=Path,
        default=REPO_ROOT / "smart_buildings" / "extra_test_scenarios",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "ablation_results")
    parser.add_argument("--timesfm-model", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    baseline_path = REPO_ROOT / "baseline_returns.json"
    baseline_returns = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}

    # Build scenario queue: (split_label, config_dir)
    queue: list[tuple[str, Path]] = []
    if args.shipped_mode == "rep5":
        for b, w in DEFAULT_5_SCENARIOS:
            queue.append(("shipped", args.scenarios_dir / b / w))
    elif args.shipped_mode == "all30":
        for b, w in load_shipped_pairs(REPO_ROOT):
            queue.append(("shipped", args.scenarios_dir / b / w))

    if args.extra_test_count > 0:
        extra_dirs = ensure_extra_test_scenarios(
            REPO_ROOT,
            args.extra_scenarios_dir,
            count=args.extra_test_count,
            start_seed=args.extra_start_seed,
        )
        for d in extra_dirs:
            queue.append(("unseen_extra10", d))

    timesfm_fc = (
        TimesFM25Forecaster(model_id=args.timesfm_model, device=args.device)
        if "C" in args.arms
        else None
    )

    all_results: list[ScenarioResult] = []
    for split, config_dir in queue:
        if not (config_dir / "config.gin").exists():
            print(f"[SKIP] Missing scenario: {config_dir}")
            continue

        print(f"\n>>> [{split}] {config_dir.parent.name}/{config_dir.name} (steps={args.steps})")
        oat_trace: np.ndarray | None = None

        need_oat = "D" in args.arms
        if "A" in args.arms or need_oat:
            res_a, oat_trace = run_one_rollout(
                config_dir, split, NoForecast(), steps=args.steps, record_oat=need_oat
            )
            if "A" in args.arms:
                all_results.append(res_a)
                print(f"  {res_a.arm:18s}: MAE={res_a.mae_vs_zero:.5f}, policy={res_a.policy_wall_s:.2f}s")

        if "B" in args.arms:
            res_b, _ = run_one_rollout(
                config_dir, split, PeriodicHeuristicForecaster(), steps=args.steps
            )
            all_results.append(res_b)
            print(f"  {res_b.arm:18s}: MAE={res_b.mae_vs_zero:.5f}, policy={res_b.policy_wall_s:.2f}s")

        if "C" in args.arms and timesfm_fc is not None:
            res_c, _ = run_one_rollout(
                config_dir, split, timesfm_fc, steps=args.steps
            )
            all_results.append(res_c)
            print(
                f"  {res_c.arm:18s}: MAE={res_c.mae_vs_zero:.5f}, "
                f"policy={res_c.policy_wall_s:.2f}s (fcst={res_c.forecast_wall_s:.2f}s)"
            )

        if "D" in args.arms and oat_trace is not None:
            res_d, _ = run_one_rollout(
                config_dir, split, OracleForecaster(oat_trace), steps=args.steps
            )
            all_results.append(res_d)
            print(f"  {res_d.arm:18s}: MAE={res_d.mae_vs_zero:.5f}, policy={res_d.policy_wall_s:.2f}s")

    if not all_results:
        print("No scenarios executed. Check --scenarios-dir.")
        return

    print_split_summary("shipped", all_results, baseline_returns)
    print_split_summary("unseen_extra10", all_results, baseline_returns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ablation_summary.json"
    csv_path = args.output_dir / "ablation_summary.csv"
    json_path.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        for r in all_results:
            writer.writerow(asdict(r))
    print(f"\nSaved detailed results to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()