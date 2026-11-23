#!/usr/bin/env python3
"""One-command Kaggle Notebook & Local Test Runner for KAC-PDW/starter-kit.

Designed for both offline Kaggle Notebooks (`/kaggle/working`) and local setups.
Handles:
  1. Offline-safe editable install check (`pip install -e . --no-deps --no-build-isolation`)
  2. Running the 4-Arm OAT Forecast Ablation (`oat_forecast_ablation.py`) across
     shipped scenarios + 10 unseen procedural test scenarios
  3. Exporting a ready-to-load Three.js 3D Viewer rollout JSON
     (`smart_buildings_3d_viewer.html`) for visual debugging

Usage in a Kaggle Notebook cell (after cloning `KAC-PDW/starter-kit`):

    # Fast 1-day smoke test + 3D viewer JSON export (~1-2 min)
    !python run_kaggle_experiment.py --mode smoke

    # Representative 5 climates + 10 unseen procedural test scenarios (full 30d)
    !python run_kaggle_experiment.py --mode rep5 --arms A B D

    # Full 30 shipped + 10 unseen test scenarios (40 total, full 30d)
    !python run_kaggle_experiment.py --mode all40 --arms A B D

    # Include TimesFM 2.5 on GPU (requires offline wheel/model dataset on Kaggle)
    !python run_kaggle_experiment.py --mode rep5 --arms A B C D --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_kaggle_env() -> bool:
    return Path("/kaggle/working").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def ensure_editable_install(skip_install: bool = False) -> None:
    """Verify `smart_buildings` is importable; install offline if needed."""
    try:
        import smart_buildings  # noqa: F401
        return
    except ImportError:
        if skip_install:
            raise
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "--no-deps",
        "--no-build-isolation",
        "-q",
    ]
    print(f"[Setup] Installing starter-kit offline: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def export_3d_viewer_json(
    building: str,
    weather: str,
    steps: int,
    output_path: Path,
    arm: str = "A",
) -> Path:
    """Run one scenario and export a rollout JSON compatible with smart_buildings_3d_viewer.html."""
    from oat_forecast_ablation import (
        ForecastAwarePIDPolicy,
        NoForecast,
        PeriodicHeuristicForecaster,
        _reset_env,
        _step_env,
    )
    from smart_buildings.evaluate import (
        DEFAULT_SCENARIOS_DIR,
        create_building_meta_data,
        load_hybrid_environment,
    )

    config_dir = REPO_ROOT / DEFAULT_SCENARIOS_DIR / building / weather
    if not (config_dir / "config.gin").exists():
        raise FileNotFoundError(f"Scenario not found for 3D export: {config_dir}")

    env = load_hybrid_environment(config_dir / "config.gin")
    meta = create_building_meta_data(env, config_dir)
    action_spec = env.action_spec()

    forecaster = PeriodicHeuristicForecaster() if arm == "B" else NoForecast()
    forecaster.reset()
    policy = ForecastAwarePIDPolicy(meta, action_spec, forecaster=forecaster)

    obs = _reset_env(env)
    names = list(meta["observation_field_names"])
    occ_idx = next((i for i, n in enumerate(names) if "occupancy" in n), None)

    # Map ordered room_1..room_N observation & action indices
    zone_obs_map: dict[int, int] = {}
    zone_act_map: dict[int, dict[str, int]] = {}
    for ctrl in policy.ahu_controllers:
        for vav_id, o_idx, a_idx in ctrl["zones"]:
            num = int(vav_id.split("_")[-1])
            zone_obs_map[num] = o_idx
            zone_act_map[num] = a_idx
    ordered_rooms = sorted(zone_obs_map.keys())

    frames: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for step in range(steps):
        action = policy.get_action(obs)
        next_obs, reward, done = _step_env(env, action)

        hour, day = policy._get_hour_and_day(obs)
        is_work = bool(day < 5 and policy.morning_start_hour <= hour < policy.evening_start_hour)
        band = list(policy.comfort_temp_window if is_work else policy.eco_temp_window)
        oat_k = float(obs[policy._primary_oat_idx]) if policy._primary_oat_idx is not None else 293.15
        occ = int(round(float(obs[occ_idx]))) if occ_idx is not None else 0

        cont = action["continuous_action"]
        disc = action["discrete_action"]

        zone_temps_k = [
            round(float(policy._denormalize_zone_temp(obs[zone_obs_map[r]])), 2)
            for r in ordered_rooms
        ]
        dampers = [
            round(
                float(
                    (cont[zone_act_map[r]["supply_air_damper_percentage_command"]] + 1.0) * 0.5
                )
                if "supply_air_damper_percentage_command" in zone_act_map[r]
                else 0.1,
                2,
            )
            for r in ordered_rooms
        ]
        reheats = [
            round(
                float(
                    (cont[zone_act_map[r]["reheat_valve_percentage_command"]] + 1.0) * 0.5
                )
                if "reheat_valve_percentage_command" in zone_act_map[r]
                else 0.0,
                2,
            )
            for r in ordered_rooms
        ]

        ahus = []
        any_heating = False
        for i, ctrl in enumerate(policy.ahu_controllers):
            on = int(disc[ctrl["run_discrete_idx"]])
            u_temp = float(cont[ctrl["temp_idx"]])
            sat_k = 285.0 + (u_temp + 1.0) * 0.5 * (305.0 - 285.0)
            p_norm = float(cont[ctrl["pressure_idx"]]) if ctrl["pressure_idx"] is not None else -1.0
            pressure_pa = int(round(max(200.0, (p_norm + 1.0) * 0.5 * 2000.0))) if on else 0
            r_norm = float(cont[ctrl["recirc_idx"]]) if ctrl.get("recirc_idx") is not None else 1.0
            recirc = round(min(0.70, max(0.0, (r_norm + 1.0) * 0.5)), 2)
            if on and u_temp > 0.0:
                any_heating = True
            ahus.append(
                {
                    "id": ctrl["ahu_id"],
                    "on": on,
                    "sat_k": round(sat_k, 2),
                    "pressure_pa": pressure_pa,
                    "recirc": recirc,
                }
            )

        hws_on = int(disc[policy.water_discrete_idx]) if 0 <= policy.water_discrete_idx < len(disc) else 0
        wt_norm = float(cont[policy.water_temp_idx]) if policy.water_temp_idx is not None else -1.0
        wp_norm = float(cont[policy.water_pressure_idx]) if policy.water_pressure_idx is not None else -1.0
        hws_temp_k = 310.0 + (wt_norm + 1.0) * 0.5 * 40.0
        hws_pump_pct = int(round(max(0.0, (wp_norm + 1.0) * 0.5 * 100.0))) if hws_on else 0

        day_num = (step // 288) + 1
        minute = (step * 5) % 60
        frames.append(
            {
                "step": step,
                "time_label": f"Day {day_num} {hour:02d}:{minute:02d}",
                "is_work": is_work,
                "oat_k": round(oat_k, 2),
                "band_k": [round(float(band[0]), 2), round(float(band[1]), 2)],
                "occupancy": occ,
                "reward": round(float(reward), 5),
                "zone_temps_k": zone_temps_k,
                "dampers": dampers,
                "reheats": reheats,
                "ahus": ahus,
                "hws": {
                    "on": hws_on,
                    "temp_k": round(hws_temp_k, 2),
                    "pump_pct": hws_pump_pct,
                    "any_heating": any_heating,
                },
            }
        )

        obs = next_obs
        if done:
            break

    payload = {
        "building": building,
        "weather": weather,
        "policy": forecaster.arm_name,
        "steps": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))
    elapsed = time.perf_counter() - t0
    print(f"[3D Export] Saved {len(frames)} steps -> {output_path} ({elapsed:.1f}s)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle & Local Quick Experiment Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "rep5", "all40", "export3d"],
        default="smoke",
        help="smoke: 1-day fast check; rep5: 5 climates + 10 unseen (30d); all40: 30 shipped + 10 unseen (30d); export3d: 3D JSON only.",
    )
    parser.add_argument("--arms", nargs="+", default=["A", "B", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--extra-test-count", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--timesfm-model", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--export-building", type=str, default="building_train_1")
    parser.add_argument("--export-weather", type=str, default="fairbanks_1")
    parser.add_argument("--export-steps", type=int, default=288)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()

    out_dir = Path("/kaggle/working/ablation_results") if is_kaggle_env() else (REPO_ROOT / "ablation_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Smart Buildings Quick Runner (env={'Kaggle' if is_kaggle_env() else 'Local'}, mode={args.mode}) ===")
    ensure_editable_install(skip_install=args.skip_install)

    if args.mode in ("smoke", "rep5", "all40"):
        steps = 288 if args.mode == "smoke" else 8640
        shipped_mode = "all30" if args.mode == "all40" else "rep5"
        extra_n = 2 if args.mode == "smoke" else args.extra_test_count

        cmd = [
            sys.executable,
            str(REPO_ROOT / "oat_forecast_ablation.py"),
            "--arms",
            *args.arms,
            "--shipped-mode",
            shipped_mode,
            "--extra-test-count",
            str(extra_n),
            "--steps",
            str(steps),
            "--output-dir",
            str(out_dir),
            "--device",
            args.device,
            "--timesfm-model",
            args.timesfm_model,
        ]
        print(f"[Run] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    if args.mode in ("smoke", "export3d"):
        export_path = out_dir / f"rollout_{args.export_building}_{args.export_weather}_ArmA.json"
        export_3d_viewer_json(
            building=args.export_building,
            weather=args.export_weather,
            steps=args.export_steps,
            output_path=export_path,
            arm="A",
        )

    print(f"\n[Done] Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
