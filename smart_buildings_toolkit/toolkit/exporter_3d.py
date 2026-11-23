"""Generalized 3D Viewer Rollout JSON Exporter for any scenario & policy."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from smart_buildings.evaluate import create_building_meta_data, load_hybrid_environment
from toolkit.evaluator import reset_env, step_env
from toolkit.forecasters import BaseOATForecaster, NoForecast
from toolkit.policy_core import GeneralizedHVACPolicy, PolicyConfig


def export_rollout_for_3d_viewer(
    config_dir: Path,
    output_path: Path,
    steps: int = 288,
    config: PolicyConfig | None = None,
    forecaster: BaseOATForecaster | None = None,
) -> Path:
    """Run one scenario and write JSON compatible with smart_buildings_3d_viewer.html."""
    env = load_hybrid_environment(config_dir / "config.gin")
    meta = create_building_meta_data(env, config_dir)
    action_spec = env.action_spec()

    fc = forecaster or NoForecast()
    fc.reset()
    policy = GeneralizedHVACPolicy(meta, action_spec, config=config, forecaster=fc)

    obs = reset_env(env)
    names = list(meta["observation_field_names"])
    occ_idx = next((i for i, n in enumerate(names) if "occupancy" in n), None)

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
        next_obs, reward, done = step_env(env, action)

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
                float((cont[zone_act_map[r]["supply_air_damper_percentage_command"]] + 1.0) * 0.5)
                if "supply_air_damper_percentage_command" in zone_act_map[r]
                else 0.1,
                2,
            )
            for r in ordered_rooms
        ]
        reheats = [
            round(
                float((cont[zone_act_map[r]["reheat_valve_percentage_command"]] + 1.0) * 0.5)
                if "reheat_valve_percentage_command" in zone_act_map[r]
                else 0.0,
                2,
            )
            for r in ordered_rooms
        ]

        ahus = []
        any_heating = False
        for ctrl in policy.ahu_controllers:
            on = int(disc[ctrl["run_discrete_idx"]])
            u_temp = float(cont[ctrl["temp_idx"]])
            sat_k = 285.0 + (u_temp + 1.0) * 0.5 * 20.0
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

        frames.append(
            {
                "step": step,
                "time_label": f"Day {(step // 288) + 1} {hour:02d}:{(step * 5) % 60:02d}",
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
                    "temp_k": round(310.0 + (wt_norm + 1.0) * 0.5 * 40.0, 2),
                    "pump_pct": int(round(max(0.0, (wp_norm + 1.0) * 0.5 * 100.0))) if hws_on else 0,
                    "any_heating": any_heating,
                },
            }
        )
        obs = next_obs
        if done:
            break

    payload = {
        "building": config_dir.parent.name,
        "weather": config_dir.name,
        "policy": fc.arm_name,
        "steps": frames,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload))
    print(f"[3D Export] Saved {len(frames)} steps -> {output_path} ({time.perf_counter() - t0:.1f}s)")
    return output_path
