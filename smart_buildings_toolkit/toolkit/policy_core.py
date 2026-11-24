"""Generalized, parameterizable HVAC policy built on starter-kit pid_water."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
from toolkit.forecasters import BaseOATForecaster, ForecastResult, NoForecast


@dataclass
class PolicyConfig:
    """Tunable parameters for GeneralizedHVACPolicy (serializable to JSON)."""

    kp: float = 0.50
    ki: float = 0.02
    kd: float = 0.10
    zone_damper_full_k: float = 2.0
    reheat_full_k: float = 2.0
    reset_full_k: float = 3.2
    reset_effort_quantile: float = 0.40
    worst_zone_guard_k: float = 3.0
    use_sensor_band: bool = True
    fix_hws_cooling_bug: bool = True
    precool_trigger_delta_k: float = 6.0
    precool_margin_k: float = 0.5
    preheat_trigger_delta_k: float = 15.0
    preheat_margin_k: float = 0.5
    pre_window_hours: int = 2
    fix_sensor_band_precond: bool = True
    fix_hws_pump_demand_check: bool = True
    worst_zone_guard_heat_k: float | None = None
    enable_cold_hws_boost: bool = False
    cold_oat_hws_floor_start_k: float = 273.15
    cold_oat_hws_floor_full_k: float = 253.15
    use_continuous_precond: bool = False
    precool_ramp_start_delta_k: float = -1.0
    preheat_ramp_start_delta_k: float = 5.0
    adaptive_eco_cap: bool = False
    eco_cap_hot_oat_margin_k: float = 2.0
    eco_cap_cold_oat_k: float = 263.15

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PolicyConfig":
        return cls(**json.loads(path.read_text()))


class GeneralizedHVACPolicy(PIDWaterResetPolicy):
    """Generalized HVAC controller with safety guards, HWS fix, and residual hook."""

    def __init__(
        self,
        building_meta_data: Mapping[str, Any],
        action_spec: Any,
        config: PolicyConfig | None = None,
        forecaster: BaseOATForecaster | None = None,
        residual_fn: Callable[[np.ndarray, dict[str, np.ndarray], Mapping[str, Any]], np.ndarray] | None = None,
    ) -> None:
        self.cfg = config or PolicyConfig()
        self.USE_SENSOR_BAND = bool(self.cfg.use_sensor_band)
        self.REHEAT_FULL_K = float(self.cfg.reheat_full_k)
        self.RESET_FULL_K = float(self.cfg.reset_full_k)
        self.RESET_EFFORT_QUANTILE = float(self.cfg.reset_effort_quantile)

        super().__init__(
            building_meta_data,
            action_spec,
            kp=self.cfg.kp,
            ki=self.cfg.ki,
            kd=self.cfg.kd,
            zone_damper_full_k=self.cfg.zone_damper_full_k,
        )
        self.forecaster = forecaster or NoForecast()
        self.residual_fn = residual_fn

        self._primary_oat_idx = next(
            (c["oat_idx"] for c in self.ahu_controllers if c.get("oat_idx") is not None),
            None,
        )
        names = list(self.field_names)
        self._comfort_soon_idx = next(
            (i for i, n in enumerate(names) if "comfort" in n and "soon" in n),
            None,
        )
        self._reheat_action_indices = [
            action_indices["reheat_valve_percentage_command"]
            for ctrl in self.ahu_controllers
            for _, _, action_indices in ctrl["zones"]
            if "reheat_valve_percentage_command" in action_indices
        ]
        self._active_band_override: tuple[float, float] | None = None
        self._last_oat_k: float = float("nan")
        self._any_ahu_heating: bool = False

    def _zone_band(self, obs: np.ndarray, vav_id: str, low: float, high: float) -> tuple[float, float]:
        lo, hi = super()._zone_band(obs, vav_id, low, high)
        if self.cfg.fix_sensor_band_precond and self._active_band_override is not None:
            ov_lo, ov_hi = self._active_band_override
            lo = max(lo, ov_lo)
            hi = min(hi, ov_hi)
            if hi <= lo + 0.1:
                hi = lo + 0.1
        return lo, hi

    def _ahu_error(self, obs: np.ndarray, ctrl: dict[str, Any], low: float, high: float) -> float:
        worst = self._worst_zone_error(obs, ctrl, low, high)
        heat_guard = (
            self.cfg.worst_zone_guard_k
            if self.cfg.worst_zone_guard_heat_k is None
            else self.cfg.worst_zone_guard_heat_k
        )
        if (worst < 0.0 and abs(worst) >= self.cfg.worst_zone_guard_k) or (
            worst > 0.0 and worst >= heat_guard
        ):
            err = worst
        else:
            temps = [self._denormalize_zone_temp(obs[j]) for j in ctrl["zone_obs_indices"]]
            err = self._deadband_error(float(np.mean(temps)), low, high)
        if err > 0.0:
            self._any_ahu_heating = True
        return err

    def _cold_oat_hws_floor(self) -> float:
        if not self.cfg.enable_cold_hws_boost or not np.isfinite(self._last_oat_k):
            return 0.0
        start_k = self.cfg.cold_oat_hws_floor_start_k
        full_k = self.cfg.cold_oat_hws_floor_full_k
        if self._last_oat_k >= start_k:
            return 0.0
        return float(np.clip((start_k - self._last_oat_k) / max(1.0, start_k - full_k), 0.0, 1.0))

    def _water_fraction(self, shortfall: float) -> float:
        frac = super()._water_fraction(shortfall)
        return max(frac, self._cold_oat_hws_floor())

    def _hours_to_morning_start(self, hour: int, day: int) -> int:
        if day < 5 and hour < self.morning_start_hour:
            return self.morning_start_hour - hour
        if ((day + 1) % 7) < 5 and hour >= self.evening_start_hour:
            return (24 - hour) + self.morning_start_hour
        return 999

    def _forecast_band_shift(
        self,
        obs: np.ndarray,
        hour: int,
        day: int,
        low: float,
        high: float,
        forecast: ForecastResult | None,
        *,
        is_work: bool | None = None,
        oat_k: float = float("nan"),
    ) -> tuple[float, float]:
        comfort_lo, comfort_hi = self.comfort_temp_window
        comfort_soon = (
            self._comfort_soon_idx is not None and float(obs[self._comfort_soon_idx]) > 0.5
        )
        if not self.cfg.use_continuous_precond:
            if forecast is None or day >= 5:
                return low, high
            hours_to_start = self.morning_start_hour - hour
            if not (comfort_soon or (0 < hours_to_start <= self.cfg.pre_window_hours)):
                return low, high

            adj_low, adj_high = low, high
            if float(np.max(forecast.p90_k)) > comfort_hi + self.cfg.precool_trigger_delta_k:
                adj_high = min(adj_high, comfort_hi - self.cfg.precool_margin_k)
            if float(np.min(forecast.p10_k)) < comfort_lo - self.cfg.preheat_trigger_delta_k:
                adj_low = max(adj_low, comfort_lo + self.cfg.preheat_margin_k)
            return adj_low, adj_high

        if is_work is None:
            is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour

        adj_low, adj_high = low, high
        if self.cfg.adaptive_eco_cap and not is_work:
            hot_ref = float(np.max(forecast.p50_k)) if forecast is not None else float("nan")
            if (np.isfinite(oat_k) and oat_k > comfort_hi - self.cfg.eco_cap_hot_oat_margin_k) or (
                np.isfinite(hot_ref) and hot_ref > comfort_hi - self.cfg.eco_cap_hot_oat_margin_k
            ):
                adj_high = min(adj_high, comfort_hi + 1.5)
            cold_ref = float(np.min(forecast.p50_k)) if forecast is not None else float("nan")
            if (np.isfinite(oat_k) and oat_k < self.cfg.eco_cap_cold_oat_k) or (
                np.isfinite(cold_ref) and cold_ref < self.cfg.eco_cap_cold_oat_k
            ):
                adj_low = max(adj_low, comfort_lo - 1.5)

        hours_to_start = self._hours_to_morning_start(hour, day)
        in_weekday_pre = (day < 5 and comfort_soon and not is_work) or (
            0 < hours_to_start <= self.cfg.pre_window_hours
        )
        if in_weekday_pre and forecast is not None:
            prog = (
                1.0
                if comfort_soon
                else float(np.clip(1.0 - (hours_to_start - 1) / max(1, self.cfg.pre_window_hours), 0.25, 1.0))
            )
            precool_thresh = comfort_hi + self.cfg.precool_ramp_start_delta_k
            p90_max = float(np.max(forecast.p90_k))
            if p90_max > precool_thresh:
                sev = float(np.clip((p90_max - precool_thresh) / 10.0, 0.0, 1.0))
                adj_high = min(adj_high, comfort_hi - prog * sev * self.cfg.precool_margin_k)

            preheat_thresh = comfort_lo - self.cfg.preheat_ramp_start_delta_k
            p10_min = float(np.min(forecast.p10_k))
            if p10_min < preheat_thresh:
                sev = float(np.clip((preheat_thresh - p10_min) / 15.0, 0.0, 1.0))
                adj_low = max(adj_low, comfort_lo + prog * sev * self.cfg.preheat_margin_k)

        if adj_high <= adj_low + 0.5:
            mid = 0.5 * (comfort_lo + comfort_hi)
            adj_low = min(adj_low, mid - 0.25)
            adj_high = max(adj_high, adj_low + 0.5)
        return adj_low, adj_high

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        oat_k = float(obs[self._primary_oat_idx]) if self._primary_oat_idx is not None else float("nan")
        self._last_oat_k = oat_k
        forecast = self.forecaster.update_and_forecast(oat_k)

        hour, day = self._get_hour_and_day(obs)
        is_work = day < 5 and self.morning_start_hour <= hour < self.evening_start_hour
        base_low, base_high = self.comfort_temp_window if is_work else self.eco_temp_window
        adj_low, adj_high = self._forecast_band_shift(
            obs, hour, day, base_low, base_high, forecast, is_work=is_work, oat_k=oat_k
        )

        orig_eco = self.eco_temp_window
        self._any_ahu_heating = False
        if not is_work:
            self.eco_temp_window = (adj_low, adj_high)
            if self.cfg.fix_sensor_band_precond and (adj_low > base_low or adj_high < base_high):
                self._active_band_override = (adj_low, adj_high)
        try:
            action = super().get_action(obs)
        finally:
            self.eco_temp_window = orig_eco
            self._active_band_override = None

        cont = np.asarray(action["continuous_action"], dtype=np.float32)
        disc = np.asarray(action["discrete_action"], dtype=np.int32)

        if self.cfg.fix_hws_cooling_bug:
            if self.cfg.fix_hws_pump_demand_check:
                any_vav_reheat = any(cont[idx] > -0.99 for idx in self._reheat_action_indices)
                any_heating = self._any_ahu_heating or any_vav_reheat
                if any_heating:
                    if 0 <= self.water_discrete_idx < self.n_discrete:
                        disc[self.water_discrete_idx] = 1
                    if self.water_pressure_idx is not None:
                        cont[self.water_pressure_idx] = 1.0
                    if (
                        self.cfg.enable_cold_hws_boost
                        and self.water_temp_idx is not None
                        and cont[self.water_temp_idx] <= -0.99
                    ):
                        cold_floor = self._cold_oat_hws_floor()
                        if cold_floor > 0.0:
                            cont[self.water_temp_idx] = np.float32(2.0 * cold_floor - 1.0)
                else:
                    if self.water_temp_idx is not None:
                        cont[self.water_temp_idx] = -1.0
                    if self.water_pressure_idx is not None:
                        cont[self.water_pressure_idx] = -1.0
                    if 0 <= self.water_discrete_idx < self.n_discrete:
                        disc[self.water_discrete_idx] = 0
            else:
                is_heating = self.water_temp_idx is not None and cont[self.water_temp_idx] > -0.99
                if not is_heating:
                    if self.water_pressure_idx is not None:
                        cont[self.water_pressure_idx] = -1.0
                    if 0 <= self.water_discrete_idx < self.n_discrete:
                        disc[self.water_discrete_idx] = 0

        if self.residual_fn is not None:
            delta = np.asarray(self.residual_fn(obs, action, self.metadata), dtype=np.float32)
            if delta.shape == cont.shape and np.all(np.isfinite(delta)):
                cont = cont + delta

        cont = np.nan_to_num(cont, nan=-1.0, posinf=1.0, neginf=-1.0)
        cont = np.clip(cont, -1.0, 1.0).astype(np.float32)
        disc = np.clip(disc, 0, 1).astype(np.int32)
        return {"continuous_action": cont, "discrete_action": disc}
