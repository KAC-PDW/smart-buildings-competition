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

    def _ahu_error(self, obs: np.ndarray, ctrl: dict[str, Any], low: float, high: float) -> float:
        worst = self._worst_zone_error(obs, ctrl, low, high)
        if abs(worst) >= self.cfg.worst_zone_guard_k:
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
        if not (comfort_soon or (0 < hours_to_start <= self.cfg.pre_window_hours)):
            return low, high

        comfort_lo, comfort_hi = self.comfort_temp_window
        adj_low, adj_high = low, high
        if float(np.max(forecast.p90_k)) > comfort_hi + self.cfg.precool_trigger_delta_k:
            adj_high = min(adj_high, comfort_hi - self.cfg.precool_margin_k)
        if float(np.min(forecast.p10_k)) < comfort_lo - self.cfg.preheat_trigger_delta_k:
            adj_low = max(adj_low, comfort_lo + self.cfg.preheat_margin_k)
        return adj_low, adj_high

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
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

        cont = np.asarray(action["continuous_action"], dtype=np.float32)
        disc = np.asarray(action["discrete_action"], dtype=np.int32)

        if self.cfg.fix_hws_cooling_bug:
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
