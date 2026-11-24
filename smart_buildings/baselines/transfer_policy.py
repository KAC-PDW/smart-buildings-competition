"""Auto-generated TransferPolicy from validated PolicyConfig."""

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import numpy as np

_TOOLKIT_DIR = Path(__file__).resolve().parents[2] / "smart_buildings_toolkit"
if str(_TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLKIT_DIR))

from toolkit.forecasters import NoForecast, PeriodicHeuristicForecaster
from toolkit.policy_core import GeneralizedHVACPolicy, PolicyConfig

BEST_CONFIG = PolicyConfig(
    kp=0.5,
    ki=0.02,
    kd=0.1,
    zone_damper_full_k=2.0,
    reheat_full_k=1.5,
    reset_full_k=3.0,
    reset_effort_quantile=0.5,
    worst_zone_guard_k=3.0,
    use_sensor_band=True,
    fix_hws_cooling_bug=True,
    precool_trigger_delta_k=6.0,
    precool_margin_k=1.0,
    preheat_trigger_delta_k=15.0,
    preheat_margin_k=1.0,
    pre_window_hours=3,
    fix_sensor_band_precond=True,
    fix_hws_pump_demand_check=True,
    worst_zone_guard_heat_k=999.0,
    enable_cold_hws_boost=True,
    cold_oat_hws_floor_start_k=273.15,
    cold_oat_hws_floor_full_k=253.15,
    use_continuous_precond=True,
    precool_ramp_start_delta_k=-1.0,
    preheat_ramp_start_delta_k=5.0,
    adaptive_eco_cap=True,
    eco_cap_hot_oat_margin_k=2.0,
    eco_cap_cold_oat_k=263.15,
)


class TransferPolicy:
    name = "transfer"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any) -> None:
        fc = PeriodicHeuristicForecaster() if True else NoForecast()
        self._impl = GeneralizedHVACPolicy(
            building_meta_data, action_spec, config=BEST_CONFIG, forecaster=fc
        )

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return self._impl.get_action(obs)
