"""Pluggable hourly Outside Air Temperature (OAT) forecasters."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

STEPS_PER_HOUR = 12  # 5-minute timestep


@dataclass
class ForecastResult:
    """Multi-hour ahead OAT forecast in Kelvin."""

    p50_k: np.ndarray
    p10_k: np.ndarray
    p90_k: np.ndarray


class BaseOATForecaster:
    """Base interface for hourly OAT forecasters."""

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
        arr = np.asarray(self._oat_history_5m, dtype=np.float32)
        n_full = len(arr) // STEPS_PER_HOUR
        if n_full == 0:
            return arr[-1:] if len(arr) else np.array([], dtype=np.float32)
        return arr[-n_full * STEPS_PER_HOUR :].reshape(n_full, STEPS_PER_HOUR).mean(axis=1)

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        raise NotImplementedError


class NoForecast(BaseOATForecaster):
    """Arm A: No forecast (pure reactive control)."""

    arm_name = "Arm_A_Baseline"

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        return None


class PeriodicHeuristicForecaster(BaseOATForecaster):
    """Arm B: 24h-lag + damped 1h linear trend forecaster (pure NumPy)."""

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
            p50 = yesterday_future + (curr - yesterday_now) * np.exp(-0.15 * h)
        else:
            p50 = curr + slope * (1.0 - np.exp(-0.35 * h)) / 0.35

        spread = 0.8 + 0.35 * h
        return ForecastResult(p50_k=p50, p10_k=p50 - spread, p90_k=p50 + spread)


class DiurnalAdaptiveForecaster(BaseOATForecaster):
    """Arm E: Multi-day EWMA diurnal profile + bias correction + empirical quantile spread."""

    arm_name = "Arm_E_DiurnalAdaptive"

    def __init__(self, horizon_hours: int = 6, bias_alpha: float = 0.35, day_decay: float = 0.60) -> None:
        super().__init__(horizon_hours=horizon_hours)
        self.bias_alpha = float(bias_alpha)
        self.day_decay = float(day_decay)
        self._pending_1h_pred: float | None = None
        self._ewma_bias: float = 0.0
        self._err_hist: deque[float] = deque(maxlen=48)

    def reset(self) -> None:
        super().reset()
        self._pending_1h_pred = None
        self._ewma_bias = 0.0
        self._err_hist.clear()

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        hourly = self._hourly_series()
        if len(hourly) < 2:
            return None
        curr = float(hourly[-1])
        if self._pending_1h_pred is not None:
            err = float(np.clip(curr - self._pending_1h_pred, -10.0, 10.0))
            self._ewma_bias = self.bias_alpha * err + (1.0 - self.bias_alpha) * self._ewma_bias
            self._err_hist.append(err)

        slope = float(np.clip(hourly[-1] - hourly[-2], -3.0, 3.0))
        h = np.arange(1, self.horizon_hours + 1, dtype=np.float32)

        n = len(hourly)
        if n >= 24 + self.horizon_hours:
            w_sum = 0.0
            diurnal_now = 0.0
            diurnal_future = np.zeros(self.horizon_hours, dtype=np.float32)
            d = 0
            while 25 + 24 * d <= n:
                w = self.day_decay ** d
                now_idx = n - 25 - 24 * d
                fut_start = n - 24 * (d + 1)
                fut_end = fut_start + self.horizon_hours
                diurnal_now += w * float(hourly[now_idx])
                diurnal_future += w * hourly[fut_start:fut_end]
                w_sum += w
                d += 1
            diurnal_now /= w_sum
            diurnal_future /= w_sum
            p50 = (
                diurnal_future
                + (curr - diurnal_now) * np.exp(-0.12 * h)
                + 0.35 * self._ewma_bias * np.exp(-0.20 * h)
                + 0.35 * slope * (1.0 - np.exp(-0.40 * h)) * np.exp(-0.25 * h)
            )
        else:
            p50 = (
                curr
                + slope * (1.0 - np.exp(-0.35 * h)) / 0.35
                + 0.35 * self._ewma_bias * np.exp(-0.25 * h)
            )

        if len(self._err_hist) >= 6:
            emp_std = float(np.std(np.asarray(self._err_hist, dtype=np.float32)))
            spread = np.maximum(0.6, 1.2816 * emp_std * np.sqrt(h)) + 0.15 * h
        else:
            spread = 0.8 + 0.35 * h

        p50 = p50.astype(np.float32)
        spread = spread.astype(np.float32)
        self._pending_1h_pred = float(p50[0])
        return ForecastResult(p50_k=p50, p10_k=p50 - spread, p90_k=p50 + spread)


class TimesFM25Forecaster(BaseOATForecaster):
    """Arm C: Zero-shot TimesFM 2.5 (200M) hourly forecaster."""

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
        self._model: Any = None
        self._backend: str | None = None
        self._load_model()

    def _load_model(self) -> None:
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
                "Install `timesfm` or `transformers` + `torch`, or omit Arm C."
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
                horizon=self.horizon_hours, inputs=[context]
            )
            p50 = np.asarray(point_fc[0][: self.horizon_hours], dtype=np.float32)
            q = np.asarray(quantile_fc[0][: self.horizon_hours], dtype=np.float32)
            p10 = q[:, 1] if q.ndim == 2 and q.shape[1] >= 10 else p50 - 1.5
            p90 = q[:, 9] if q.ndim == 2 and q.shape[1] >= 10 else p50 + 1.5
            return ForecastResult(p50_k=p50, p10_k=p10, p90_k=p90)

        import torch

        with torch.no_grad():
            inp = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self._model(past_values=inp)
            p50 = out.mean_predictions[0, : self.horizon_hours].cpu().numpy().astype(np.float32)
            if hasattr(out, "full_predictions") and out.full_predictions is not None:
                q = out.full_predictions[0, : self.horizon_hours].cpu().numpy().astype(np.float32)
                p10 = q[:, 1] if q.ndim == 2 and q.shape[1] >= 10 else p50 - 1.5
                p90 = q[:, 9] if q.ndim == 2 and q.shape[1] >= 10 else p50 + 1.5
            else:
                p10, p90 = p50 - 1.5, p50 + 1.5
            return ForecastResult(p50_k=p50, p10_k=p10, p90_k=p90)


class OracleForecaster(BaseOATForecaster):
    """Arm D: Ground-truth future OAT series (local dev upper bound only)."""

    arm_name = "Arm_D_Oracle"

    def __init__(self, future_oat_5m: np.ndarray, horizon_hours: int = 6) -> None:
        super().__init__(horizon_hours=horizon_hours)
        self._future_oat_5m = np.asarray(future_oat_5m, dtype=np.float32)

    def _predict_hourly(self, step_idx: int) -> ForecastResult | None:
        n = len(self._future_oat_5m)
        vals = []
        for h in range(1, self.horizon_hours + 1):
            s = min(step_idx + (h - 1) * STEPS_PER_HOUR, n - 1)
            e = min(step_idx + h * STEPS_PER_HOUR, n)
            vals.append(
                float(np.mean(self._future_oat_5m[s:e])) if e > s else float(self._future_oat_5m[-1])
            )
        p50 = np.asarray(vals, dtype=np.float32)
        return ForecastResult(p50_k=p50, p10_k=p50, p90_k=p50)


def build_forecaster(
    arm: str,
    *,
    horizon_hours: int = 6,
    timesfm_model: str = "google/timesfm-2.5-200m-pytorch",
    device: str = "cpu",
    oracle_trace: np.ndarray | None = None,
) -> BaseOATForecaster:
    arm = arm.upper()
    if arm == "A":
        return NoForecast(horizon_hours=horizon_hours)
    if arm == "B":
        return PeriodicHeuristicForecaster(horizon_hours=horizon_hours)
    if arm == "C":
        return TimesFM25Forecaster(
            horizon_hours=horizon_hours, model_id=timesfm_model, device=device
        )
    if arm == "D":
        if oracle_trace is None:
            raise ValueError("Arm D (Oracle) requires `oracle_trace`.")
        return OracleForecaster(oracle_trace, horizon_hours=horizon_hours)
    if arm == "E":
        return DiurnalAdaptiveForecaster(horizon_hours=horizon_hours)
    raise ValueError(f"Unknown arm: {arm}")
