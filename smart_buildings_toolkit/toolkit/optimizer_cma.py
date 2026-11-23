"""Generalized Leave-Buildings-Out parameter optimizer (Optuna / CMA-ES / NumPy fallback)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402

# Works whether the toolkit sits at the repo root or in its own subdirectory.
REPO_ROOT = bootstrap(TOOLKIT_DIR)

from smart_buildings.evaluate import DEFAULT_SCENARIOS_DIR  # noqa: E402
from toolkit.evaluator import run_rollout  # noqa: E402
from toolkit.forecasters import NoForecast  # noqa: E402
from toolkit.policy_core import PolicyConfig  # noqa: E402
from toolkit.scenarios import DEFAULT_5_SCENARIOS  # noqa: E402

SEARCH_SPACE: list[tuple[str, float, float]] = [
    ("kp", 0.20, 1.20),
    ("ki", 0.00, 0.08),
    ("kd", 0.00, 0.30),
    ("zone_damper_full_k", 1.0, 4.0),
    ("reheat_full_k", 1.0, 4.0),
    ("reset_full_k", 1.5, 5.0),
    ("reset_effort_quantile", 0.15, 0.75),
    ("worst_zone_guard_k", 1.5, 4.8),
]


def vector_to_config(vec: np.ndarray) -> PolicyConfig:
    kwargs = {}
    for (name, lo, hi), v in zip(SEARCH_SPACE, vec):
        kwargs[name] = float(np.clip(v, lo, hi))
    return PolicyConfig(**kwargs)


def evaluate_config(config_dirs: list[Path], cfg: PolicyConfig, steps: int) -> float:
    maes = []
    for cdir in config_dirs:
        if not (cdir / "config.gin").exists():
            continue
        res, _ = run_rollout(cdir, "opt", NoForecast(), steps=steps, config=cfg)
        maes.append(res.mae_vs_zero)
    return float(np.mean(maes)) if maes else 999.0


def run_optimization(
    config_dirs: list[Path], trials: int, steps: int, seed: int = 42
) -> tuple[PolicyConfig, float]:
    try:
        import optuna  # type: ignore
    except ImportError:
        # Random search fallback only when Optuna is genuinely absent: a broken
        # study must not silently degrade into a different optimizer.
        return _random_search(config_dirs, trials, steps, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.CmaEsSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: Any) -> float:
        vec = np.array([trial.suggest_float(n, lo, hi) for n, lo, hi in SEARCH_SPACE], dtype=float)
        return evaluate_config(config_dirs, vector_to_config(vec), steps)

    study.optimize(objective, n_trials=trials)
    best_vec = np.array([study.best_params[n] for n, _, _ in SEARCH_SPACE], dtype=float)
    return vector_to_config(best_vec), float(study.best_value)


def _random_search(
    config_dirs: list[Path], trials: int, steps: int, seed: int
) -> tuple[PolicyConfig, float]:
    rng = np.random.default_rng(seed)
    best_cfg = PolicyConfig()
    best_mae = evaluate_config(config_dirs, best_cfg, steps)
    print(f"[Fallback Opt] Init MAE={best_mae:.5f}")
    for t in range(1, trials + 1):
        vec = np.array([rng.uniform(lo, hi) for _, lo, hi in SEARCH_SPACE], dtype=float)
        cfg = vector_to_config(vec)
        mae = evaluate_config(config_dirs, cfg, steps)
        if mae < best_mae:
            best_mae, best_cfg = mae, cfg
            print(f"[Fallback Opt] Trial {t}/{trials}: NEW BEST MAE={best_mae:.5f}")
    return best_cfg, best_mae


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize GeneralizedHVACPolicy parameters")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "ablation_results" / "best_policy_config.json")
    args = parser.parse_args()

    sc_dir = REPO_ROOT / DEFAULT_SCENARIOS_DIR
    config_dirs = [sc_dir / b / w for b, w in DEFAULT_5_SCENARIOS]
    best_cfg, best_mae = run_optimization(config_dirs, trials=args.trials, steps=args.steps, seed=args.seed)
    best_cfg.save(args.out)
    print(f"\nSaved best config (MAE={best_mae:.5f}) -> {args.out}")


if __name__ == "__main__":
    main()
