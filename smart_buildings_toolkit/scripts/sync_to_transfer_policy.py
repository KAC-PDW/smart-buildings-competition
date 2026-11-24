#!/usr/bin/env python3
"""Generate a clean transfer_policy.py from a validated PolicyConfig JSON."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402

# Works whether the toolkit sits at the repo root or in its own subdirectory.
REPO_ROOT = bootstrap(TOOLKIT_DIR)

from toolkit.policy_core import PolicyConfig  # noqa: E402

TEMPLATE = '''"""Auto-generated TransferPolicy from validated PolicyConfig."""

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
{config_fields}
)


class TransferPolicy:
    name = "transfer"

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any) -> None:
        fc = PeriodicHeuristicForecaster() if {use_heuristic} else NoForecast()
        self._impl = GeneralizedHVACPolicy(
            building_meta_data, action_spec, config=BEST_CONFIG, forecaster=fc
        )

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        return self._impl.get_action(obs)
'''


def _format_config_fields(cfg: PolicyConfig) -> str:
    lines = []
    for k, v in asdict(cfg).items():
        lines.append(f"    {k}={repr(v)},")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write transfer_policy.py from PolicyConfig JSON")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--use-heuristic", action="store_true")
    parser.add_argument(
        "--target",
        type=Path,
        default=REPO_ROOT / "smart_buildings" / "baselines" / "transfer_policy.py",
    )
    args = parser.parse_args()

    cfg = PolicyConfig.load(args.config_json) if args.config_json and args.config_json.exists() else PolicyConfig()
    code = TEMPLATE.format(
        config_fields=_format_config_fields(cfg),
        use_heuristic=bool(args.use_heuristic),
    )
    compile(code, str(args.target), "exec")
    args.target.parent.mkdir(parents=True, exist_ok=True)
    args.target.write_text(code)
    print(f"Updated {args.target} (use_heuristic={args.use_heuristic})")


if __name__ == "__main__":
    main()
