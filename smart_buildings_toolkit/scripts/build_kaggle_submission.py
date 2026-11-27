#!/usr/bin/env python3
"""Build a 100% self-contained, Internet-OFF Kaggle submission notebook.

Inlines PIDPolicy, PIDWaterResetPolicy, PeriodicHeuristicForecaster (pure NumPy),
PolicyConfig, and GeneralizedHVACPolicy into a single Kaggle notebook that subclasses
`smart_buildings_inference_server.Policy` and runs `SmartBuildingsInferenceServer`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

TOOLKIT_DIR = Path(__file__).resolve().parents[1]
if str(TOOLKIT_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_DIR))

from toolkit.paths import bootstrap  # noqa: E402

REPO_ROOT = bootstrap(TOOLKIT_DIR)

from toolkit.policy_core import PolicyConfig  # noqa: E402


def _strip_lines(src: str, prefixes: tuple[str, ...]) -> str:
    kept = []
    for line in src.splitlines():
        s = line.strip()
        if any(s.startswith(p) for p in prefixes):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _extract_pure_numpy_forecasters(src: str) -> str:
    """Keep only BaseOATForecaster, NoForecast, and PeriodicHeuristicForecaster."""
    marker = "class DiurnalAdaptiveForecaster("
    if marker in src:
        src = src.split(marker, 1)[0]
    return _strip_lines(
        src,
        ("from __future__", "import time", "from collections import", "from dataclasses import", "from typing import", "import numpy as np"),
    )


def _format_config(cfg: PolicyConfig) -> str:
    lines = [f"    {k}={repr(v)}," for k, v in asdict(cfg).items()]
    return "BEST_CONFIG = PolicyConfig(\n" + "\n".join(lines) + "\n)"


def build_submission_code(cfg: PolicyConfig, use_heuristic: bool = True) -> str:
    pid_src = _strip_lines(
        (REPO_ROOT / "smart_buildings" / "baselines" / "pid.py").read_text(encoding="utf-8"),
        ("import re", "from typing import", "import numpy as np"),
    )
    pid_plus_src = _strip_lines(
        (REPO_ROOT / "smart_buildings" / "baselines" / "pid_plus.py").read_text(encoding="utf-8"),
        (
            "from typing import",
            "import numpy as np",
            "from smart_buildings.baselines.pid import",
        ),
    )
    fc_src = _extract_pure_numpy_forecasters(
        (REPO_ROOT / "smart_buildings_toolkit" / "toolkit" / "forecasters.py").read_text(encoding="utf-8")
    )
    core_src = _strip_lines(
        (REPO_ROOT / "smart_buildings_toolkit" / "toolkit" / "policy_core.py").read_text(encoding="utf-8"),
        (
            "from __future__",
            "import json",
            "from dataclasses import",
            "from pathlib import",
            "from typing import",
            "import numpy as np",
            "from smart_buildings.baselines.pid_plus import",
            "from toolkit.forecasters import",
        ),
    )

    header = '''"""Self-contained Internet-OFF Kaggle Submission for NeurIPS 2026 Smart Buildings Competition.

Pure NumPy + Python standard library. No external network calls, no third-party ML weights,
no dataset-specific hardcoding. Subclasses `smart_buildings_inference_server.Policy`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable

import numpy as np

sys.path.append("/kaggle/input/competitions/neurips-2026-smart-buildings/kaggle_evaluation")
from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = "/kaggle/input/competitions/neurips-2026-smart-buildings/"
'''

    footer = f'''
{_format_config(cfg)}


class TransferPolicy(smart_buildings_inference_server.Policy):
    """Competition submission policy wrapping GeneralizedHVACPolicy."""

    def __init__(self, building_meta_data: Mapping[str, Any], action_spec: Any) -> None:
        super().__init__(building_meta_data, action_spec)
        fc = PeriodicHeuristicForecaster() if {bool(use_heuristic)} else NoForecast()
        self._impl = GeneralizedHVACPolicy(
            building_meta_data, action_spec, config=BEST_CONFIG, forecaster=fc
        )
        self._cont_low = np.asarray(action_spec["continuous_action"].low, dtype=np.float32)
        self._cont_high = np.asarray(action_spec["continuous_action"].high, dtype=np.float32)
        self._disc_low = np.asarray(action_spec["discrete_action"].low, dtype=np.int32)
        self._disc_high = np.asarray(action_spec["discrete_action"].high, dtype=np.int32)

    def get_action(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        action = self._impl.get_action(obs)
        cont = np.clip(
            np.nan_to_num(action["continuous_action"], nan=-1.0, posinf=1.0, neginf=-1.0),
            self._cont_low,
            self._cont_high,
        ).astype(np.float32)
        disc = np.clip(
            action["discrete_action"],
            self._disc_low,
            self._disc_high,
        ).astype(np.int32)
        return {{"continuous_action": cont, "discrete_action": disc}}


if __name__ == "__main__":
    smart_buildings_inference_server.SmartBuildingsInferenceServer(TransferPolicy).run(
        competition_data_folder=COMPETITION_DATA_FOLDER,
    )
'''

    code = "\n\n".join([header.strip(), pid_src, pid_plus_src, fc_src, core_src, footer.strip()]) + "\n"
    compile(code, "<kaggle_submission_notebook>", "exec")
    return code


def write_notebook(code: str, out_path: Path) -> None:
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "sb-sub-header",
                "metadata": {},
                "source": [
                    "# NeurIPS 2026 Smart Buildings — Self-Contained Offline Submission\n",
                    "\n",
                    "- **Internet:** OFF (100% pure NumPy + stdlib, zero external dependencies)\n",
                    "- **Policy:** `GeneralizedHVACPolicy` + `PeriodicHeuristicForecaster` (Run 4 + Thermodynamic OAT Gate)\n",
                    "- **Interface:** Subclasses `smart_buildings_inference_server.Policy` and runs `SmartBuildingsInferenceServer`\n",
                ],
            },
            {
                "cell_type": "code",
                "id": "sb-sub-policy",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build self-contained Kaggle submission notebook")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--no-heuristic", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "kaggle" / "submission_notebook.ipynb",
    )
    args = parser.parse_args()

    if args.config_json and args.config_json.exists():
        cfg = PolicyConfig.load(args.config_json)
    else:
        from smart_buildings.baselines.transfer_policy import BEST_CONFIG

        cfg = BEST_CONFIG

    code = build_submission_code(cfg, use_heuristic=not args.no_heuristic)
    write_notebook(code, args.out)
    print(f"Generated self-contained Kaggle submission notebook -> {args.out} ({len(code.splitlines())} lines)")


if __name__ == "__main__":
    main()
