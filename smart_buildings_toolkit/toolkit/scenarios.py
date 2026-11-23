"""Scenario discovery, Leave-Buildings-Out CV splits, and procedural extra tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from smart_buildings.evaluate import (
    DEFAULT_BUILDINGS_DIR,
    DEFAULT_SCENARIOS_DIR,
    DEFAULT_WEATHER_DIR,
)
from smart_buildings.scenario_paths import normalize_gin_paths
from smart_buildings.smart_control.utils.scenario_generator import (
    generate_scenario_from_config,
    load_scenario_from_parts,
)

DEFAULT_5_SCENARIOS: list[tuple[str, str]] = [
    ("building_train_1", "atlanta_2"),
    ("building_train_1", "denver_7"),
    ("building_train_1", "fairbanks_1"),
    ("building_train_1", "phoenix_5"),
    ("building_train_1", "singapore_8"),
]


def load_shipped_pairs(repo_root: Path) -> list[tuple[str, str]]:
    ts_path = repo_root / "smart_buildings" / "training_set.yaml"
    data = yaml.safe_load(ts_path.read_text()) or {}
    return [(s["building"], s["weather"]) for s in data.get("scenarios", [])]


def build_lbo_splits(repo_root: Path) -> list[tuple[str, list[Path], list[Path]]]:
    """Leave-One-Building-Out CV splits across the 6 shipped development buildings."""
    pairs = load_shipped_pairs(repo_root)
    scenarios_dir = repo_root / DEFAULT_SCENARIOS_DIR
    buildings = sorted({b for b, _ in pairs})
    splits = []
    for val_b in buildings:
        train_dirs = [scenarios_dir / b / w for b, w in pairs if b != val_b]
        val_dirs = [scenarios_dir / b / w for b, w in pairs if b == val_b]
        splits.append((val_b, train_dirs, val_dirs))
    return splits


def select_unused_weather_strips(repo_root: Path, count: int) -> list[str]:
    used = {w for _, w in load_shipped_pairs(repo_root)}
    weather_dir = repo_root / DEFAULT_WEATHER_DIR
    all_strips = sorted(p.stem for p in weather_dir.glob("*.yaml"))
    unused = [w for w in all_strips if w not in used]

    by_station: dict[str, list[str]] = {}
    for w in unused:
        by_station.setdefault(w.rsplit("_", 1)[0], []).append(w)

    picked: list[str] = []
    minn = by_station.pop("minneapolis", [])
    picked.extend(minn[: min(5, count)])

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
    """Generate `count` unseen procedural buildings x unused weather strips."""
    extra_scenarios_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.parent for p in extra_scenarios_dir.glob("*/*/config.gin"))
    for config_gin in (p / "config.gin" for p in existing):
        normalize_gin_paths(config_gin)
    if len(existing) >= count:
        return existing[:count]

    buildings_dir = repo_root / DEFAULT_BUILDINGS_DIR
    weather_dir = repo_root / DEFAULT_WEATHER_DIR

    def _find_proc() -> list[Path]:
        return sorted(
            p
            for p in buildings_dir.glob("building_*.yaml")
            if not p.stem.startswith("building_train_")
            and p.stem.split("_")[-1].isdigit()
            and int(p.stem.split("_")[-1]) >= start_seed
        )

    proc_buildings = _find_proc()
    if len(proc_buildings) < count:
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
        print(f"[Scenarios] Generating procedural buildings: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_root), check=True)
        proc_buildings = _find_proc()

    if len(proc_buildings) < count:
        raise RuntimeError(
            f"Requested {count} unseen buildings, found only {len(proc_buildings)} under {buildings_dir}."
        )

    unused_weathers = select_unused_weather_strips(repo_root, count)
    generated: list[Path] = []
    for b_yaml, w_stem in zip(proc_buildings[:count], unused_weathers):
        w_yaml = weather_dir / f"{w_stem}.yaml"
        target_dir = extra_scenarios_dir / b_yaml.stem / w_stem
        if not (target_dir / "config.gin").exists():
            sc_cfg = load_scenario_from_parts(
                str(b_yaml), str(w_yaml), output_base_dir=str(extra_scenarios_dir)
            )
            res = generate_scenario_from_config(sc_cfg)
            target_dir = Path(res["output_dir"])
            normalize_gin_paths(target_dir / "config.gin")
        generated.append(target_dir)
    return generated
