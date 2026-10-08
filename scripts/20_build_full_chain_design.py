from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from revision3.config import load_config
from revision3.full_chain import (
    assert_policy_ordering,
    build_experiment_manifest,
    expected_policy_grid,
    summarize_policy_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    parser.add_argument("--items", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    item_table = pd.read_csv(args.items)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = expected_policy_grid(item_table, config, repetitions=args.repetitions)
    summary = summarize_policy_grid(grid)
    assert_policy_ordering(summary)
    manifest = build_experiment_manifest(summary)

    grid.to_csv(output_dir / "full_chain_grid.csv", index=False)
    summary.to_csv(output_dir / "full_chain_summary.csv", index=False)
    manifest.to_csv(output_dir / "stage5_experiment_manifest.csv", index=False)

    commands = {
        "equal_size": [
            "python scripts/09_train_dpo_policy.py --policy {policy} --seed {seed} --max-train-examples 2000 --max-steps 300"
            for seed in range(42, 52)
        ],
        "endogenous_volume": [
            "python scripts/09_train_dpo_policy.py --policy {policy} --seed {seed} --max-train-examples {target} --max-steps 300"
            for seed in range(42, 52)
        ],
    }
    (output_dir / "stage5_command_templates.json").write_text(
        json.dumps(commands, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Wrote Stage 4 full-chain design to {output_dir}")


if __name__ == "__main__":
    main()
