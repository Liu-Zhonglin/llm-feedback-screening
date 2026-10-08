from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    row = manifest.loc[
        (manifest["policy"] == args.policy) & (manifest["seed"] == args.seed)
    ]
    if len(row) != 1:
        raise SystemExit(f"Expected one manifest row, found {len(row)}")
    row = row.iloc[0]
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    checks = {
        "accepted_pool_size": int(row["accepted_pool_size"]),
        "processed_pair_budget": int(row["processed_pair_budget"]),
        "optimizer_steps": int(row["optimizer_steps"]),
        "effective_batch_size": int(row["effective_batch_size"]),
        "distinct_pair_count": int(row["distinct_pair_count"]),
        "distinct_tree_count": int(row["distinct_tree_count"]),
    }
    errors = []
    for key, expected in checks.items():
        observed = metrics.get(key)
        if observed != expected:
            errors.append(f"{key}: expected {expected}, observed {observed}")

    processed_examples = metrics.get("processed_examples")
    if processed_examples != int(row["processed_pair_budget"]):
        errors.append(
            f"processed_examples: expected {int(row['processed_pair_budget'])}, observed {processed_examples}"
        )

    if errors:
        raise SystemExit("Volume-budget validation failed:\n" + "\n".join(errors))
    print(f"Validated {args.policy} seed {args.seed}: {int(row['processed_pair_budget'])} processed pairs")


if __name__ == "__main__":
    main()
