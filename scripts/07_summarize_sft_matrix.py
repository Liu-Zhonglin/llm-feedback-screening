from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        train_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            continue
        eval_metrics = json.loads(eval_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "policy": train_metrics["policy"],
                "seed": int(train_metrics["seed"]),
                "harm_mode": train_metrics.get("harm_mode"),
                "helpful_share": train_metrics.get("helpful_share"),
                "mean_weight": train_metrics.get("mean_weight"),
                "response_nll": eval_metrics.get("response_nll", float("nan")),
                "preference_accuracy": eval_metrics.get("accuracy", eval_metrics.get("preference_accuracy")),
                "mean_margin": eval_metrics.get("mean_margin", float("nan")),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["policy", "seed"])
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_suffix(".csv"), index=False)
    summary = (
        frame.groupby("policy", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            helpful_share_mean=("helpful_share", "mean"),
            response_nll_mean=("response_nll", "mean"),
            response_nll_std=("response_nll", "std"),
            preference_accuracy_mean=("preference_accuracy", "mean"),
            preference_accuracy_std=("preference_accuracy", "std"),
        )
        .sort_values("preference_accuracy_mean", ascending=False)
    )
    summary.to_csv(output_prefix.with_name(output_prefix.stem + "_summary.csv"), index=False)
    print(frame.to_string(index=False))
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
