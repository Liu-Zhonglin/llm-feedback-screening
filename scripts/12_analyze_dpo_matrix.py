from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def t_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    half_width = stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1) * stats.sem(values)
    return float(values.mean() - half_width), float(values.mean() + half_width)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.summary).sort_values(["policy", "seed"])
    base = float(json.loads(Path(args.base_eval).read_text(encoding="utf-8"))["accuracy"])

    policy_rows: list[dict[str, Any]] = []
    for policy, group in frame.groupby("policy", sort=False):
        accuracies = group["preference_accuracy"].to_numpy(dtype=float)
        delta = accuracies - base
        low, high = t_interval(delta)
        policy_rows.append(
            {
                "policy": policy,
                "n_seeds": int(len(group)),
                "mean_accuracy": float(accuracies.mean()),
                "sd_accuracy": float(accuracies.std(ddof=1)) if len(group) > 1 else float("nan"),
                "mean_delta_pp": float(delta.mean() * 100.0),
                "sd_delta_pp": float(delta.std(ddof=1) * 100.0) if len(group) > 1 else float("nan"),
                "delta_ci_low_pp": float(low * 100.0),
                "delta_ci_high_pp": float(high * 100.0),
                "mean_helpful_share": float(group["helpful_share"].mean()),
                "mean_weight": float(group["mean_weight"].mean()),
            }
        )
    policy_summary = pd.DataFrame(policy_rows).sort_values("mean_accuracy", ascending=False)
    policy_summary.to_csv(output_dir / "policy_summary.csv", index=False)

    comparisons: list[dict[str, Any]] = []
    pivot = frame.pivot(index="seed", columns="policy", values="preference_accuracy")
    for policy_a in pivot.columns:
        for policy_b in pivot.columns:
            if policy_a <= policy_b:
                continue
            paired = (pivot[policy_a] - pivot[policy_b]).dropna().to_numpy(dtype=float)
            low, high = t_interval(paired)
            comparisons.append(
                {
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "n_paired_seeds": int(len(paired)),
                    "mean_difference_pp": float(paired.mean() * 100.0),
                    "sd_difference_pp": float(paired.std(ddof=1) * 100.0) if len(paired) > 1 else float("nan"),
                    "ci_low_pp": float(low * 100.0),
                    "ci_high_pp": float(high * 100.0),
                    "fraction_a_gt_b": float(np.mean(paired > 0.0)),
                }
            )
    comparison = pd.DataFrame(comparisons)
    priority_pairs = {
        ("oracle", "normal_screening"),
        ("oracle", "reverse_screening"),
        ("normal_screening", "reverse_screening"),
        ("pooling_verified", "pooling_unverified"),
    }
    comparison["priority"] = comparison.apply(
        lambda row: (row["policy_a"], row["policy_b"]) in priority_pairs
        or (row["policy_b"], row["policy_a"]) in priority_pairs,
        axis=1,
    )
    comparison.sort_values(["priority", "mean_difference_pp"], ascending=[False, False]).to_csv(
        output_dir / "paired_comparisons.csv",
        index=False,
    )
    report = {
        "base_accuracy": base,
        "policies": policy_summary.to_dict(orient="records"),
        "priority_comparisons": comparison.loc[comparison["priority"]].to_dict(orient="records"),
    }
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(policy_summary.to_string(index=False))
    print()
    print(comparison.loc[comparison["priority"]].to_string(index=False))


if __name__ == "__main__":
    main()
