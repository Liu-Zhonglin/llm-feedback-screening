from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small-summary", required=True)
    parser.add_argument("--small-base", required=True)
    parser.add_argument("--large-summary", required=True)
    parser.add_argument("--large-base", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def add_scale(summary_path: str, base_path: str, scale: str) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    base = float(json.loads(Path(base_path).read_text(encoding="utf-8"))["accuracy"])
    frame = frame.copy()
    frame["scale"] = scale
    frame["base_accuracy"] = base
    frame["delta_pp"] = 100.0 * (frame["preference_accuracy_mean"] - base)
    return frame


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(
        [
            add_scale(args.small_summary, args.small_base, "0.5B"),
            add_scale(args.large_summary, args.large_base, "1.5B"),
        ],
        ignore_index=True,
    )
    pivot = frame.pivot(index="policy", columns="scale", values="delta_pp")
    policy_order = ["oracle", "normal_screening", "pooling_verified", "pooling_unverified", "reverse_screening"]
    pivot = pivot.reindex(policy_order)
    pivot.to_csv(output_dir / "scale_delta_comparison.csv")

    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, pivot["0.5B"], width, label="Qwen 0.5B")
    ax.bar(x + width / 2, pivot["1.5B"], width, label="Qwen 1.5B")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, pivot.index, rotation=20, ha="right")
    ax.set_ylabel("Accuracy change vs same-scale base (pp)")
    ax.set_title("Screening effect at two model scales")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "scale_comparison.png", dpi=220)
    plt.close(fig)
    print(frame[["scale", "policy", "preference_accuracy_mean", "delta_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
