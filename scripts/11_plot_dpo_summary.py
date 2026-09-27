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
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--base-eval", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.summary).sort_values("preference_accuracy_mean", ascending=False)
    base = json.loads(Path(args.base_eval).read_text(encoding="utf-8"))["accuracy"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(frame))
    ax.bar(
        x,
        frame["preference_accuracy_mean"],
        yerr=frame["preference_accuracy_std"].fillna(0.0),
        capsize=5,
        color="#4c78a8",
    )
    ax.axhline(base, color="#c92a2a", linestyle="--", linewidth=2, label="Base model")
    ax.set_xticks(list(x), frame["policy"], rotation=25, ha="right")
    ax.set_ylabel("Held-out preference accuracy")
    ax.set_ylim(0.65, 0.75)
    n_seeds = int(frame["seeds"].max())
    ax.set_title(f"HPC DPO screening pilot ({n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    comparison = frame.copy()
    comparison["accuracy_delta_pp_vs_base"] = 100.0 * (
        comparison["preference_accuracy_mean"] - base
    )
    comparison.sort_values("accuracy_delta_pp_vs_base", ascending=False).to_csv(
        output.with_name("comparison.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
