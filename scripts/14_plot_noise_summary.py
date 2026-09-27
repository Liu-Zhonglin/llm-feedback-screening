from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


NOISE_ORDER = ["perfect", "standard_tpr0.8_fpr0.1", "noisy", "uninformative"]
NOISE_LABELS = ["Perfect", "Standard", "Noisy", "Uninformative"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.summary)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    noise_policies = {"normal_screening", "pooling_verified", "reverse_screening"}
    for policy, group in frame.groupby("policy"):
        if policy not in noise_policies:
            continue
        group = group.set_index("noise").reindex(NOISE_ORDER).reset_index()
        ax.errorbar(
            NOISE_LABELS,
            group["accuracy"],
            yerr=group["sd"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=policy,
        )
    oracle_rows = frame.loc[frame["policy"] == "oracle", "accuracy"]
    if not oracle_rows.empty:
        ax.axhline(float(oracle_rows.mean()), color="black", linestyle="--", linewidth=1.5, label="oracle")
    ax.set_ylabel("Held-out preference accuracy")
    ax.set_ylim(0.67, 0.73)
    ax.set_title("Verification-noise robustness (5 seeds)")
    ax.legend()
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
