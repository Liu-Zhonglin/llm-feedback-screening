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


POLICY_LABELS = {
    "oracle": "Oracle",
    "normal_screening": "Normal screening",
    "reverse_screening": "Reverse screening",
    "pooling_verified": "Verified pooling",
    "pooling_unverified": "Unverified pooling",
}
NOISE_ORDER = ["perfect", "standard_tpr0.8_fpr0.1", "noisy", "uninformative"]
NOISE_LABELS = ["Perfect", "Standard", "Noisy", "Uninformative"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def make_type_figure(output_dir: Path) -> None:
    contributor = pd.read_csv(ROOT / "results/types/contributor_types.csv")
    summary = json.loads((ROOT / "results/types/summary.json").read_text(encoding="utf-8"))
    contributor = contributor.dropna(subset=["type_name"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    bins = np.linspace(0.0, 1.0, 16)
    for type_name, color in (("L", "#d95f02"), ("H", "#1b9e77")):
        values = contributor.loc[contributor["type_name"] == type_name, "loo_rate"]
        axes[0].hist(values, bins=bins, alpha=0.65, label=f"{type_name}-type", color=color)
    axes[0].set_xlabel("Leave-one-out helpfulness rate")
    axes[0].set_ylabel("Contributors")
    axes[0].set_title("(a) Contributor reliability")
    axes[0].legend(frameon=False)
    means = [summary["eta_L"], summary["eta_H"]]
    lows = [summary["eta_L_ci"], summary["eta_H_ci"]]
    highs = [summary["eta_L_hi"], summary["eta_H_hi"]]
    errors = np.array([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    axes[1].bar(["L-type", "H-type"], means, yerr=errors, capsize=5, color=["#d95f02", "#1b9e77"])
    axes[1].set_ylim(0.0, 0.7)
    axes[1].set_ylabel("Pr(helpful feedback)")
    axes[1].set_title("(b) Type-specific helpfulness")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_type_calibration.png", dpi=300)
    plt.close(fig)


def make_scale_figure(output_dir: Path) -> None:
    frame = pd.read_csv(ROOT / "results/scale/scale_delta_comparison.csv")
    order = ["oracle", "normal_screening", "pooling_verified", "pooling_unverified", "reverse_screening"]
    frame = frame.set_index("policy").reindex(order)
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.bar(x - width / 2, frame["0.5B"], width, label="Qwen 0.5B (10 seeds)", color="#4c78a8")
    ax.bar(x + width / 2, frame["1.5B"], width, label="Qwen 1.5B (10 seeds)", color="#f58518")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, [POLICY_LABELS[name] for name in order], rotation=18, ha="right")
    ax.set_ylabel("Accuracy change vs base (pp)")
    ax.set_title("Screening effect across model scales")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_scale_comparison.png", dpi=300)
    plt.close(fig)


def make_noise_figure(output_dir: Path) -> None:
    frame = pd.read_csv(ROOT / "results/noise/hpc_noise_noise_summary.csv")
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for policy, label, color in (
        ("pooling_verified", "Verified pooling", "#d95f02"),
        ("normal_screening", "Normal screening", "#1b9e77"),
        ("reverse_screening", "Reverse screening", "#7570b3"),
    ):
        group = frame.loc[frame["policy"] == policy].set_index("noise").reindex(NOISE_ORDER)
        ax.errorbar(NOISE_LABELS, group["accuracy"], yerr=group["sd"], marker="o", capsize=4, linewidth=2, label=label, color=color)
    ax.set_ylabel("Held-out preference accuracy")
    ax.set_ylim(0.68, 0.72)
    ax.set_title("Robustness to verification noise")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_noise_robustness.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_type_figure(output_dir)
    make_scale_figure(output_dir)
    make_noise_figure(output_dir)


if __name__ == "__main__":
    main()
