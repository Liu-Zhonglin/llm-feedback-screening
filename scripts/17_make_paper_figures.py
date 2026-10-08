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
    scale = pd.read_csv(ROOT / "results/scale/scale_delta_comparison.csv")
    equal = pd.read_csv(ROOT / "results/stage5/equal_size/policy_summary.csv")
    endogenous = pd.read_csv(ROOT / "results/stage5/endogenous/policy_summary.csv")
    order = ["oracle", "normal_screening", "pooling_verified", "pooling_unverified", "reverse_screening"]
    scale = scale.set_index("policy").reindex(order)
    equal = equal.set_index("policy").reindex(order)
    endogenous = endogenous.set_index("policy").reindex(order)

    series = {
        "0.5B": scale["0.5B"].to_numpy(),
        "1.5B equal-size": equal["mean_delta_pp"].to_numpy(),
        "1.5B endogenous-volume": endogenous["mean_delta_pp"].to_numpy(),
    }
    colors = ["#9ecae1", "#3182bd", "#fdae6b"]
    x = np.arange(len(order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    for i, ((label, values), color) in enumerate(zip(series.items(), colors)):
        bars = ax.bar(x + (i - 1) * width, values, width, label=label, color=color, edgecolor="white", linewidth=0.5)
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            delta = 0.035 if value >= 0 else -0.035
            ax.text(bar.get_x() + bar.get_width() / 2, value + delta, f"{value:+.2f}", ha="center", va=va, fontsize=6.2)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel(r"$\Delta$ preference accuracy (pp)", fontsize=9)
    ax.set_xticks(x, [POLICY_LABELS[name] for name in order], fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(-0.9, 2.65)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=7.4, ncol=3, loc="upper right", bbox_to_anchor=(1.0, 1.04))
    fig.tight_layout(pad=0.35)
    fig.savefig(output_dir / "fig_scale_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
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
