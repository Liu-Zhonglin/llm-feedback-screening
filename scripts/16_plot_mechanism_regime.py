from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

from revision3.mechanism import (  # noqa: E402
    boundary_phi_H,
    expected_accepted_metrics,
    screening_regime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=str(ROOT / "results" / "types" / "summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "mechanism"),
    )
    return parser.parse_args()


def participation_from_thresholds(
    *,
    eta_H: float,
    eta_L: float,
    phi_H: float,
    phi_L: float,
    p_rho: float,
    participation_cost: float,
) -> tuple[float, float]:
    threshold_H = participation_cost + phi_H * p_rho * (1.0 - eta_H)
    threshold_L = participation_cost + phi_L * p_rho * (1.0 - eta_L)
    regime = screening_regime(
        eta_H=eta_H,
        eta_L=eta_L,
        phi_H=phi_H,
        phi_L=phi_L,
    )
    if regime == "normal":
        return 1.0, 0.0
    if regime == "reverse":
        return 0.0, 1.0
    # At the knife-edge, strict separation is infeasible. Use pooling as the
    # transparent boundary convention for visualization.
    return 1.0, 1.0


def metrics_for_exposure(
    *,
    eta_H: float,
    eta_L: float,
    lambda_high: float,
    phi_H: float,
    phi_L: float,
    p_rho: float,
    participation_cost: float,
    rho: float,
    verifier_tpr: float,
    verifier_fpr: float,
    unverified_harmful_weight: float,
) -> dict[str, float | str]:
    high_participation, low_participation = participation_from_thresholds(
        eta_H=eta_H,
        eta_L=eta_L,
        phi_H=phi_H,
        phi_L=phi_L,
        p_rho=p_rho,
        participation_cost=participation_cost,
    )
    omega, accepted_helpfulness = expected_accepted_metrics(
        eta_H=eta_H,
        eta_L=eta_L,
        lambda_high=lambda_high,
        high_participation=high_participation,
        low_participation=low_participation,
        rho=rho,
        verifier_tpr=verifier_tpr,
        verifier_fpr=verifier_fpr,
        unverified_harmful_weight=unverified_harmful_weight,
    )
    return {
        "phi_H": phi_H,
        "phi_L": phi_L,
        "p_rho": p_rho,
        "regime": screening_regime(
            eta_H=eta_H,
            eta_L=eta_L,
            phi_H=phi_H,
            phi_L=phi_L,
        ),
        "high_participation": high_participation,
        "low_participation": low_participation,
        "accepted_high_share": omega,
        "accepted_helpfulness": accepted_helpfulness,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    eta_H = float(summary["eta_H"])
    eta_L = float(summary["eta_L"])
    lambda_high = 0.5
    participation_cost = 0.2
    rho = 0.3
    verifier_tpr = 0.8
    verifier_fpr = 0.1
    unverified_harmful_weight = 0.25

    phi_L = 0.8
    phi_H_star = boundary_phi_H(eta_H=eta_H, eta_L=eta_L, phi_L=phi_L)
    p_rho_main = 0.075
    phi_H_grid = np.linspace(0.05, 1.6, 200)

    crossing = pd.DataFrame(
        [
            metrics_for_exposure(
                eta_H=eta_H,
                eta_L=eta_L,
                lambda_high=lambda_high,
                phi_H=float(phi_H),
                phi_L=phi_L,
                p_rho=p_rho_main,
                participation_cost=participation_cost,
                rho=rho,
                verifier_tpr=verifier_tpr,
                verifier_fpr=verifier_fpr,
                unverified_harmful_weight=unverified_harmful_weight,
            )
            for phi_H in phi_H_grid
        ]
    )
    crossing.to_csv(output_dir / "exposure_crossing.csv", index=False)

    # Regime map over the two reduced-form exposure primitives.
    phi_grid = np.linspace(0.0, 1.6, 400)
    PH, PL = np.meshgrid(phi_grid, phi_grid)
    exposure_H = PH * (1.0 - eta_H)
    exposure_L = PL * (1.0 - eta_L)
    normal_region = exposure_H < exposure_L
    regime_code = np.where(normal_region, 1.0, 0.0)
    boundary_line = (1.0 - eta_H) / (1.0 - eta_L) * phi_grid
    regime_map = pd.DataFrame(
        {
            "phi_H": PH.ravel(),
            "phi_L": PL.ravel(),
            "exposure_H": exposure_H.ravel(),
            "exposure_L": exposure_L.ravel(),
            "regime": np.where(normal_region.ravel(), "normal", "reverse"),
        }
    )
    regime_map.to_csv(output_dir / "regime_map.csv", index=False)

    # Policy-intensity variation under a fixed reward slack.
    p_rho_grid = np.linspace(0.005, 0.30, 120)
    reward_slack = 0.08
    scenario_specs = {
        "normal": {"phi_H": 0.30, "phi_L": 0.80},
        "reverse": {"phi_H": 0.90, "phi_L": 0.10},
    }
    intensity_rows: list[dict[str, float | str]] = []
    for scenario, spec in scenario_specs.items():
        for p_rho in p_rho_grid:
            high_participation = float(
                reward_slack >= spec["phi_H"] * p_rho * (1.0 - eta_H)
            )
            low_participation = float(
                reward_slack >= spec["phi_L"] * p_rho * (1.0 - eta_L)
            )
            omega, accepted_helpfulness = expected_accepted_metrics(
                eta_H=eta_H,
                eta_L=eta_L,
                lambda_high=lambda_high,
                high_participation=high_participation,
                low_participation=low_participation,
                rho=rho,
                verifier_tpr=verifier_tpr,
                verifier_fpr=verifier_fpr,
                unverified_harmful_weight=unverified_harmful_weight,
            )
            intensity_rows.append(
                {
                    "scenario": scenario,
                    "phi_H": spec["phi_H"],
                    "phi_L": spec["phi_L"],
                    "p_rho": float(p_rho),
                    "high_participation": high_participation,
                    "low_participation": low_participation,
                    "accepted_high_share": omega,
                    "accepted_helpfulness": accepted_helpfulness,
                }
            )
    intensity = pd.DataFrame(intensity_rows)
    intensity.to_csv(output_dir / "intensity_scenarios.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))

    # (a) Analytical regime boundary.
    ax = axes[0, 0]
    ax.imshow(
        regime_code,
        origin="lower",
        extent=[phi_grid[0], phi_grid[-1], phi_grid[0], phi_grid[-1]],
        aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["#f3c4c4", "#cfe8d4"]),
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot(
        phi_grid,
        boundary_line,
        color="#222222",
        linewidth=1.4,
        label=r"$\phi_H(1-\eta_H)=\phi_L(1-\eta_L)$",
    )
    ax.scatter([0.30], [0.80], color="#14532d", marker="o", s=28, zorder=3)
    ax.annotate("normal point", (0.30, 0.80), xytext=(0.42, 1.08), fontsize=8)
    ax.scatter([0.90], [0.10], color="#7f1d1d", marker="o", s=28, zorder=3)
    ax.annotate("reverse point", (0.90, 0.10), xytext=(0.72, 0.25), fontsize=8)
    ax.text(0.30, 1.35, "normal\nscreening", ha="center", va="center", fontsize=8)
    ax.text(1.25, 0.35, "reverse\nscreening", ha="center", va="center", fontsize=8)
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel(r"High-type exposure $\phi_H$")
    ax.set_ylabel(r"Low-type exposure $\phi_L$")
    ax.text(
        0.52,
        0.47,
        "boundary",
        rotation=32,
        fontsize=8,
        color="#222222",
    )
    ax.set_title("(a) Regime boundary under calibrated types")
    ax.legend(loc="upper left", frameon=False)

    # (b) Participation as high-type exposure crosses the boundary.
    ax = axes[0, 1]
    ax.plot(
        crossing["phi_H"],
        crossing["high_participation"],
        color="#1f6f8b",
        linewidth=1.6,
        label="High type",
    )
    ax.plot(
        crossing["phi_H"],
        crossing["low_participation"],
        color="#c46a1d",
        linewidth=1.6,
        linestyle="--",
        label="Low type",
    )
    ax.axvline(phi_H_star, color="#333333", linewidth=1.0, linestyle=":")
    ax.text(
        phi_H_star + 0.03,
        0.53,
        r"boundary $\phi_H^*$",
        rotation=90,
        va="center",
        fontsize=8,
    )
    ax.set_xlim(float(phi_H_grid[0]), float(phi_H_grid[-1]))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"High-type exposure $\phi_H$")
    ax.set_ylabel("Participation probability")
    ax.set_title(r"(b) Screening direction at $P\rho=0.075$")
    ax.legend(loc="center right", frameon=False)

    # (c) Policy intensity under fixed reward slack.
    ax = axes[1, 0]
    for scenario, color in [("normal", "#14532d"), ("reverse", "#7f1d1d")]:
        frame = intensity.loc[intensity["scenario"] == scenario]
        ax.plot(
            frame["p_rho"],
            frame["high_participation"],
            color=color,
            linewidth=1.5,
            linestyle="-",
            label=f"{scenario}: H",
        )
        ax.plot(
            frame["p_rho"],
            frame["low_participation"],
            color=color,
            linewidth=1.5,
            linestyle="--",
            label=f"{scenario}: L",
        )
    ax.set_xlim(0.0, 0.30)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Screening intensity $P\rho$")
    ax.set_ylabel("Participation probability")
    ax.set_title("(c) Policy intensity with fixed reward slack")
    ax.annotate(
        "pooling",
        xy=(0.015, 1.0),
        xytext=(0.035, 0.88),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#333333"},
        fontsize=8,
        color="#333333",
    )
    ax.legend(ncol=2, loc="lower left", frameon=False)

    # (d) Accepted composition and helpfulness as exposure crosses the boundary.
    ax = axes[1, 1]
    ax.plot(
        crossing["phi_H"],
        crossing["accepted_high_share"],
        color="#1f6f8b",
        linewidth=1.7,
        label=r"Accepted high-type share $\omega$",
    )
    ax.plot(
        crossing["phi_H"],
        crossing["accepted_helpfulness"],
        color="#6b4c9a",
        linewidth=1.7,
        linestyle="--",
        label="Accepted helpfulness",
    )
    ax.axvline(phi_H_star, color="#333333", linewidth=1.0, linestyle=":")
    ax.set_xlim(float(phi_H_grid[0]), float(phi_H_grid[-1]))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"High-type exposure $\phi_H$")
    ax.set_ylabel("Accepted feedback composition")
    ax.set_title(r"(d) Composition at $P\rho=0.075$")
    ax.legend(loc="center right", frameon=False)

    fig.tight_layout()
    figure_path = output_dir / "fig_mechanism_regime.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "eta_H": eta_H,
        "eta_L": eta_L,
        "lambda_high": lambda_high,
        "phi_L_fixed": phi_L,
        "boundary_phi_H": phi_H_star,
        "p_rho_main": p_rho_main,
        "rho": rho,
        "verifier_tpr": verifier_tpr,
        "verifier_fpr": verifier_fpr,
        "unverified_harmful_weight": unverified_harmful_weight,
        "scenario_specs": scenario_specs,
        "reward_slack_for_intensity_panel": reward_slack,
    }
    (output_dir / "mechanism_regime_summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote mechanism regime figure and grids to {output_dir}")


if __name__ == "__main__":
    main()
