from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from revision3.mechanism import Policy, build_policies, simulate_policy


POLICY_ORDER = [
    "oracle",
    "normal_screening",
    "reverse_screening",
    "pooling_verified",
    "pooling_unverified",
]


def _policy_by_name(policies: list[Policy], name: str) -> Policy:
    for policy in policies:
        if policy.name == name:
            return policy
    raise KeyError(name)


def canonical_policies(config: dict[str, Any]) -> dict[str, tuple[Policy, dict[str, Any]]]:
    """Return the five policy archetypes with the scenario used to derive them."""
    mechanism = config["mechanism"]
    normal_config = deepcopy(config)
    normal_config["mechanism"]["phi_high"] = float(mechanism.get("phi_high", 0.30))
    normal_config["mechanism"]["phi_low"] = float(mechanism.get("phi_low", 0.80))

    reverse_config = deepcopy(config)
    reverse_config["mechanism"]["phi_high"] = 0.90
    reverse_config["mechanism"]["phi_low"] = 0.10

    normal_policies = build_policies(
        eta_H=0.540,
        eta_L=0.282,
        rho=float(mechanism["verification_rate"]),
        penalty=float(mechanism["penalty"]),
        participation_cost=float(mechanism["participation_cost"]),
        phi_H=float(normal_config["mechanism"]["phi_high"]),
        phi_L=float(normal_config["mechanism"]["phi_low"]),
        strict_slack=float(mechanism["strict_slack"]),
    )
    reverse_policies = build_policies(
        eta_H=0.540,
        eta_L=0.282,
        rho=float(mechanism["verification_rate"]),
        penalty=float(mechanism["penalty"]),
        participation_cost=float(mechanism["participation_cost"]),
        phi_H=float(reverse_config["mechanism"]["phi_high"]),
        phi_L=float(reverse_config["mechanism"]["phi_low"]),
        strict_slack=float(mechanism["strict_slack"]),
    )

    oracle = Policy(
        name="oracle",
        rho=0.0,
        reward=float(mechanism["participation_cost"]) + float(mechanism["strict_slack"]),
        penalty=0.0,
    )
    normal = _policy_by_name(normal_policies, "normal_screening")
    reverse = _policy_by_name(reverse_policies, "reverse_screening")
    pooling_verified = _policy_by_name(normal_policies, "pooling_verification")
    pooling_unverified = _policy_by_name(normal_policies, "pooling_no_verification")
    return {
        "oracle": (oracle, normal_config),
        "normal_screening": (normal, normal_config),
        "reverse_screening": (reverse, reverse_config),
        "pooling_verified": (pooling_verified, normal_config),
        "pooling_unverified": (pooling_unverified, normal_config),
    }


def expected_policy_grid(
    item_table: pd.DataFrame,
    config: dict[str, Any],
    *,
    repetitions: int = 20,
    base_seed: int | None = None,
) -> pd.DataFrame:
    """Simulate participation, verification, volume, composition, and quality."""
    policies = canonical_policies(config)
    seed = int(config.get("seed", 42) if base_seed is None else base_seed)
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(POLICY_ORDER):
        policy, policy_config = policies[name]
        simulated_items = item_table.copy()
        if name == "oracle":
            simulated_items["helpful"] = 1
        for repetition in range(repetitions):
            result = simulate_policy(
                simulated_items,
                policy,
                policy_config,
                seed=seed + index * 1009 + repetition * 17,
            )
            row = result.__dict__.copy()
            row.update(
                {
                    "policy": name,
                    "scenario_phi_H": float(policy_config["mechanism"]["phi_high"]),
                    "scenario_phi_L": float(policy_config["mechanism"]["phi_low"]),
                    "rho": policy.rho,
                    "penalty": policy.penalty,
                    "repetition": repetition,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_policy_grid(grid: pd.DataFrame) -> pd.DataFrame:
    summary = (
        grid.groupby("policy", as_index=False)
        .agg(
            participation_rate=("participation_rate", "mean"),
            accepted_volume=("accepted", "mean"),
            accepted_volume_sd=("accepted", "std"),
            accepted_high_share=("accepted_high_share", "mean"),
            accepted_helpful_share=("accepted_helpful_share", "mean"),
            accepted_helpful_sd=("accepted_helpful_share", "std"),
            platform_payoff=("platform_payoff", "mean"),
            harmful_detected=("harmful_detected", "mean"),
            helpful_false_positive=("helpful_false_positive", "mean"),
        )
        .fillna(0.0)
    )
    return summary.set_index("policy").loc[POLICY_ORDER].reset_index()


def build_experiment_manifest(
    summary: pd.DataFrame,
    *,
    equal_size: int = 2000,
    endogenous_cap: int = 8000,
    endogenous_floor: int = 500,
) -> pd.DataFrame:
    """Convert policy-grid means into equal-size and endogenous-volume DPO targets."""
    if "accepted_volume" not in summary:
        raise ValueError("summary must contain accepted_volume")
    frame = summary.copy()
    normal_volume = float(frame.loc[frame["policy"] == "normal_screening", "accepted_volume"].iloc[0])
    scale = equal_size / normal_volume if normal_volume > 0 else 1.0
    frame["equal_size_target"] = int(equal_size)
    frame["endogenous_target"] = np.clip(
        np.round(frame["accepted_volume"] * scale),
        endogenous_floor,
        endogenous_cap,
    ).astype(int)
    frame["volume_ratio_to_normal"] = frame["accepted_volume"] / normal_volume
    return frame[
        [
            "policy",
            "participation_rate",
            "accepted_volume",
            "volume_ratio_to_normal",
            "accepted_high_share",
            "accepted_helpful_share",
            "equal_size_target",
            "endogenous_target",
        ]
    ]


def assert_policy_ordering(summary: pd.DataFrame) -> None:
    """Raise if the expected normal/reverse quality ordering is absent."""
    lookup = summary.set_index("policy")
    if lookup.loc["normal_screening", "accepted_helpful_share"] <= lookup.loc["reverse_screening", "accepted_helpful_share"]:
        raise AssertionError("Normal screening must have higher accepted helpfulness than reverse screening.")
    if lookup.loc["oracle", "accepted_helpful_share"] <= lookup.loc["reverse_screening", "accepted_helpful_share"]:
        raise AssertionError("Oracle must have higher accepted helpfulness than reverse screening.")
