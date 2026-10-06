from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Policy:
    name: str
    rho: float
    reward: float
    penalty: float


@dataclass
class SimulationResult:
    policy: str
    n_users: int
    participation_rate: float
    high_participation_rate: float
    low_participation_rate: float
    submissions: int
    accepted: int
    accepted_high_share: float
    accepted_helpful_share: float
    accepted_harmful_share: float
    verified_fraction: float
    harmful_detected: int
    helpful_false_positive: int
    quality_value: float
    reward_payments: float
    penalty_recoveries: float
    verification_costs: float
    enforcement_costs: float
    platform_payoff: float


def participation_threshold(
    *,
    eta: float,
    phi: float,
    penalty: float,
    rho: float,
    participation_cost: float,
) -> float:
    return participation_cost + phi * penalty * rho * (1.0 - eta)


def screening_regime(
    *,
    eta_H: float,
    eta_L: float,
    phi_H: float,
    phi_L: float,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> str:
    """Classify the screening region implied by effective sanction exposure."""
    exposure_H = phi_H * (1.0 - eta_H)
    exposure_L = phi_L * (1.0 - eta_L)
    if np.isclose(exposure_H, exposure_L, rtol=rtol, atol=atol):
        return "boundary"
    return "normal" if exposure_H < exposure_L else "reverse"


def boundary_phi_H(
    *,
    eta_H: float,
    eta_L: float,
    phi_L: float,
) -> float:
    """Return the high-type exposure that places the system on the boundary."""
    return phi_L * (1.0 - eta_L) / (1.0 - eta_H)


def expected_training_mass(
    *,
    eta: float,
    participation: float,
    rho: float,
    verifier_tpr: float,
    verifier_fpr: float,
    unverified_harmful_weight: float,
) -> float:
    """Expected retained training mass for one type after verification."""
    helpful_retained = eta * (1.0 - rho * verifier_fpr)
    harmful_retained = (
        (1.0 - eta)
        * (1.0 - rho * verifier_tpr)
        * unverified_harmful_weight
    )
    return float(participation * (helpful_retained + harmful_retained))


def expected_accepted_metrics(
    *,
    eta_H: float,
    eta_L: float,
    lambda_high: float,
    high_participation: float,
    low_participation: float,
    rho: float,
    verifier_tpr: float,
    verifier_fpr: float,
    unverified_harmful_weight: float,
) -> tuple[float, float]:
    """Return accepted high-type share and accepted helpfulness."""
    mass_H = expected_training_mass(
        eta=eta_H,
        participation=high_participation,
        rho=rho,
        verifier_tpr=verifier_tpr,
        verifier_fpr=verifier_fpr,
        unverified_harmful_weight=unverified_harmful_weight,
    ) * lambda_high
    mass_L = expected_training_mass(
        eta=eta_L,
        participation=low_participation,
        rho=rho,
        verifier_tpr=verifier_tpr,
        verifier_fpr=verifier_fpr,
        unverified_harmful_weight=unverified_harmful_weight,
    ) * (1.0 - lambda_high)
    total_mass = mass_H + mass_L
    if total_mass <= 0.0:
        return float("nan"), float("nan")

    helpful_H = lambda_high * high_participation * eta_H * (1.0 - rho * verifier_fpr)
    helpful_L = (1.0 - lambda_high) * low_participation * eta_L * (1.0 - rho * verifier_fpr)
    harmful_H = (
        lambda_high
        * high_participation
        * (1.0 - eta_H)
        * (1.0 - rho * verifier_tpr)
        * unverified_harmful_weight
    )
    harmful_L = (
        (1.0 - lambda_high)
        * low_participation
        * (1.0 - eta_L)
        * (1.0 - rho * verifier_tpr)
        * unverified_harmful_weight
    )
    helpfulness = (helpful_H + helpful_L) / (helpful_H + helpful_L + harmful_H + harmful_L)
    return float(mass_H / total_mass), float(helpfulness)


def build_policies(
    *,
    eta_H: float,
    eta_L: float,
    rho: float,
    penalty: float,
    participation_cost: float,
    phi_H: float,
    phi_L: float,
    strict_slack: float,
) -> list[Policy]:
    threshold_H = participation_threshold(
        eta=eta_H,
        phi=phi_H,
        penalty=penalty,
        rho=rho,
        participation_cost=participation_cost,
    )
    threshold_L = participation_threshold(
        eta=eta_L,
        phi=phi_L,
        penalty=penalty,
        rho=rho,
        participation_cost=participation_cost,
    )

    policies: list[Policy] = [
        Policy(
            name="pooling_no_verification",
            rho=0.0,
            reward=participation_cost + strict_slack,
            penalty=0.0,
        ),
        Policy(
            name="pooling_verification",
            rho=rho,
            reward=max(threshold_H, threshold_L) + strict_slack,
            penalty=penalty,
        ),
    ]

    if threshold_H < threshold_L:
        policies.append(
            Policy(
                name="normal_screening",
                rho=rho,
                reward=(threshold_H + threshold_L) / 2.0,
                penalty=penalty,
            )
        )
    elif threshold_L < threshold_H:
        policies.append(
            Policy(
                name="reverse_screening",
                rho=rho,
                reward=(threshold_H + threshold_L) / 2.0,
                penalty=penalty,
            )
        )
    return policies


def _sample_item_pool(
    item_table: pd.DataFrame,
    *,
    types: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    pools = {
        type_name: item_table.loc[
            item_table["type_name"] == type_name,
            ["helpful", "user_id"],
        ].reset_index(drop=True)
        for type_name in ("H", "L")
    }
    sampled = []
    for type_name in types:
        pool = pools[type_name]
        if pool.empty:
            raise ValueError(f"No empirical feedback items for type {type_name}.")
        index = int(rng.integers(0, len(pool)))
        row = pool.iloc[index]
        sampled.append((type_name, int(row["helpful"])))
    return pd.DataFrame(sampled, columns=["type_name", "helpful"])


def simulate_policy(
    item_table: pd.DataFrame,
    policy: Policy,
    config: dict[str, Any],
    *,
    seed: int,
    policies_are_deterministic: bool = True,
) -> SimulationResult:
    mechanism = config["mechanism"]
    rng = np.random.default_rng(seed)
    n_users = int(mechanism["n_users"])
    lambda_high = float(mechanism["lambda_high"])
    cost = float(mechanism["participation_cost"])
    phi_by_type = {
        "H": float(mechanism["phi_high"]),
        "L": float(mechanism["phi_low"]),
    }

    types = np.where(rng.random(n_users) < lambda_high, "H", "L")
    sampled_items = _sample_item_pool(item_table, types=types, rng=rng)
    sampled_items["type_name"] = types
    sampled_items["phi"] = sampled_items["type_name"].map(phi_by_type)

    helper = _type_helpfulness(item_table)
    sampled_items["eta"] = sampled_items["type_name"].map(helper)
    sampled_items["delta"] = (
        policy.reward
        - cost
        - sampled_items["phi"] * policy.penalty * policy.rho * (1.0 - sampled_items["eta"])
    )

    if policies_are_deterministic:
        participates = sampled_items["delta"] >= 0
    else:
        temperature = float(mechanism.get("participation_temperature", 0.02))
        probability = 1.0 / (1.0 + np.exp(-sampled_items["delta"] / temperature))
        participates = rng.random(n_users) < probability

    submitted = sampled_items.loc[participates].copy()
    if submitted.empty:
        return SimulationResult(
            policy=policy.name,
            n_users=n_users,
            participation_rate=0.0,
            high_participation_rate=0.0,
            low_participation_rate=0.0,
            submissions=0,
            accepted=0,
            accepted_high_share=float("nan"),
            accepted_helpful_share=float("nan"),
            accepted_harmful_share=float("nan"),
            verified_fraction=0.0,
            harmful_detected=0,
            helpful_false_positive=0,
            quality_value=0.0,
            reward_payments=0.0,
            penalty_recoveries=0.0,
            verification_costs=0.0,
            enforcement_costs=0.0,
            platform_payoff=0.0,
        )

    submitted["verified"] = rng.random(len(submitted)) < policy.rho
    submitted["harmful"] = 1 - submitted["helpful"]
    submitted["detected_harmful"] = submitted["verified"] & (submitted["harmful"] == 1) & (
        rng.random(len(submitted)) < float(mechanism["verifier_tpr"])
    )
    submitted["false_positive"] = submitted["verified"] & (submitted["helpful"] == 1) & (
        rng.random(len(submitted)) < float(mechanism["verifier_fpr"])
    )
    submitted["excluded"] = submitted["detected_harmful"] | submitted["false_positive"]
    accepted = submitted.loc[~submitted["excluded"]].copy()
    accepted["training_weight"] = np.where(
        accepted["verified"] | (accepted["helpful"] == 1),
        1.0,
        float(mechanism["unverified_harmful_weight"]),
    )

    gamma_verified = float(mechanism["gamma_verified"])
    gamma_unverified_net = float(mechanism["gamma_unverified_net"])
    verified_helpful = accepted["verified"] & (accepted["helpful"] == 1)
    unverified_helpful = (~accepted["verified"]) & (accepted["helpful"] == 1)
    unverified_harmful = (~accepted["verified"]) & (accepted["harmful"] == 1)
    quality_value = (
        gamma_verified * float(verified_helpful.sum())
        + gamma_unverified_net
        * (
            float(unverified_helpful.sum())
            - float(mechanism["unverified_harmful_weight"]) * float(unverified_harmful.sum())
        )
    )

    reward_payments = policy.reward * len(submitted)
    penalty_recoveries = policy.penalty * int(submitted["detected_harmful"].sum())
    verification_costs = float(mechanism["verification_cost"]) * int(submitted["verified"].sum())
    enforcement_costs = float(mechanism["enforcement_cost"]) * (policy.penalty**2) * len(submitted)
    platform_payoff = quality_value - reward_payments - verification_costs + penalty_recoveries - enforcement_costs

    high_submitted = submitted["type_name"] == "H"
    low_submitted = submitted["type_name"] == "L"
    high_n = int((types == "H").sum())
    low_n = int((types == "L").sum())

    return SimulationResult(
        policy=policy.name,
        n_users=n_users,
        participation_rate=len(submitted) / n_users,
        high_participation_rate=float(high_submitted.sum() / high_n) if high_n else float("nan"),
        low_participation_rate=float(low_submitted.sum() / low_n) if low_n else float("nan"),
        submissions=len(submitted),
        accepted=len(accepted),
        accepted_high_share=(
            float((accepted["type_name"] == "H").mean()) if not accepted.empty else float("nan")
        ),
        accepted_helpful_share=float(accepted["helpful"].mean()) if not accepted.empty else float("nan"),
        accepted_harmful_share=float(accepted["harmful"].mean()) if not accepted.empty else float("nan"),
        verified_fraction=float(submitted["verified"].mean()),
        harmful_detected=int(submitted["detected_harmful"].sum()),
        helpful_false_positive=int(submitted["false_positive"].sum()),
        quality_value=quality_value,
        reward_payments=reward_payments,
        penalty_recoveries=penalty_recoveries,
        verification_costs=verification_costs,
        enforcement_costs=enforcement_costs,
        platform_payoff=platform_payoff,
    )


def _type_helpfulness(item_table: pd.DataFrame) -> dict[str, float]:
    return {
        str(type_name): float(group["helpful"].mean())
        for type_name, group in item_table.groupby("type_name", sort=False)
    }


def run_policy_grid(
    item_table: pd.DataFrame,
    config: dict[str, Any],
    *,
    penalty_grid: list[float] | None = None,
    repetitions: int = 20,
) -> pd.DataFrame:
    helper = _type_helpfulness(item_table)
    mechanism = config["mechanism"]
    penalties = penalty_grid or [float(mechanism["penalty"])]
    rows: list[dict[str, Any]] = []

    for penalty in penalties:
        policies = build_policies(
            eta_H=helper["H"],
            eta_L=helper["L"],
            rho=float(mechanism["verification_rate"]),
            penalty=penalty,
            participation_cost=float(mechanism["participation_cost"]),
            phi_H=float(mechanism["phi_high"]),
            phi_L=float(mechanism["phi_low"]),
            strict_slack=float(mechanism["strict_slack"]),
        )
        for policy in policies:
            for repetition in range(repetitions):
                result = simulate_policy(
                    item_table,
                    policy,
                    config,
                    seed=int(config["seed"]) + repetition * 1009 + int(round(penalty * 10000)),
                )
                row = asdict(result)
                row.update(
                    {
                        "rho": policy.rho,
                        "reward": policy.reward,
                        "penalty": policy.penalty,
                        "repetition": repetition,
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)
