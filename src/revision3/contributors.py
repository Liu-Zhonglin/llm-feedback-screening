from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class TypeEstimate:
    contributor_table: pd.DataFrame
    item_table: pd.DataFrame
    summary: dict[str, Any]


def _cluster_types(reliability: np.ndarray, weights: np.ndarray, seed: int) -> np.ndarray:
    if len(reliability) < 10 or np.unique(reliability).size < 2:
        cutoff = float(np.median(reliability))
        return (reliability >= cutoff).astype(int)

    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        max_iter=1000,
        random_state=seed,
    )
    model.fit(reliability.reshape(-1, 1))
    labels = model.predict(reliability.reshape(-1, 1))
    means = [reliability[labels == component].mean() for component in (0, 1)]
    high_component = int(np.argmax(means))
    return (labels == high_component).astype(int)


def estimate_contributor_types(
    items: pd.DataFrame,
    *,
    min_items: int,
    seed: int,
    bootstrap_repetitions: int,
    confidence_level: float,
) -> TypeEstimate:
    work = items.copy()
    work["helpful"] = work["helpful"].astype(int)

    grouped = work.groupby("user_id", sort=False)
    contributor = grouped.agg(
        n_items=("helpful", "size"),
        helpful_count=("helpful", "sum"),
        mean_helpful=("helpful", "mean"),
        mean_rank=("rank", "mean"),
        total_reviews=("review_count", "sum"),
    ).reset_index()
    contributor["shrunken_rate"] = (
        contributor["helpful_count"] + 1.0
    ) / (contributor["n_items"] + 2.0)

    eligible = contributor["n_items"] >= min_items
    contrib_eligible = contributor.loc[eligible].copy()
    item_eligible = work.merge(
        contrib_eligible[["user_id"]],
        on="user_id",
        how="inner",
    ).copy()

    joined = item_eligible.merge(
        contrib_eligible[["user_id", "n_items", "helpful_count"]],
        on="user_id",
        how="left",
        suffixes=("", "_contrib"),
    )
    joined["loo_helpful_rate"] = np.where(
        joined["n_items"] > 1,
        (joined["helpful_count"] - joined["helpful"]) / (joined["n_items"] - 1),
        joined["helpful"],
    )

    contributor_loo = (
        joined.groupby("user_id", as_index=False)["loo_helpful_rate"].mean()
        .rename(columns={"loo_helpful_rate": "loo_rate"})
    )
    contrib_eligible = contrib_eligible.merge(contributor_loo, on="user_id", how="left")
    contrib_eligible["type"] = _cluster_types(
        contrib_eligible["loo_rate"].to_numpy(dtype=float),
        contrib_eligible["n_items"].to_numpy(dtype=float),
        seed,
    )
    contrib_eligible["type_name"] = contrib_eligible["type"].map({1: "H", 0: "L"})
    contributor = contributor.merge(
        contrib_eligible[["user_id", "loo_rate", "type", "type_name"]],
        on="user_id",
        how="left",
    )

    item_table = work.merge(
        contributor[["user_id", "n_items", "loo_rate", "type", "type_name"]],
        on="user_id",
        how="left",
        validate="many_to_one",
    )
    item_table = item_table.rename(columns={"n_items": "contributor_n_items"})

    eta = {}
    for type_name in ("H", "L"):
        subset = item_table.loc[item_table["type_name"] == type_name, "helpful"]
        eta[type_name] = float(subset.mean()) if len(subset) else float("nan")

    lo, hi = _cluster_bootstrap(
        item_table,
        repetitions=bootstrap_repetitions,
        seed=seed + 1,
        confidence_level=confidence_level,
    )

    summary = {
        "n_items": int(len(work)),
        "n_contributors": int(work["user_id"].nunique()),
        "n_contributors_classified": int(contrib_eligible["user_id"].nunique()),
        "n_items_classified": int(item_table["type_name"].notna().sum()),
        "eta_H": eta["H"],
        "eta_L": eta["L"],
        "eta_H_ci": lo["H"],
        "eta_L_ci": lo["L"],
        "eta_H_hi": hi["H"],
        "eta_L_hi": hi["L"],
        "contributor_type_counts": {
            str(k): int(v) for k, v in contrib_eligible["type_name"].value_counts().items()
        },
    }
    return TypeEstimate(contributor, item_table, summary)


def _cluster_bootstrap(
    item_table: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> tuple[dict[str, float], dict[str, float]]:
    rng = np.random.default_rng(seed)
    classified = item_table.loc[item_table["type_name"].notna()].copy()
    grouped = {name: group for name, group in classified.groupby("type_name", sort=False)}
    if set(grouped) != {"H", "L"}:
        return ({"H": float("nan"), "L": float("nan")}, {"H": float("nan"), "L": float("nan")})

    estimates = {"H": [], "L": []}
    for _ in range(repetitions):
        for type_name, group in grouped.items():
            users = group["user_id"].unique()
            sampled_users = rng.choice(users, size=len(users), replace=True)
            sampled_values: list[float] = []
            values_by_user = {
                user_id: user_group["helpful"].to_numpy(dtype=float)
                for user_id, user_group in group.groupby("user_id", sort=False)
            }
            for user_id in sampled_users:
                values = values_by_user[user_id]
                sampled_values.extend(rng.choice(values, size=len(values), replace=True).tolist())
            estimates[type_name].append(float(np.mean(sampled_values)))

    alpha = (1.0 - confidence_level) / 2.0
    lo = {name: float(np.quantile(values, alpha)) for name, values in estimates.items()}
    hi = {name: float(np.quantile(values, 1.0 - alpha)) for name, values in estimates.items()}
    return lo, hi
