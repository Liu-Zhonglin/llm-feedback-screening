from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="Full extracted feedback items CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    return parser.parse_args()


def _prepare_contributors(items: pd.DataFrame) -> pd.DataFrame:
    work = items.dropna(subset=["user_id", "tree_id", "parent_id", "helpful"]).copy()
    work["helpful"] = work["helpful"].astype(float)
    work["user_helpful_total"] = work.groupby("user_id")["helpful"].transform("sum")
    work["user_n_total"] = work.groupby("user_id")["helpful"].transform("size")
    work["tree_helpful"] = work.groupby(["user_id", "tree_id"])["helpful"].transform("sum")
    work["tree_n"] = work.groupby(["user_id", "tree_id"])["helpful"].transform("size")
    work["other_helpful"] = work["user_helpful_total"] - work["tree_helpful"]
    work["other_n"] = work["user_n_total"] - work["tree_n"]
    work["tree_loo_rate"] = np.where(work["other_n"] > 0, work["other_helpful"] / work["other_n"], np.nan)
    contributor = (
        work.groupby("user_id", as_index=False)
        .agg(
            n_items=("helpful", "size"),
            helpful_count=("helpful", "sum"),
            reliability=("tree_loo_rate", "mean"),
            other_n=("other_n", "max"),
        )
        .loc[lambda frame: (frame["n_items"] >= 3) & (frame["other_n"] > 0)]
    )
    return work.merge(contributor, on="user_id", how="inner", suffixes=("", "_contrib"))


def _fit_types(reliability: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    values = np.asarray(reliability, dtype=float).reshape(-1, 1)
    if len(values) < 10 or np.unique(values).size < 2:
        cutoff = float(np.median(values))
        return (values.ravel() >= cutoff).astype(int), cutoff
    model = GaussianMixture(n_components=2, covariance_type="full", n_init=10, max_iter=1000, random_state=seed)
    labels = model.fit_predict(values)
    means = np.array([values.ravel()[labels == component].mean() for component in (0, 1)])
    high_component = int(np.argmax(means))
    return (labels == high_component).astype(int), float(means[high_component])


def _eta_from_types(frame: pd.DataFrame) -> tuple[float, float, float]:
    eta = frame.groupby("type_name")["helpful"].mean().to_dict()
    eta_h = float(eta.get("H", np.nan))
    eta_l = float(eta.get("L", np.nan))
    ratio = float((1.0 - eta_l) / (1.0 - eta_h)) if np.isfinite(eta_h) and np.isfinite(eta_l) and eta_h < 1.0 else float("nan")
    return eta_h, eta_l, ratio


def _bootstrap_types(items: pd.DataFrame, *, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    users = items["user_id"].drop_duplicates().to_numpy()
    by_user = {user: group for user, group in items.groupby("user_id", sort=False)}
    rows: list[dict[str, float]] = []
    for repetition in range(repetitions):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sampled_rows = []
        for user in sampled_users:
            group = by_user[user]
            sampled_rows.append(group.sample(n=len(group), replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
        sample = pd.concat(sampled_rows, ignore_index=True)
        contributor = sample.groupby("user_id", as_index=False).agg(
            n_items=("helpful", "size"),
            helpful_count=("helpful", "sum"),
            reliability=("helpful", "mean"),
        )
        if len(contributor) < 10:
            continue
        labels, _ = _fit_types(contributor["reliability"].to_numpy(), int(seed + repetition))
        contributor["type_name"] = np.where(labels == 1, "H", "L")
        joined = sample.merge(contributor[["user_id", "type_name"]], on="user_id", how="inner")
        eta_h, eta_l, ratio = _eta_from_types(joined)
        rows.append({"repetition": repetition, "eta_H": eta_h, "eta_L": eta_l, "eta_diff": eta_h - eta_l, "critical_ratio": ratio})
    return pd.DataFrame(rows)


def _bootstrap_fixed_labels(items: pd.DataFrame, *, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contributor = items.groupby("user_id", as_index=False).agg(
        n_items=("helpful", "size"),
        reliability=("helpful", "mean"),
    )
    contributor = contributor.loc[contributor["n_items"] >= 3].copy()
    labels, _ = _fit_types(contributor["reliability"].to_numpy(), seed)
    contributor["type_name"] = np.where(labels == 1, "H", "L")
    label_map = contributor.set_index("user_id")["type_name"].to_dict()
    by_user = {user: group for user, group in items.groupby("user_id", sort=False)}
    users = np.array([user for user in by_user if user in label_map], dtype=object)
    rows = []
    for repetition in range(repetitions):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sampled = []
        for user in sampled_users:
            group = by_user[user]
            sampled.append(group.sample(n=len(group), replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
        sample = pd.concat(sampled, ignore_index=True)
        sample["type_name"] = sample["user_id"].map(label_map)
        eta_h, eta_l, ratio = _eta_from_types(sample)
        rows.append({"repetition": repetition, "eta_H": eta_h, "eta_L": eta_l, "eta_diff": eta_h - eta_l, "critical_ratio": ratio})
    return pd.DataFrame(rows)


def _continuous_sensitivity(items: pd.DataFrame) -> pd.DataFrame:
    contributor = items.groupby("user_id", as_index=False).agg(
        n_items=("helpful", "size"),
        helpful_count=("helpful", "sum"),
        reliability=("helpful", "mean"),
    )
    contributor = contributor.loc[contributor["n_items"] >= 3].copy()
    quality = contributor["reliability"].to_numpy(dtype=float)
    std = float(np.std(quality)) if np.std(quality) > 0 else 1.0
    z = (quality - float(np.mean(quality))) / std
    schedules = {
        "constant": np.full_like(z, 0.5),
        "mild_increasing": np.clip(0.5 + 0.5 * z, 0.1, 2.0),
        "strong_increasing": np.clip(0.5 + 1.0 * z, 0.1, 2.0),
        "decreasing": np.clip(0.5 - 0.5 * z, 0.1, 2.0),
    }
    rho, tpr, fpr, harmful_weight = 0.30, 0.80, 0.10, 0.25
    reward_slack = 0.2
    total_weight = float(contributor["n_items"].sum())
    baseline_quality = float(np.average(quality, weights=contributor["n_items"]))
    rows: list[dict[str, Any]] = []
    for schedule_name, phi in schedules.items():
        g = phi * (1.0 - quality)
        for s in np.linspace(0.0, 3.0, 121):
            selected = np.ones(len(contributor), dtype=bool) if s == 0.0 else (g * s <= reward_slack)
            weights = contributor["n_items"].to_numpy(dtype=float)[selected]
            if weights.sum() <= 0:
                continue
            selected_quality = quality[selected]
            selected_phi = phi[selected]
            helpful_retained = selected_quality * (1.0 - rho * fpr)
            harmful_retained = (1.0 - selected_quality) * (1.0 - rho * tpr) * harmful_weight
            helpfulness = float(np.average(helpful_retained / (helpful_retained + harmful_retained), weights=weights))
            rows.append(
                {
                    "schedule": schedule_name,
                    "screening_intensity": float(s),
                    "participation": float(weights.sum() / total_weight),
                    "selected_quality": float(np.average(selected_quality, weights=weights)),
                    "selected_helpfulness": helpfulness,
                    "quality_change_vs_no_screening": float(np.average(selected_quality, weights=weights) - baseline_quality),
                    "mean_exposure": float(np.average(selected_phi, weights=weights)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    items = pd.read_csv(args.items)
    work = _prepare_contributors(items)

    bootstrap = _bootstrap_types(work, repetitions=args.bootstrap_repetitions, seed=args.seed)
    bootstrap.to_csv(output_dir / "boundary_bootstrap.csv", index=False)
    bootstrap_fixed = _bootstrap_fixed_labels(work, repetitions=args.bootstrap_repetitions, seed=args.seed + 17)
    bootstrap_fixed.to_csv(output_dir / "boundary_bootstrap_fixed.csv", index=False)
    summary = {
        "eta_H": {
            "mean": float(bootstrap["eta_H"].mean()),
            "lower": float(bootstrap["eta_H"].quantile(0.025)),
            "upper": float(bootstrap["eta_H"].quantile(0.975)),
        },
        "eta_L": {
            "mean": float(bootstrap["eta_L"].mean()),
            "lower": float(bootstrap["eta_L"].quantile(0.025)),
            "upper": float(bootstrap["eta_L"].quantile(0.975)),
        },
        "eta_diff": {
            "mean": float(bootstrap["eta_diff"].mean()),
            "lower": float(bootstrap["eta_diff"].quantile(0.025)),
            "upper": float(bootstrap["eta_diff"].quantile(0.975)),
        },
        "critical_ratio": {
            "mean": float(bootstrap["critical_ratio"].mean()),
            "lower": float(bootstrap["critical_ratio"].quantile(0.025)),
            "upper": float(bootstrap["critical_ratio"].quantile(0.975)),
        },
    }
    summary["fixed_label_quantiles"] = {
        "eta_H": [float(bootstrap_fixed["eta_H"].quantile(0.025)), float(bootstrap_fixed["eta_H"].quantile(0.975))],
        "eta_L": [float(bootstrap_fixed["eta_L"].quantile(0.025)), float(bootstrap_fixed["eta_L"].quantile(0.975))],
        "eta_diff": [float(bootstrap_fixed["eta_diff"].quantile(0.025)), float(bootstrap_fixed["eta_diff"].quantile(0.975))],
        "critical_ratio": [float(bootstrap_fixed["critical_ratio"].quantile(0.025)), float(bootstrap_fixed["critical_ratio"].quantile(0.975))],
    }
    (output_dir / "boundary_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    scenarios = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    regime_rows = []
    for scenario in scenarios:
        regime_rows.append(
            {
                "exposure_ratio": scenario,
                "prob_normal": float(np.mean(bootstrap["critical_ratio"] > scenario)),
                "prob_reverse": float(np.mean(bootstrap["critical_ratio"] < scenario)),
                "prob_close_boundary": float(np.mean(np.abs(bootstrap["critical_ratio"] - scenario) < 0.05)),
            }
        )
    pd.DataFrame(regime_rows).to_csv(output_dir / "regime_probabilities.csv", index=False)

    continuous = _continuous_sensitivity(work)
    continuous.to_csv(output_dir / "continuous_sensitivity.csv", index=False)
    endpoint = continuous.loc[continuous["screening_intensity"].isin([0.0, 1.5])].copy()
    endpoint.to_csv(output_dir / "continuous_endpoints.csv", index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote boundary and continuous-type sensitivity to {output_dir}")


if __name__ == "__main__":
    main()
