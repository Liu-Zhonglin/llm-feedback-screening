from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from revision3.preferences import deterministic_split

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="Full extracted feedback items CSV")
    parser.add_argument("--pairs", required=True, help="Preference pairs CSV/gzip")
    parser.add_argument("--test-pairs", required=True, help="Held-out preference pairs CSV/gzip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    return parser.parse_args()


def _fit_cluster(reliability: np.ndarray, method: str, seed: int) -> np.ndarray:
    values = np.asarray(reliability, dtype=float).reshape(-1, 1)
    if len(values) < 10 or np.unique(values).size < 2:
        cutoff = float(np.median(values))
        return (values.ravel() >= cutoff).astype(int)
    if method == "median":
        cutoff = float(np.median(values))
        return (values.ravel() >= cutoff).astype(int)
    if method == "quantile":
        cutoff = float(np.quantile(values, 0.5))
        return (values.ravel() >= cutoff).astype(int)
    covariance = {
        "gmm_full": "full",
        "gmm_diag": "diag",
        "gmm_spherical": "spherical",
    }.get(method)
    if covariance is None:
        raise ValueError(f"Unknown clustering method: {method}")
    model = GaussianMixture(
        n_components=2,
        covariance_type=covariance,
        n_init=20,
        max_iter=1000,
        random_state=seed,
    )
    labels = model.fit_predict(values)
    means = np.array([values.ravel()[labels == component].mean() for component in (0, 1)])
    high_component = int(np.argmax(means))
    return (labels == high_component).astype(int)


def _contributor_table(items: pd.DataFrame, *, min_items: int, reliability: str, adjustment: str) -> pd.DataFrame:
    work = items.copy()
    work["helpful"] = work["helpful"].astype(float)
    if adjustment == "prompt_adjusted":
        global_mean = float(work["helpful"].mean())
        parent_mean = work.groupby("parent_id", dropna=False)["helpful"].transform("mean")
        work["criterion_value"] = work["helpful"] - parent_mean + global_mean
    else:
        work["criterion_value"] = work["helpful"]

    grouped = work.groupby("user_id", sort=False)
    contributor = grouped.agg(
        n_items=("helpful", "size"),
        helpful_count=("helpful", "sum"),
        mean_helpful=("helpful", "mean"),
        mean_rank=("rank", "mean"),
        total_reviews=("review_count", "sum"),
        mean_adjusted=("criterion_value", "mean"),
    ).reset_index()
    contributor["shrunken_rate"] = (contributor["helpful_count"] + 1.0) / (contributor["n_items"] + 2.0)

    if reliability == "tree_loo":
        work["user_helpful_total"] = work.groupby("user_id")["helpful"].transform("sum")
        work["user_n_total"] = work.groupby("user_id")["helpful"].transform("size")
        work["tree_helpful"] = work.groupby(["user_id", "tree_id"])["helpful"].transform("sum")
        work["tree_n"] = work.groupby(["user_id", "tree_id"])["helpful"].transform("size")
        work["other_helpful"] = work["user_helpful_total"] - work["tree_helpful"]
        work["other_n"] = work["user_n_total"] - work["tree_n"]
        work["tree_loo_rate"] = np.where(
            work["other_n"] > 0,
            work["other_helpful"] / work["other_n"],
            np.nan,
        )
        contributor = (
            work.groupby("user_id", as_index=False)
            .agg(
                n_items=("helpful", "size"),
                helpful_count=("helpful", "sum"),
                mean_helpful=("helpful", "mean"),
                mean_rank=("rank", "mean"),
                total_reviews=("review_count", "sum"),
                mean_adjusted=("criterion_value", "mean"),
                reliability=("tree_loo_rate", "mean"),
                other_n=("other_n", "max"),
            )
            .loc[lambda frame: (frame["n_items"] >= min_items) & (frame["other_n"] > 0)]
        )
        return contributor.copy()

    contributor["loo_sum"] = np.nan
    contributor = contributor.set_index("user_id")
    for user_id, group in work.groupby("user_id", sort=False):
        n = len(group)
        helpful = float(group["helpful"].sum())
        if reliability == "raw_loo":
            loo = float(np.mean((helpful - group["helpful"].astype(float)) / max(n - 1, 1))) if n > 1 else float(group["helpful"].mean())
        elif reliability == "shrunken_rate":
            loo = (helpful + 1.0) / (n + 2.0)
        elif reliability == "prompt_adjusted_loo":
            adjusted = group["criterion_value"].astype(float)
            loo = float(np.mean((adjusted.sum() - adjusted) / max(n - 1, 1))) if n > 1 else float(adjusted.mean())
        else:
            raise ValueError(f"Unknown reliability: {reliability}")
        contributor.loc[user_id, "loo_sum"] = loo
    contributor = contributor.reset_index().rename(columns={"loo_sum": "reliability"})
    return contributor.loc[contributor["n_items"] >= min_items].copy()


def _evaluate_spec(items: pd.DataFrame, *, min_items: int, reliability: str, method: str, adjustment: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
    base = items.dropna(subset=["user_id", "parent_id", "tree_id"]).copy()
    contrib = _contributor_table(base, min_items=min_items, reliability=reliability, adjustment=adjustment)
    if len(contrib) < 10:
        return ({
            "min_items": min_items,
            "reliability": reliability,
            "method": method,
            "adjustment": adjustment,
            "n_contributors": len(contrib),
            "n_items": 0,
            "eta_H": float("nan"),
            "eta_L": float("nan"),
            "eta_diff": float("nan"),
            "critical_exposure_ratio": float("nan"),
        }, pd.DataFrame())
    contrib["type"] = _fit_cluster(contrib["reliability"].to_numpy(), method, seed)
    contrib["type_name"] = contrib["type"].map({1: "H", 0: "L"})
    joined = base.merge(contrib[["user_id", "reliability", "type", "type_name", "n_items"]], on="user_id", how="inner", suffixes=("", "_contrib"))
    eta = joined.groupby("type_name")["helpful"].mean().to_dict()
    eta_h = float(eta.get("H", np.nan))
    eta_l = float(eta.get("L", np.nan))
    ratio = float((1.0 - eta_l) / (1.0 - eta_h)) if np.isfinite(eta_h) and np.isfinite(eta_l) and eta_h < 1.0 else float("nan")
    result = {
        "min_items": min_items,
        "reliability": reliability,
        "method": method,
        "adjustment": adjustment,
        "n_contributors": int(len(contrib)),
        "n_items": int(len(joined)),
        "eta_H": eta_h,
        "eta_L": eta_l,
        "eta_diff": eta_h - eta_l,
        "critical_exposure_ratio": ratio,
    }
    return result, joined


def _bootstrap(joined: pd.DataFrame, *, repetitions: int, seed: int) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame(columns=["repetition", "eta_H", "eta_L", "eta_diff", "critical_exposure_ratio"])
    rng = np.random.default_rng(seed)
    rows = []
    by_type = {name: group for name, group in joined.groupby("type_name", sort=False)}
    for repetition in range(repetitions):
        sampled_frames = []
        for name, group in by_type.items():
            users = group["user_id"].unique()
            sampled_users = rng.choice(users, size=len(users), replace=True)
            chunks = []
            for sampled_user in sampled_users:
                user_group = group.loc[group["user_id"] == sampled_user]
                chunks.append(user_group.sample(n=len(user_group), replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
            sampled_frames.append(pd.concat(chunks, ignore_index=True))
        sample = pd.concat(sampled_frames, ignore_index=True)
        eta = sample.groupby("type_name")["helpful"].mean().to_dict()
        eta_h = float(eta.get("H", np.nan))
        eta_l = float(eta.get("L", np.nan))
        ratio = float((1.0 - eta_l) / (1.0 - eta_h)) if np.isfinite(eta_h) and np.isfinite(eta_l) and eta_h < 1.0 else float("nan")
        rows.append({"repetition": repetition, "eta_H": eta_h, "eta_L": eta_l, "eta_diff": eta_h - eta_l, "critical_exposure_ratio": ratio})
    return pd.DataFrame(rows)


def _pair_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["pair_id"].astype(str).str.split("::", expand=True)
    return pd.DataFrame({"pair_id": frame["pair_id"], "chosen_id": parts[0], "rejected_id": parts[1], "tree_id": frame["tree_id"]})


def _leakage_audit(items: pd.DataFrame, train_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> dict[str, Any]:
    item_ids = set(items["message_id"])
    train = _pair_ids(train_pairs)
    test = _pair_ids(test_pairs)
    train_ids = set(train["chosen_id"]).union(train["rejected_id"])
    test_ids = set(test["chosen_id"]).union(test["rejected_id"])
    item_tree = items.set_index("message_id")["tree_id"]
    train_trees = set(train["tree_id"].dropna())
    test_trees = set(test["tree_id"].dropna())
    item_trees = set(item_tree.dropna())
    return {
        "n_items": int(len(items)),
        "n_train_pairs": int(len(train_pairs)),
        "n_test_pairs": int(len(test_pairs)),
        "typed_items_in_train_pairs": int(len(item_ids & train_ids)),
        "typed_items_in_test_pairs": int(len(item_ids & test_ids)),
        "typed_tree_overlap_train": int(len(item_trees & train_trees)),
        "typed_tree_overlap_test": int(len(item_trees & test_trees)),
        "pair_tree_overlap_train_test": int(len(train_trees & test_trees)),
        "pair_id_overlap_train_test": int(len(set(train_pairs['pair_id']) & set(test_pairs['pair_id']))),
        "typed_item_coverage_train": float(len(item_ids & train_ids) / max(len(item_ids), 1)),
        "typed_item_coverage_test": float(len(item_ids & test_ids) / max(len(item_ids), 1)),
    }


def _split_generalization(items: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    work = items.dropna(subset=["user_id", "tree_id", "helpful"]).copy()
    contrib = _contributor_table(work, min_items=3, reliability="raw_loo", adjustment="none")
    if len(contrib) < 10:
        return {"coverage": 0.0, "eta_H": float("nan"), "eta_L": float("nan")}
    rng = np.random.default_rng(seed)
    users = np.array(sorted(contrib["user_id"].unique()), dtype=object)
    rng.shuffle(users)
    split = int(0.8 * len(users))
    train_users = set(users[:split])
    test_users = set(users[split:])
    train_contrib = contrib.loc[contrib["user_id"].isin(train_users)].copy()
    test_contrib = contrib.loc[contrib["user_id"].isin(test_users)].copy()
    if len(train_contrib) < 10 or test_contrib.empty:
        return {"coverage": 0.0, "eta_H": float("nan"), "eta_L": float("nan")}
    train_values = train_contrib["reliability"].to_numpy(dtype=float)
    test_values = test_contrib["reliability"].to_numpy(dtype=float)
    model = GaussianMixture(n_components=2, covariance_type="full", n_init=20, max_iter=1000, random_state=seed)
    model.fit(train_values.reshape(-1, 1))
    train_labels = model.predict(train_values.reshape(-1, 1))
    means = np.array([train_values[train_labels == component].mean() for component in (0, 1)])
    high_component = int(np.argmax(means))
    test_labels = model.predict(test_values.reshape(-1, 1))
    test_contrib["type"] = (test_labels == high_component).astype(int)
    test_contrib["type_name"] = test_contrib["type"].map({1: "H", 0: "L"})
    joined = work.merge(test_contrib[["user_id", "type_name"]], on="user_id", how="inner")
    if joined.empty:
        return {"coverage": 0.0, "eta_H": float("nan"), "eta_L": float("nan")}
    eta = joined.groupby("type_name")["helpful"].mean().to_dict()
    return {
        "train_contributors": int(train_contrib["user_id"].nunique()),
        "test_contributors": int(test_contrib["user_id"].nunique()),
        "test_items": int(len(joined)),
        "test_item_coverage": float(len(joined) / max(len(work), 1)),
        "eta_H": float(eta.get("H", np.nan)),
        "eta_L": float(eta.get("L", np.nan)),
        "eta_diff": float(eta.get("H", np.nan) - eta.get("L", np.nan)),
    }


def _tree_transfer(items: pd.DataFrame, *, seed: int) -> dict[str, Any]:
    work = items.dropna(subset=["user_id", "tree_id", "helpful"]).copy()
    trees = work["tree_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(trees)
    split = int(0.8 * len(trees))
    train_trees = set(trees[:split])
    test_trees = set(trees[split:])
    train = work.loc[work["tree_id"].isin(train_trees)].copy()
    test = work.loc[work["tree_id"].isin(test_trees)].copy()

    train_contrib = _contributor_table(train, min_items=3, reliability="raw_loo", adjustment="none")
    if len(train_contrib) < 10 or test.empty:
        return {"coverage": 0.0, "eta_H": float("nan"), "eta_L": float("nan")}
    train_contrib["type"] = _fit_cluster(train_contrib["reliability"].to_numpy(), "gmm_full", seed)
    train_contrib["type_name"] = train_contrib["type"].map({1: "H", 0: "L"})
    joined = test.merge(train_contrib[["user_id", "type_name"]], on="user_id", how="inner")
    if joined.empty:
        return {"coverage": 0.0, "eta_H": float("nan"), "eta_L": float("nan")}
    eta = joined.groupby("type_name")["helpful"].mean().to_dict()
    return {
        "train_trees": int(len(train_trees)),
        "test_trees": int(len(test_trees)),
        "test_items": int(len(joined)),
        "test_item_coverage": float(len(joined) / max(len(test), 1)),
        "test_contributors": int(joined["user_id"].nunique()),
        "eta_H": float(eta.get("H", np.nan)),
        "eta_L": float(eta.get("L", np.nan)),
        "eta_diff": float(eta.get("H", np.nan) - eta.get("L", np.nan)),
    }


def _write_figure(specs: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    frame = specs.dropna(subset=["eta_H", "eta_L"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    labels = [f"{r.min_items}/{r.reliability}/{r.method}" for r in frame.itertuples()]
    x = np.arange(len(frame))
    axes[0].bar(x - 0.18, frame["eta_H"], width=0.36, label="H")
    axes[0].bar(x + 0.18, frame["eta_L"], width=0.36, label="L")
    axes[0].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[0].set_ylabel("Helpfulness probability")
    axes[0].set_title("Calibration variants")
    axes[0].legend(frameon=False)
    axes[1].bar(x, frame["critical_exposure_ratio"], color="#4c78a8")
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    axes[1].set_ylabel(r"Critical ratio $\frac{1-\eta_L}{1-\eta_H}$")
    axes[1].set_title("Exposure-ratio boundary")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = pd.read_csv(args.items)
    pairs = pd.read_csv(args.pairs)
    test_pairs = pd.read_csv(args.test_pairs)

    train_pairs, validation_pairs, split_test_pairs = deterministic_split(
        pairs,
        seed=args.seed,
        train_fraction=0.8,
        validation_fraction=0.1,
    )
    leakage = _leakage_audit(items, train_pairs, test_pairs)
    leakage["saved_test_matches_split"] = bool(
        set(split_test_pairs["pair_id"]) == set(test_pairs["pair_id"])
    )
    (output_dir / "leakage_audit.json").write_text(json.dumps(leakage, indent=2, sort_keys=True), encoding="utf-8")

    specs = []
    joined_by_key: dict[tuple[str, str, str, int], pd.DataFrame] = {}
    for min_items in (3, 5, 10, 20):
        for reliability in ("raw_loo", "tree_loo", "prompt_adjusted_loo", "shrunken_rate"):
            for method in ("gmm_full", "gmm_diag", "gmm_spherical", "median", "quantile"):
                result, joined = _evaluate_spec(
                    items,
                    min_items=min_items,
                    reliability=reliability,
                    method=method,
                    adjustment="prompt_adjusted" if reliability == "prompt_adjusted_loo" else "none",
                    seed=args.seed,
                )
                specs.append(result)
                joined_by_key[(reliability, method, "prompt_adjusted" if reliability == "prompt_adjusted_loo" else "none", min_items)] = joined
    specs_frame = pd.DataFrame(specs)
    specs_frame.to_csv(output_dir / "calibration_variants.csv", index=False)

    primary_result, primary_joined = _evaluate_spec(
        items,
        min_items=3,
        reliability="raw_loo",
        method="gmm_full",
        adjustment="none",
        seed=args.seed,
    )
    bootstrap = _bootstrap(primary_joined, repetitions=args.bootstrap_repetitions, seed=args.seed + 1)
    bootstrap.to_csv(output_dir / "primary_bootstrap.csv", index=False)

    split = _split_generalization(items, seed=args.seed)
    (output_dir / "split_generalization.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")

    tree_transfer = _tree_transfer(items, seed=args.seed)
    (output_dir / "tree_transfer.json").write_text(json.dumps(tree_transfer, indent=2, sort_keys=True), encoding="utf-8")

    (output_dir / "primary_estimates.json").write_text(json.dumps(primary_result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"primary": primary_result, "leakage": leakage, "split": split}, indent=2, sort_keys=True))
    print(f"Wrote calibration audit to {output_dir}")


if __name__ == "__main__":
    main()
