from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision3.config import load_config
from revision3.dpo_policy import derive_policy_pairs, optimizer_steps_for_examples
from revision3.preferences import deterministic_split


POLICIES = [
    "oracle",
    "normal_screening",
    "reverse_screening",
    "pooling_verified",
    "pooling_unverified",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--stage5-manifest", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run-report", default=None)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pairs = pd.read_csv(args.pairs)
    items = pd.read_csv(args.items)
    test_pairs = pd.read_csv(args.test_pairs)
    stage5 = pd.read_csv(args.stage5_manifest).set_index("policy")
    batch_size = int(config["model"]["batch_size"])
    train, _, _ = deterministic_split(
        pairs,
        seed=int(config["seed"]),
        train_fraction=0.8,
        validation_fraction=0.1,
    )
    pair_overlap = len(set(train["pair_id"]) & set(test_pairs["pair_id"]))
    tree_overlap = len(set(train["tree_id"]) & set(test_pairs["tree_id"]))
    if len(test_pairs) != 800:
        raise SystemExit(f"Expected 800 held-out test pairs, found {len(test_pairs)}")
    if pair_overlap != 0 or tree_overlap != 0:
        raise SystemExit(
            f"Evaluation isolation failed: pair_overlap={pair_overlap}, tree_overlap={tree_overlap}"
        )

    rows: list[dict[str, object]] = []
    for policy in POLICIES:
        accepted_target = int(stage5.loc[policy, "endogenous_target"])
        for seed in range(args.seed_start, args.seed_start + args.n_seeds):
            feedback = derive_policy_pairs(
                train,
                items,
                config,
                policy=policy,
                seed=seed,
                n_examples=accepted_target,
            )
            processed_pair_budget = int(len(feedback))
            optimizer_steps = optimizer_steps_for_examples(processed_pair_budget, batch_size)
            rows.append(
                {
                    "design": "volume_proportional",
                    "policy": policy,
                    "seed": seed,
                    "accepted_pool_size": processed_pair_budget,
                    "distinct_pair_count": int(feedback["pair_id"].nunique()),
                    "distinct_tree_count": int(feedback["tree_id"].nunique()),
                    "processed_pair_budget": processed_pair_budget,
                    "effective_batch_size": batch_size,
                    "optimizer_steps": optimizer_steps,
                    "sampling_rule": "with_replacement",
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if len(frame) != len(POLICIES) * args.n_seeds:
        raise SystemExit(f"Expected {len(POLICIES) * args.n_seeds} manifest rows, found {len(frame)}")
    if (frame["accepted_pool_size"] != frame["processed_pair_budget"]).any():
        raise SystemExit("Accepted-pool size and processed-pair budget must match in this design.")
    expected_steps = [
        optimizer_steps_for_examples(int(n), batch_size)
        for n in frame["processed_pair_budget"]
    ]
    if (frame["optimizer_steps"].astype(int).tolist() != expected_steps):
        raise SystemExit("Manifest optimizer steps do not match the processed-pair budget.")
    frame.to_csv(output, index=False)
    if args.dry_run_report:
        import json
        report = {
            "status": "pass",
            "manifest_rows": len(frame),
            "held_out_pairs": int(len(test_pairs)),
            "pair_overlap_with_train": int(pair_overlap),
            "tree_overlap_with_train": int(tree_overlap),
            "effective_batch_size": batch_size,
            "processed_pair_budget_by_policy": {
                policy: int(group["processed_pair_budget"].iloc[0])
                for policy, group in frame.groupby("policy", sort=False)
            },
            "optimizer_steps_by_policy": {
                policy: int(group["optimizer_steps"].iloc[0])
                for policy, group in frame.groupby("policy", sort=False)
            },
        }
        Path(args.dry_run_report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(frame.groupby("policy").agg(
        seeds=("seed", "nunique"),
        accepted_pool_size=("accepted_pool_size", "first"),
        distinct_pair_min=("distinct_pair_count", "min"),
        distinct_pair_max=("distinct_pair_count", "max"),
        processed_pair_budget=("processed_pair_budget", "first"),
        optimizer_steps=("optimizer_steps", "first"),
    ).to_string())
    print(f"\nWrote {len(frame)} runs to {output}")


if __name__ == "__main__":
    main()
