from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / ".mplconfig").mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from revision3.config import load_config
from revision3.dpo_policy import derive_policy_pairs, train_dpo_policy
from revision3.preferences import deterministic_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--items", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-train-examples", type=int, default=None)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--max-steps", type=int, default=None)
    budget.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.pairs)
    items = pd.read_csv(args.items)
    train, validation, test = deterministic_split(
        pairs,
        seed=int(config["seed"]),
        train_fraction=0.8,
        validation_fraction=0.1,
    )
    train.to_csv(output_dir / "train_pairs.csv", index=False)
    validation.to_csv(output_dir / "validation_pairs.csv", index=False)
    test.to_csv(output_dir / "test_pairs.csv", index=False)
    n_examples = args.max_train_examples or int(config["model"]["max_train_examples"])
    max_steps = None if args.max_examples is not None else (args.max_steps or int(config["model"]["max_steps"]))
    feedback = derive_policy_pairs(
        train,
        items,
        config,
        policy=args.policy,
        seed=args.seed,
        n_examples=n_examples,
    )
    feedback.to_csv(output_dir / "policy_pairs.csv", index=False)
    metrics = train_dpo_policy(
        feedback,
        config,
        output_dir=output_dir,
        seed=args.seed,
        max_steps=max_steps,
        max_examples=args.max_examples,
    )
    metrics.update(
        {
            "policy": args.policy,
            "harm_mode": "preference_flip",
            "helpful_share": float(feedback["helpful"].mean()),
            "verified_share": float(feedback["verified"].mean()),
            "mean_weight": float(feedback["weight"].mean()),
            "high_type_share": float((feedback["type_name"] == "H").mean()),
            "accepted_pool_size": int(len(feedback)),
            "distinct_pair_count": int(feedback["pair_id"].nunique()),
            "distinct_tree_count": int(feedback["tree_id"].nunique()),
            "sampling_rule": "with_replacement",
            "processed_pair_budget": int(metrics["requested_processed_examples"]),
        }
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
