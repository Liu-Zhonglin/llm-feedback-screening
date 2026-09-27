from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from revision3.config import load_config
from revision3.contributors import estimate_contributor_types
from revision3.oasst import extract_feedback_items, load_oasst
from revision3.preferences import build_preference_pairs, deterministic_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the processed OpenAssistant feedback data used in the experiments."
    )
    parser.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "data" / "processed")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_oasst(config)
    items = extract_feedback_items(dataset, config)
    estimate = estimate_contributor_types(
        items,
        min_items=int(config["types"]["min_items_per_contributor"]),
        seed=int(config["seed"]),
        bootstrap_repetitions=int(config["types"]["bootstrap_repetitions"]),
        confidence_level=float(config["types"]["confidence_level"]),
    )
    pairs = build_preference_pairs(dataset, estimate.item_table)
    _, _, test_pairs = deterministic_split(
        pairs,
        seed=int(config["seed"]),
        train_fraction=0.8,
        validation_fraction=0.1,
    )

    estimate.item_table[
        ["message_id", "user_id", "helpful", "type_name", "contributor_n_items", "loo_rate"]
    ].to_csv(output_dir / "oasst_typed_items.csv.gz", index=False, compression="gzip")
    pairs.to_csv(output_dir / "oasst_preference_pairs.csv.gz", index=False, compression="gzip")
    test_pairs.to_csv(output_dir / "oasst_test_pairs.csv.gz", index=False, compression="gzip")
    estimate.contributor_table.to_csv(
        output_dir / "oasst_contributor_types.csv.gz", index=False, compression="gzip"
    )
    (output_dir / "type_summary.json").write_text(
        json.dumps(estimate.summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Eligible feedback items: {len(items):,}")
    print(f"Preference pairs: {len(pairs):,}")
    print(f"Held-out test pairs: {len(test_pairs):,}")
    print(f"Wrote processed data to {output_dir}")


if __name__ == "__main__":
    main()
