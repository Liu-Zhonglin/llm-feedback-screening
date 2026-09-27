from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from revision3.config import load_config
from revision3.dpo_eval import evaluate_frame
from revision3.sft import load_sft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-pairs", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    test = pd.read_csv(args.test_pairs)
    model, tokenizer, device = load_sft_model(args.adapter, config)
    metrics, _ = evaluate_frame(
        model,
        test,
        tokenizer,
        device=device,
        max_length=int(config["model"]["max_length"]),
        batch_size=int(config["model"]["batch_size"]),
    )
    result = {"label": args.label, "adapter": args.adapter, **metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
