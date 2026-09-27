from __future__ import annotations

import argparse
from pathlib import Path

import yaml


NOISE_SETTINGS = {
    "perfect": {"verifier_tpr": 1.0, "verifier_fpr": 0.0},
    "noisy": {"verifier_tpr": 0.6, "verifier_fpr": 0.2},
    "uninformative": {"verifier_tpr": 0.5, "verifier_fpr": 0.5},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_config)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, overrides in NOISE_SETTINGS.items():
        config = yaml.safe_load(yaml.safe_dump(base))
        config["mechanism"].update(overrides)
        output = output_dir / f"noise_{name}.yaml"
        output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
