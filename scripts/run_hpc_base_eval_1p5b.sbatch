#!/usr/bin/env bash
#SBATCH --job-name=feedback-base-1p5b
#SBATCH --account=math_amacl
#SBATCH --partition=l40s
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd /home/u3597461/revision3
source .venv/bin/activate
export PYTHONPATH=src
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python scripts/10_evaluate_dpo_policy.py \
  --config configs/hpc_dpo_confirm_1p5b.yaml \
  --test-pairs data/processed/oasst_test_pairs.csv.gz \
  --adapter base \
  --label base \
  --output runs/hpc_dpo_confirm_1p5b_300/base_eval.json
