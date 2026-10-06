#!/usr/bin/env bash
#SBATCH --job-name=feedback-dpo-1p5b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --array=0-49%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs runs/hpc_dpo_confirm_1p5b_300
export CONFIG=configs/hpc_dpo_confirm_1p5b.yaml
export OUTPUT_ROOT=runs/hpc_dpo_confirm_1p5b_300
export STEPS=300
export EXAMPLES=2000
export SEED_START=42
export N_SEEDS=10
bash scripts/run_hpc_dpo_scale.sh
