#!/usr/bin/env bash
#SBATCH --job-name=feedback-dpo-scale
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --array=0-24%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs runs/hpc_dpo_scale_300
bash scripts/run_hpc_dpo_scale.sh
