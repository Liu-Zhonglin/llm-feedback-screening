#!/usr/bin/env bash
#SBATCH --job-name=stage5-combined
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --array=0-19%20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs runs/stage5
export OUTPUT_ROOT=runs/stage5
export N_TASKS=20
bash scripts/run_hpc_stage5_combined.sh
