#!/usr/bin/env bash
#SBATCH --job-name=feedback-noise
#SBATCH --account=math_amacl
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --array=0-14%15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
mkdir -p logs "runs/hpc_noise_300/${NOISE_NAME}"
bash scripts/run_hpc_noise_matrix.sh
