#!/usr/bin/env bash
#SBATCH --job-name=feedback-notify
#SBATCH --account=math_amacl
#SBATCH --partition=intel
#SBATCH --mail-user=u3597461@connect.hku.hk
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs
echo "Completion notification dispatched at $(date '+%Y-%m-%d %H:%M:%S %Z')."
echo "Results directory: ${RESULTS_DIR:-runs}"
