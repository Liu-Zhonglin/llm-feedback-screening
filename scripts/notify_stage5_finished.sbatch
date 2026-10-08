#!/usr/bin/env bash
#SBATCH --job-name=stage5-notify
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=intel
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
echo "Stage 5 array dependency has finished."
