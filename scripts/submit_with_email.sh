#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <sbatch script> [sbatch arguments ...]" >&2
  exit 2
fi

job_id=$(sbatch --parsable "$@")
echo "Submitted job: $job_id"
notify_id=$(sbatch --parsable \
  --dependency="afterany:${job_id}" \
  --export="ALL,RESULTS_DIR=${RESULTS_DIR:-runs}" \
  scripts/notify_completion.sbatch)
echo "Submitted email notifier: $notify_id (depends on $job_id)"
