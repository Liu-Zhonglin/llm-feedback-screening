#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

CONFIG="${CONFIG:-configs/hpc_dpo_confirm_1p5b.yaml}"
PAIRS="${PAIRS:-data/processed/oasst_preference_pairs.csv.gz}"
ITEMS="${ITEMS:-data/processed/oasst_typed_items.csv.gz}"
TEST_PAIRS="${TEST_PAIRS:-data/processed/oasst_test_pairs.csv.gz}"
MANIFEST="${MANIFEST:-pilot_results/volume_proportional/volume_manifest.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/volume_proportional}"
SEED_START="${SEED_START:-42}"
N_SEEDS="${N_SEEDS:-10}"
N_TASKS="${N_TASKS:-20}"

policies=(oracle normal_screening reverse_screening pooling_verified pooling_unverified)
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
total_tasks=$(( ${#policies[@]} * N_SEEDS ))
for ((global_id=task_id; global_id<total_tasks; global_id+=N_TASKS)); do
  policy="${policies[$((global_id % ${#policies[@]}))]}"
  seed=$((SEED_START + global_id / ${#policies[@]}))
  row="$(awk -F, -v p="$policy" -v s="$seed" '$1=="volume_proportional" && $2==p && $3==s {print; exit}' "$MANIFEST")"
  if [[ -z "$row" ]]; then
    echo "No manifest row for policy=$policy seed=$seed" >&2
    exit 3
  fi
  accepted_pool_size="$(printf '%s\n' "$row" | cut -d, -f4)"
  processed_pair_budget="$(printf '%s\n' "$row" | cut -d, -f7)"
  run_dir="$OUTPUT_ROOT/${policy}_seed${seed}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN policy=$policy seed=$seed accepted_pool_size=$accepted_pool_size processed_pair_budget=$processed_pair_budget run_dir=$run_dir"
    continue
  fi
  mkdir -p "$run_dir"

  python scripts/09_train_dpo_policy.py \
    --config "$CONFIG" \
    --pairs "$PAIRS" \
    --items "$ITEMS" \
    --policy "$policy" \
    --seed "$seed" \
    --max-train-examples "$accepted_pool_size" \
    --max-examples "$processed_pair_budget" \
    --output-dir "$run_dir"

  python scripts/23_validate_volume_metrics.py \
    --manifest "$MANIFEST" \
    --metrics "$run_dir/metrics.json" \
    --policy "$policy" \
    --seed "$seed"

  python scripts/10_evaluate_dpo_policy.py \
    --config "$CONFIG" \
    --test-pairs "$TEST_PAIRS" \
    --adapter "$run_dir/adapter" \
    --label "${policy}_seed${seed}_volume_proportional" \
    --output "$run_dir/eval.json"

  echo "Completed policy=$policy seed=$seed processed_pairs=$processed_pair_budget" >> "$OUTPUT_ROOT/completed.txt"
done
