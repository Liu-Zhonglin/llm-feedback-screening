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
MANIFEST="${MANIFEST:-pilot_results/full_chain/stage5_experiment_manifest.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/stage5}"
STEPS="${STEPS:-300}"
SEED_START="${SEED_START:-42}"
N_SEEDS="${N_SEEDS:-10}"
N_TASKS="${N_TASKS:-20}"

policies=(oracle normal_screening reverse_screening pooling_verified pooling_unverified)
seeds=()
for ((offset=0; offset<N_SEEDS; offset++)); do
  seeds+=("$((SEED_START + offset))")
done

run_one() {
  local mode="$1"
  local global_id="$2"
  local policy="${policies[$((global_id % ${#policies[@]}))]}"
  local seed="${seeds[$((global_id / ${#policies[@]}))]}"
  local target_column target run_dir

  if [[ "$mode" == "equal_size" ]]; then
    target_column=7
  elif [[ "$mode" == "endogenous" ]]; then
    target_column=8
  else
    echo "Unknown mode: $mode" >&2
    exit 2
  fi

  target="$(awk -F, -v p="$policy" -v c="$target_column" '$1==p {print $c}' "$MANIFEST" | head -n 1)"
  if [[ -z "$target" ]]; then
    echo "No target found for policy=$policy mode=$mode" >&2
    exit 3
  fi

  run_dir="$OUTPUT_ROOT/$mode/${policy}_seed${seed}"
  mkdir -p "$run_dir"

  python scripts/09_train_dpo_policy.py \
    --config "$CONFIG" \
    --pairs "$PAIRS" \
    --items "$ITEMS" \
    --policy "$policy" \
    --seed "$seed" \
    --max-train-examples "$target" \
    --max-steps "$STEPS" \
    --output-dir "$run_dir"

  python scripts/10_evaluate_dpo_policy.py \
    --config "$CONFIG" \
    --test-pairs "$TEST_PAIRS" \
    --adapter "$run_dir/adapter" \
    --label "${policy}_seed${seed}_${mode}" \
    --output "$run_dir/eval.json"

  echo "Completed policy=$policy seed=$seed mode=$mode target=$target" >> "$OUTPUT_ROOT/completed.txt"
}

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
total_tasks=$(( ${#policies[@]} * N_SEEDS ))
for ((global_id=task_id; global_id<total_tasks; global_id+=N_TASKS)); do
  for mode in equal_size endogenous; do
    run_one "$mode" "$global_id"
  done
done
