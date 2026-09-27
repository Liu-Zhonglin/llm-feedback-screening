#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6

NOISE_NAME="${NOISE_NAME:?NOISE_NAME is required}"
CONFIG="configs/noise_${NOISE_NAME}.yaml"
PAIRS="${PAIRS:-data/processed/oasst_preference_pairs.csv.gz}"
ITEMS="${ITEMS:-data/processed/oasst_typed_items.csv.gz}"
TEST_PAIRS="${TEST_PAIRS:-data/processed/oasst_test_pairs.csv.gz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/hpc_noise_300/$NOISE_NAME}"
STEPS="${STEPS:-300}"
EXAMPLES="${EXAMPLES:-2000}"
SEED_START="${SEED_START:-42}"
N_SEEDS="${N_SEEDS:-5}"

policies=(pooling_verified normal_screening reverse_screening)
seeds=()
for ((offset=0; offset<N_SEEDS; offset++)); do
  seeds+=("$((SEED_START + offset))")
done
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
policy="${policies[$((task_id % ${#policies[@]}))]}"
seed="${seeds[$((task_id / ${#policies[@]}))]}"
run_dir="$OUTPUT_ROOT/${policy}_seed${seed}"

python scripts/09_train_dpo_policy.py \
  --config "$CONFIG" \
  --pairs "$PAIRS" \
  --items "$ITEMS" \
  --policy "$policy" \
  --seed "$seed" \
  --max-train-examples "$EXAMPLES" \
  --max-steps "$STEPS" \
  --output-dir "$run_dir"

python scripts/10_evaluate_dpo_policy.py \
  --config "$CONFIG" \
  --test-pairs "$TEST_PAIRS" \
  --adapter "$run_dir/adapter" \
  --label "${NOISE_NAME}_${policy}_seed${seed}" \
  --output "$run_dir/eval.json"
