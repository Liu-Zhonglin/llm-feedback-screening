# HPC Reproduction

The final experiments used an institutional HPC cluster with L40S GPU nodes.

## Environment

```bash
conda create -n feedback-repro python=3.11
conda activate feedback-repro
pip install -e .
```

Alternatively, create a virtual environment with `--system-site-packages` from a CUDA-enabled PyTorch environment.

Download the models on a login node before submitting offline compute jobs:

```bash
export HF_HOME="$PWD/.cache/huggingface"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
snapshot_download("Qwen/Qwen2.5-1.5B-Instruct")
PY
```

## 0.5B scaling matrix

```bash
sbatch --array=0-24%20 \
  --export=ALL,CONFIG=configs/hpc_dpo_smoke.yaml,OUTPUT_ROOT=results/raw/hpc_dpo_scale_300,STEPS=300,EXAMPLES=2000,SEED_START=42,N_SEEDS=5 \
  scripts/run_hpc_dpo_scale.sbatch

sbatch --array=0-24%20 \
  --export=ALL,CONFIG=configs/hpc_dpo_smoke.yaml,OUTPUT_ROOT=results/raw/hpc_dpo_scale_300,STEPS=300,EXAMPLES=2000,SEED_START=47,N_SEEDS=5 \
  scripts/run_hpc_dpo_scale.sbatch
```

## 1.5B confirmation

```bash
sbatch scripts/run_hpc_dpo_confirm_1p5b.sbatch
```

The confirmation script runs the ten seeds 42-51 used in the paper.

## Stage 5: equal-size and endogenous-volume confirmation

The final confirmation evaluates the same five policies under two accepted-pool designs:

- equal-size: 2,000 accepted pairs for every policy;
- endogenous-volume: oracle 4,584; normal screening 2,000; reverse screening 1,882; verified pooling 3,880; unverified pooling 4,584.

The combined runner executes both designs for each policy-seed pair. It maps 50 policy-seed combinations across 20 array tasks, with `N_TASKS=20` matching the array size:

```bash
sbatch scripts/run_hpc_stage5_combined.sbatch
```

The combined script requests one GPU per task and uses up to 20 concurrent array tasks. Outputs are written under `runs/stage5/equal_size/` and `runs/stage5/endogenous/`.

If the cluster permits separate submissions, the equivalent per-design runners are:

```bash
sbatch scripts/run_hpc_stage5_equal.sbatch
sbatch scripts/run_hpc_stage5_endogenous.sbatch
```

For a one-time notification after the combined array finishes, create the dependency job after capturing the array ID:

```bash
ARRAY_JOB=$(sbatch --parsable scripts/run_hpc_stage5_combined.sbatch)
sbatch --dependency=afterany:${ARRAY_JOB} scripts/notify_stage5_finished.sbatch
```

## Volume-proportional confirmation

The volume-proportional design processes one accepted-pool entry per policy-specific accepted target. Build the manifest and run the dry-run gate:

```bash
python scripts/22_build_volume_manifest.py \
  --config configs/hpc_dpo_confirm_1p5b.yaml \
  --pairs data/processed/oasst_preference_pairs.csv.gz \
  --items data/processed/oasst_typed_items.csv.gz \
  --stage5-manifest pilot_results/full_chain/stage5_experiment_manifest.csv \
  --test-pairs data/processed/oasst_test_pairs.csv.gz \
  --output pilot_results/volume_proportional/volume_manifest.csv \
  --dry-run-report pilot_results/volume_proportional/dry_run_report.json
```

After the dry run passes, submit the volume-proportional array:

```bash
sbatch scripts/run_hpc_volume_proportional.sbatch
```

## Verification noise

```bash
sbatch --export=ALL,NOISE_NAME=perfect scripts/run_hpc_noise_matrix.sbatch
sbatch --export=ALL,NOISE_NAME=noisy scripts/run_hpc_noise_matrix.sbatch
sbatch --export=ALL,NOISE_NAME=uninformative scripts/run_hpc_noise_matrix.sbatch
```

## Summaries

```bash
python scripts/12_analyze_dpo_matrix.py \
  --summary results/raw/hpc_dpo_scale_300/summary.csv \
  --base-eval results/raw/hpc_base_0p5b/base_eval.json \
  --output-dir results/scale/analysis

python scripts/12_analyze_dpo_matrix.py \
  --summary results/raw/hpc_dpo_confirm_1p5b_300/summary.csv \
  --base-eval results/raw/hpc_dpo_confirm_1p5b_300/base_eval.json \
  --output-dir results/confirm_1p5b/analysis
```

For the Stage 5 accepted-pool designs, aggregate each design and then run the paired analysis:

```bash
python scripts/07_summarize_sft_matrix.py \
  --root runs/stage5/equal_size \
  --output-prefix results/stage5/equal_size_summary

python scripts/07_summarize_sft_matrix.py \
  --root runs/stage5/endogenous \
  --output-prefix results/stage5/endogenous_summary

python scripts/12_analyze_dpo_matrix.py \
  --summary results/stage5/equal_size_summary.csv \
  --base-eval results/raw/hpc_dpo_confirm_1p5b_300/base_eval.json \
  --output-dir results/stage5/equal_size

python scripts/12_analyze_dpo_matrix.py \
  --summary results/stage5/endogenous_summary.csv \
  --base-eval results/raw/hpc_dpo_confirm_1p5b_300/base_eval.json \
  --output-dir results/stage5/endogenous
```
