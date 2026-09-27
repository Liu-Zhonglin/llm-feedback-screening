# HPC Reproduction

The final experiments used the HKU HPC 2021 cluster and L40S GPU nodes.

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

The confirmation script runs seeds 42-44. The same array mechanism can be used for seeds 45-51 with:

```bash
sbatch --array=0-24%20 \
  --export=ALL,CONFIG=configs/hpc_dpo_confirm_1p5b.yaml,OUTPUT_ROOT=results/raw/hpc_dpo_confirm_1p5b_300,STEPS=300,EXAMPLES=2000,SEED_START=45,N_SEEDS=5 \
  scripts/run_hpc_dpo_scale.sbatch
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
