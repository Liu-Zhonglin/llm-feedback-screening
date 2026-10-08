# Volume-Proportional Experiment: Stage 2 Dry-Run Report

Status: PASS. No GPU jobs were launched.

## Protocol summary

- Existing fixed-compute design: 300 optimizer steps x batch size 2 = 600 processed pairs.
- New volume-proportional design: processed-pair budget = policy-specific accepted-pool size.
- Optimizer steps = `ceil(processed_pair_budget / effective_batch_size)`.
- Accepted pools are constructed with replacement.
- Seeds: 42–51.
- Evaluation: the same 800 held-out preference pairs.

## Budgets

| Policy | Accepted pool size | Processed pairs | Optimizer steps |
|---|---:|---:|---:|
| Oracle | 4,584 | 4,584 | 2,292 |
| Normal screening | 2,000 | 2,000 | 1,000 |
| Reverse screening | 1,882 | 1,882 | 941 |
| Verified pooling | 3,880 | 3,880 | 1,940 |
| Unverified pooling | 4,584 | 4,584 | 2,292 |

## Code changes

- `src/revision3/dpo_policy.py`: example-budget training path and step conversion.
- `scripts/09_train_dpo_policy.py`: `--max-examples` support and processed-exposure metrics.
- `scripts/22_build_volume_manifest.py`: manifest generation and dry-run assertions.
- `scripts/23_validate_volume_metrics.py`: realized-run validation against the manifest.
- `scripts/run_hpc_volume_proportional.sh`: volume-proportional HPC runner with `DRY_RUN` support.
- `scripts/run_hpc_volume_proportional.sbatch`: 20-task, up-to-20-GPU SLURM array.
- `tests/test_dpo_policy.py`: budget arithmetic tests.

## Dry-run checks

Passed locally and on HPC:

- 50 policy-seed manifest rows;
- accepted pool size equals processed-pair budget;
- optimizer steps match the batch-size conversion;
- held-out evaluation contains 800 pairs;
- pair overlap with training: 0;
- tree overlap with training: 0;
- runner dry-run resolves policy, seed, pool size, and budget correctly;
- synthetic metric validation passes for a representative manifest row.

## Test result

- `14 passed` in both the reproducibility clone and the local `Codes/Revision3` project.

## Gate decision

Pass. The manifest and runner are ready for Stage 3 HPC launch, but no GPU jobs have been submitted.
