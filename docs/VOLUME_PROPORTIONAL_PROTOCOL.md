# Volume-Proportional DPO Protocol

Status: preregistered for Stage 3. No GPU jobs are launched in Stage 2.

## Objective

Test whether the normal-versus-reverse screening ordering remains visible when the processed-pair budget is proportional to the policy-specific accepted-pool size.

This design operationalizes accepted volume through training exposure. It does not isolate a pure volume channel independently of pool composition, and it does not identify behavioral responses to sanctions.

## Designs

### Existing fixed-compute design

- Every policy processes 300 optimizer steps.
- Effective batch size: 2.
- Processed pairs per run: 600.
- Interpretation: policy-specific pool sizes and compositions under a matched optimization budget.

### New volume-proportional design

- Accepted pool size \(V_p\) is the Stage 5 simulator target for policy \(p\).
- Processed-pair budget is \(V_p\).
- Optimizer steps: \(\lceil V_p / 2 \rceil\).
- Sampling: accepted pools are constructed with replacement; every pool entry is processed once during the run.
- Interpretation: combined accepted-pool and optimization-exposure effects.

Neither design alone identifies a pure composition effect or a pure volume effect.

## Policy budgets

| Policy | Accepted pool size | Processed pairs | Optimizer steps |
|---|---:|---:|---:|
| Oracle | 4,584 | 4,584 | 2,292 |
| Normal screening | 2,000 | 2,000 | 1,000 |
| Reverse screening | 1,882 | 1,882 | 941 |
| Verified pooling | 3,880 | 3,880 | 1,940 |
| Unverified pooling | 4,584 | 4,584 | 2,292 |

## Fixed training settings

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Seeds: 42–51
- LoRA rank: 16
- Learning rate: `5e-5`
- DPO beta: `0.5`
- Maximum sequence length: 256
- Optimizer: AdamW with constant learning rate; no duration-normalized schedule
- Evaluation: the same 800 clean held-out preference pairs for every run

## Data accounting

The manifest records:

- accepted pool size;
- distinct pair count;
- distinct tree count;
- processed-pair budget;
- effective batch size;
- optimizer steps;
- sampling rule.

The pool size is not the number of distinct pairs because pool construction samples with replacement.

## Assertions

The dry-run verifier checks:

1. Manifest has 50 policy-seed rows.
2. Accepted pool size equals the processed-pair budget.
3. Optimizer steps equal `ceil(processed_pair_budget / effective_batch_size)`.
4. Held-out evaluation has 800 pairs.
5. Pair overlap between training and evaluation is zero.
6. Tree overlap between training and evaluation is zero.

After each GPU run, `scripts/23_validate_volume_metrics.py` checks that realized metrics match the manifest.

## Dry-run result

`pilot_results/volume_proportional/dry_run_report.json` records:

- status: pass;
- 50 manifest rows;
- 800 held-out pairs;
- pair overlap: 0;
- tree overlap: 0.

## Launch and failure rules

- No GPU jobs are launched until the dry-run gate passes.
- Failed runs are identified by missing or invalid metrics, not by performance.
- If the experiment is incomplete or invalid, retain the fixed-compute results and narrow the manuscript claim.
- Null or adverse results are reported and the conclusion is revised accordingly.

## Deliverables for Stage 3

- HPC logs and run metrics;
- policy-level and paired analyses;
- realized processed-pair and compute accounting;
- result tables and figures;
- implementation notes for the manuscript.
