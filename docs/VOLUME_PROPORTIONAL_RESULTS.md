# Volume-Proportional Results

Status: PASS with a narrowed claim. No pure volume effect is identified.

## Execution

- HPC array: `4245628_[0-19%20]`
- Notification job: `4245629`
- Completed tasks: 20/20, exit code 0
- Completed runs: 50/50, with 50 `metrics.json` and 50 `eval.json` files
- Accepted-pool budgets and processed-pair counts matched the manifest for all 50 runs.

## Results

| Policy | Mean accuracy | Delta vs base | 95% CI |
|---|---:|---:|---:|
| Oracle | 73.61% | +3.49 pp | [2.90, 4.08] |
| Normal screening | 70.73% | +0.60 pp | [-0.09, 1.29] |
| Unverified pooling | 70.23% | +0.10 pp | [-0.37, 0.57] |
| Reverse screening | 70.18% | +0.05 pp | [-0.19, 0.29] |
| Verified pooling | 70.05% | -0.08 pp | [-1.15, 1.00] |

Critical paired comparison:

- Normal screening minus reverse screening: **+0.55 pp**, 95% CI `[-0.10, 1.20]`, normal favored in **6/10 seeds**.
- Oracle minus normal screening: **+2.89 pp**, 95% CI `[2.07, 3.71]`, favored in **10/10 seeds**.
- Oracle minus reverse screening: **+3.44 pp**, 95% CI `[3.01, 3.86]`, favored in **10/10 seeds**.
- Verified minus unverified pooling: **-0.18 pp**, 95% CI `[-1.51, 1.16]`, not distinguishable.

## Interpretation

The volume-proportional design preserves the direction of the normal-versus-reverse comparison, but the paired difference is smaller than in the fixed-compute designs and its confidence interval includes zero. It therefore does not confirm a distinguishable normal-versus-reverse advantage under volume-proportional exposure at ten seeds.

The oracle advantage remains large and statistically distinguishable. This is consistent with a quality-selection story, but it does not establish that accepted volume alone improves the normal/reverse contrast.

The design operationalizes accepted volume through increased training exposure while pool composition varies by policy. It does not identify a pure volume channel or a causal decomposition of volume and composition.

## Manuscript consequence

- Retain the fixed-compute equal-size and policy-specific-pool designs as the primary learning evidence.
- Report the volume-proportional experiment as an operationalization check.
- State explicitly that the volume-proportional normal-versus-reverse difference is directional but not statistically distinguishable.
- Do not use “full volume channel” or “confirmed volume effect” language.

## Gate decision

Pass with a narrowed claim. Proceed to Stage 4 integration of calibration and boundary robustness, and Stage 6 for final manuscript consolidation.
