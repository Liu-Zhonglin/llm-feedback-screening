# Stage 5 Confirmation: Equal-Size and Endogenous-Volume Designs

## Execution

- Model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Seeds: 42--51.
- Policies: oracle, normal screening, reverse screening, verified pooling, unverified pooling.
- Equal-size design: 2,000 accepted pairs for every policy.
- Endogenous-volume design: oracle 4,584; normal screening 2,000; reverse screening 1,882; verified pooling 3,880; unverified pooling 4,584.
- Training: 300 DPO steps, maximum sequence length 256, LoRA rank 16, learning rate `5e-5`, beta `0.5`.
- Evaluation: the same 800 clean held-out preference pairs for every run.
- Base accuracy: 70.125%.

## Equal-Size Results

| Policy | Mean accuracy | Delta vs base (pp) | 95% CI | Accepted helpful share |
|---|---:|---:|---:|---:|
| Oracle | 71.80% | +1.675 | [1.205, 2.145] | 1.000 |
| Normal screening | 70.89% | +0.762 | [0.285, 1.240] | 0.597 |
| Unverified pooling | 70.13% | 0.000 | [-0.179, 0.179] | 0.409 |
| Verified pooling | 70.11% | -0.013 | [-0.189, 0.164] | 0.469 |
| Reverse screening | 70.05% | -0.075 | [-0.216, 0.066] | 0.335 |

Paired comparisons:

| Comparison | Difference (pp) | 95% CI | Seeds favoring first policy |
|---|---:|---:|---:|
| Normal screening vs. reverse screening | +0.838 | [0.393, 1.282] | 9/10 |
| Oracle vs. reverse screening | +1.750 | [1.290, 2.210] | 10/10 |
| Oracle vs. normal screening | +0.912 | [0.136, 1.689] | 7/10 |
| Verified vs. unverified pooling | -0.013 | [-0.274, 0.249] | 4/10 |

## Endogenous-Volume Results

| Policy | Target pairs | Mean accuracy | Delta vs base (pp) | 95% CI | Accepted helpful share |
|---|---:|---:|---:|---:|---:|
| Oracle | 4,584 | 71.31% | +1.188 | [0.338, 2.037] | 1.000 |
| Normal screening | 2,000 | 70.91% | +0.788 | [0.179, 1.396] | 0.597 |
| Verified pooling | 3,880 | 70.11% | -0.013 | [-0.229, 0.204] | 0.471 |
| Reverse screening | 1,882 | 70.11% | -0.013 | [-0.249, 0.224] | 0.335 |
| Unverified pooling | 4,584 | 69.94% | -0.188 | [-0.347, -0.028] | 0.405 |

Paired comparisons:

| Comparison | Difference (pp) | 95% CI | Seeds favoring first policy |
|---|---:|---:|---:|
| Normal screening vs. reverse screening | +0.800 | [0.147, 1.453] | 8/10 |
| Oracle vs. reverse screening | +1.200 | [0.337, 2.063] | 9/10 |
| Oracle vs. normal screening | +0.400 | [-0.951, 1.751] | 8/10 |
| Verified vs. unverified pooling | +0.175 | [-0.112, 0.462] | 5/10 |

## Interpretation

The mechanism direction survives both designs. Oracle and normal screening improve held-out preference accuracy over the same-scale base model, whereas reverse screening does not. Normal screening beats reverse screening by about 0.8 percentage points with a 95% confidence interval excluding zero in both designs. The result therefore does not depend only on an equal accepted-volume constraint or only on composition.

The experiment remains semi-synthetic. Participation, verification, sanctions, and accepted volume are simulated; harmful feedback is represented by preference inversion; and the model scale is below a production system. The result supports the mechanism direction rather than deployment performance.

## Reproducibility

See `docs/HPC.md` for the Stage 5 submission commands and `scripts/07_summarize_sft_matrix.py` plus `scripts/12_analyze_dpo_matrix.py` for aggregation.
