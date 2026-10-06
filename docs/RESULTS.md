# Experiment Results

This document records the complete experimental progression, including early pilot runs and the final HPC results.

Date: 2026-09-22

## 1. Contributor-quality estimation

Source: OpenAssistant `oasst1`, English assistant messages, not deleted, not synthetic, reviewed, and ranked.

Pilot sample after filtering:

- 21,944 eligible feedback items;
- 3,354 contributors;
- 1,228 contributors with at least three items;
- 19,303 items with a contributor type.

Type-specific probability that an item is ranked best (`rank == 0`):

- `eta_H = 0.540` with 95% clustered bootstrap interval `[0.523, 0.566]`;
- `eta_L = 0.282` with 95% clustered bootstrap interval `[0.268, 0.297]`.

Type counts:

- high reliability: 343 contributors;
- low reliability: 885 contributors.

The separation is large and estimated from real repeated contributors. It supports using OpenAssistant for the pilot, while SHP or HH-RLHF can provide scale and robustness later.

## 2. Independent screening simulation

For the pilot configuration `rho = 0.30`, `P = 0.25`, `phi_H = 0.30`, `phi_L = 0.80`, and `eta_H/eta_L` taken from the data:

| Policy | Accepted helpful share | Platform payoff |
|---|---:|---:|
| Normal screening | 0.601 | 4,765 |
| Pooling, no verification | 0.411 | -1,908 |
| Pooling with verification | 0.471 | 5,695 |

Interpretation:

- Normal screening raises the quality of accepted feedback.
- Pooling with verification yields higher total payoff in this calibration because it retains both types and verification removes much of the harmful contamination.
- This is a useful, theory-consistent trade-off: screening quality and total platform value are different objectives.
- The result is still conditional on the cost and sanction parameters. It is not a substitute for the eventual training experiment.

## 3. Mac DPO smoke test

Hardware: M1 Pro, 16 GB, PyTorch MPS.

Model: `distilgpt2` with LoRA (`r = 8`, `alpha = 16`, target module `c_attn`).

The following runs completed:

- 20-step pooling smoke run;
- 50-step high-reliability-only run;
- 50-step low-reliability-only run;
- 150-step clean-label run;
- 150-step 60%-corrupted-label run;
- 75-step length-normalized clean-label run.

The initial high-only and low-only runs reported 62.7% versus 55.0% on different policy-specific test subsets. Those subsets are not comparable. On a common test set, the adapters did not outperform the base model in a stable way.

Common-test pilot accuracy:

| Model | Accuracy | Mean normalized margin |
|---|---:|---:|
| Base `distilgpt2` | 57.7% | 0.685 |
| High-only, 50 steps | 57.7% | 0.688 |
| Low-only, 50 steps | 57.7% | 0.682 |
| Clean labels, 150 steps | 57.3% | 0.668 |
| 60% corrupted labels, 150 steps | 57.3% | 0.692 |
| Length-normalized clean labels, DPO, 75 steps | 57.7% | 0.694 |

Conclusion: the infrastructure works, but these runs are **not evidence** for or against the mechanism. The training budget is too small, the noisy-label construction is provisional, and the evaluation shows that the pilot harness must be strengthened before a substantive claim is made.

## 4. Technical issues found and fixed

- Hugging Face access must run outside the Codex sandbox because the dataset and model use the user's cache.
- MPS does not support scaled dot-product attention with dropout; the training script uses eager attention.
- Matplotlib can abort under the sandboxed Python process when it initializes the macOS GUI stack; scripts force the `Agg` backend and local cache directories.
- Context construction used the full OpenAssistant tree and the derived pairs are split by `tree_id` to avoid leakage.
- Standard summed DPO was dominated by response length in the pilot; the current training objective uses length-normalized log probabilities.

## 5. Required next steps

1. Replace the provisional corruption process with a policy-derived noisy-feedback generator based on `eta_H`, `eta_L`, and verification errors.
2. Train high-only, low-only, normal screening, reverse screening, pooling, and verified pooling for the same token budget and at least three seeds.
3. Evaluate every adapter on one common held-out set.
4. Increase the training budget until the clean-label adapter improves over the base checkpoint; otherwise the checkpoint/objective is not informative.
5. Add a second open model or dataset before making any WWW-level claim.


## 6. Mechanism-derived equal-budget SFT pilot

The next pilot replaced the ad hoc corruption path with policy-derived feedback:

- OpenAssistant preference pairs provide prompts and candidate responses.
- Contributor type determines the empirical helpful probability.
- Harmful feedback uses a controlled stress condition: 60% of response tokens are corrupted.
- Verification removes detected harmful feedback and false positives.
- Unverified harmful feedback receives weight 0.25.
- Every policy receives the same 600 training examples and 50 optimizer steps.
- Every policy is evaluated on the same clean 800-pair held-out set.

The controlled corruption condition is deliberately severe and is not presented as a realistic estimate of deployed feedback.

Three-seed results for the key policies:

| Policy | Seeds | Mean NLL | SD | Mean preference accuracy |
|---|---:|---:|---:|---:|
| Oracle | 3 | 3.505 | 0.023 | 61.38% |
| Reverse screening | 3 | 3.577 | 0.048 | 60.83% |
| Normal screening | 3 | 3.595 | 0.014 | 60.54% |
| Verified pooling | 1 | 3.631 | - | 60.63% |
| Unverified pooling | 1 | 3.664 | - | 60.38% |

Base model reference: NLL `3.960`, preference accuracy `59.25%`.

Interpretation:

- Oracle feedback gives the best mean NLL, consistent with the intended direction.
- The separation is small and not stable enough: reverse screening is better than normal screening on average.
- The pilot demonstrates infrastructure and sensitivity, not a publishable mechanism effect.
- The model is still learning general answer style, which may dominate response-quality differences at this small training budget.

The next empirical iteration should use a more realistic harmful-feedback process and a model/training budget large enough to move preference behavior, not just language-model NLL.


## 7. HPC DPO pilot

The first GPU-backed pilot used `Qwen/Qwen2.5-0.5B-Instruct` on institutional HPC L40S nodes. Each policy received 2,000 simulated feedback pairs and 300 normalized DPO steps. Harmful feedback inverted the observed preference pair, verification used `rho = 0.30`, `TPR = 0.80`, and `FPR = 0.10`, and unverified harmful feedback received weight 0.25.

Three seeds were run for each policy. Every adapter was evaluated on the same clean held-out preference set.

| Policy | Mean accuracy | SD across seeds | Delta vs base |
|---|---:|---:|---:|
| Oracle | 71.79% | 0.95 pp | +2.04 pp |
| Normal screening | 70.75% | 0.88 pp | +1.00 pp |
| Verified pooling | 70.17% | 0.26 pp | +0.42 pp |
| Base model | 69.75% | - | 0.00 pp |
| Unverified pooling | 69.50% | 0.45 pp | -0.25 pp |
| Reverse screening | 69.50% | 0.22 pp | -0.25 pp |

The ordering matches the intended mechanism direction:

- oracle-quality feedback performs best;
- normal screening improves held-out preference accuracy;
- verification improves pooling relative to unverified pooling;
- reverse screening does not improve the model and is slightly worse than base.

The effect is modest but consistent enough to justify scaling. It is not yet evidence for deployed platforms: participation and sanctions remain simulated, harmful feedback is generated by preference inversion, and the model is 0.5B rather than a production-scale LLM.

Artifacts:

- `results/scale/summary_summary.csv`
- `results/scale/analysis/paired_comparisons.csv`
- `figures/fig_scale_comparison.png`


## 8. Ten-seed HPC scaling and verification-noise ablation

The DPO pilot was scaled to ten seeds on institutional HPC L40S nodes using up to 20 concurrent array tasks.

| Policy | Mean accuracy | SD across seeds | Mean delta vs base | 95% CI for delta |
|---|---:|---:|---:|---:|
| Oracle | 71.94% | 0.80 pp | +2.19 pp | +1.61 to +2.76 pp |
| Normal screening | 70.90% | 0.73 pp | +1.15 pp | +0.63 to +1.67 pp |
| Verified pooling | 69.99% | 0.49 pp | +0.24 pp | -0.12 to +0.59 pp |
| Unverified pooling | 69.74% | 0.30 pp | -0.01 pp | -0.23 to +0.20 pp |
| Reverse screening | 69.29% | 0.22 pp | -0.46 pp | -0.62 to -0.30 pp |
| Base model | 69.75% | - | 0.00 pp | - |

Paired comparisons:

- Oracle beats normal screening by 1.04 pp on average in 9 of 10 seeds.
- Normal screening beats reverse screening by 1.61 pp in 10 of 10 seeds.
- Oracle beats reverse screening by 2.65 pp in 10 of 10 seeds.

Verification-noise ablation with five seeds per setting:

| Noise setting | Normal screening | Verified pooling | Reverse screening |
|---|---:|---:|---:|
| Perfect (TPR 1.0, FPR 0.0) | 70.38% | 70.50% | 69.75% |
| Standard (TPR 0.8, FPR 0.1) | 70.70% | 70.03% | 69.35% |
| Noisy (TPR 0.6, FPR 0.2) | 70.55% | 69.35% | 69.23% |
| Uninformative (TPR 0.5, FPR 0.5) | 70.33% | 69.68% | 68.83% |

Normal screening is the most robust policy across verifier quality. Verified pooling is competitive under perfect verification but degrades under noisy or uninformative verification. Reverse screening is consistently weakest.


## 9. Ten-seed 1.5B confirmation

The confirmation run used `Qwen/Qwen2.5-1.5B-Instruct`, the same five policies, 2,000 feedback pairs, 300 normalized DPO steps, and seeds 42-51. The same-scale base model achieved 70.125%.

| Policy | Mean accuracy | SD | Mean delta vs base | 95% CI for delta |
|---|---:|---:|---:|---:|
| Oracle | 71.39% | 0.85 pp | +1.26 pp | +0.66 to +1.87 pp |
| Normal screening | 70.86% | 0.69 pp | +0.74 pp | +0.24 to +1.23 pp |
| Verified pooling | 70.15% | 0.21 pp | +0.03 pp | -0.13 to +0.18 pp |
| Unverified pooling | 70.13% | 0.26 pp | 0.00 pp | -0.19 to +0.19 pp |
| Reverse screening | 70.09% | 0.17 pp | -0.04 pp | -0.16 to +0.08 pp |
| Base | 70.125% | - | 0.00 pp | - |

Paired comparisons:

- Oracle beats reverse screening by 1.30 pp in 10/10 seeds.
- Normal screening beats reverse screening by 0.78 pp in 9/10 seeds.
- Oracle beats normal screening by 0.53 pp in 7/10 seeds, with a confidence interval crossing zero.

The 1.5B confirmation supports the directional predictions, but the effect size shrinks relative to the 0.5B run. This is important for the paper: the mechanism ordering replicates, while claims about magnitude should be model- and calibration-dependent.

Scale comparison:

| Policy | 0.5B delta vs base | 1.5B delta vs base |
|---|---:|---:|
| Oracle | +2.19 pp | +1.26 pp |
| Normal screening | +1.15 pp | +0.74 pp |
| Verified pooling | +0.24 pp | +0.03 pp |
| Unverified pooling | -0.01 pp | 0.00 pp |
| Reverse screening | -0.46 pp | -0.04 pp |

Artifacts:

- `results/confirm_1p5b/analysis/policy_summary.csv`
- `results/confirm_1p5b/analysis/paired_comparisons.csv`
- `results/scale/scale_delta_comparison.csv`
- `figures/fig_scale_comparison.png`
