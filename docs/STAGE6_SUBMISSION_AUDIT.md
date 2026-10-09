# Stage 6 — Final Integration and Submission Audit

Status: PASS. Main theoretical results unchanged.

## 1. Manuscript build

- ACM `acmart` anonymous review build.
- Final output: `main.pdf`, 12 pages.
- Main text and conclusion end on page 8.
- Appendix and references occupy pages 9--12.
- No undefined references or citations.
- No overfull `\hbox` warnings.
- No author identity, institutional email, local username, or HPC account string in the compiled PDF.

The introduction was reduced to three contributions, the abstract no longer foregrounds aggregation details or verifier noise, and the volume-proportional result is reported as an inconclusive sensitivity rather than a confirmed volume effect.

## 2. Stage 5 numeric consistency

Equal-size 1.5B results:

| Policy | Accuracy | Delta vs base |
|---|---:|---:|
| Oracle | 71.80% | +1.68 pp |
| Normal screening | 70.89% | +0.76 pp |
| Verified pooling | 70.11% | -0.01 pp |
| Unverified pooling | 70.13% | 0.00 pp |
| Reverse screening | 70.05% | -0.08 pp |

Policy-specific-pool 1.5B results under the same processed-pair budget:

| Policy | Accuracy | Delta vs base |
|---|---:|---:|
| Oracle | 71.31% | +1.19 pp |
| Normal screening | 70.91% | +0.79 pp |
| Verified pooling | 70.11% | -0.01 pp |
| Reverse screening | 70.11% | -0.01 pp |
| Unverified pooling | 69.94% | -0.19 pp |

Critical paired results:

- Equal-size normal minus reverse: `+0.84 pp`, 95% CI `[0.39,1.28]`, 9/10 seeds.
- Policy-specific-pool normal minus reverse: `+0.80 pp`, 95% CI `[0.15,1.45]`, 8/10 seeds.
- Base accuracy: `70.125%`.

## 3. Volume-proportional sensitivity

The Stage 3 array completed 20/20 tasks and 50/50 runs. All processed-pair budgets matched the preregistered manifest.

- Oracle: `73.61%`, `+3.49 pp`, 95% CI `[2.90,4.08]`.
- Normal screening: `70.73%`, `+0.60 pp`, 95% CI `[-0.09,1.29]`.
- Reverse screening: `70.18%`, `+0.05 pp`, 95% CI `[-0.19,0.29]`.
- Normal minus reverse: `+0.55 pp`, 95% CI `[-0.10,1.20]`, 6/10 seeds.

The sensitivity preserves the direction but is not statistically distinguishable. The matched-budget designs remain the primary learning evidence.

## 4. Calibration and boundary robustness

- 80 calibration specifications preserve \(\eta_H>\eta_L\).
- Separation range: `0.193–0.263`.
- Critical-ratio range: `1.369–1.560`.
- Fixed-label boundary interval: `[1.492,1.635]`.
- Type-refit boundary interval: `[1.533,1.705]`.
- Bootstrap unit: contributor-level resampling; repetitions: 100.
- Continuous reliability exhibits the reverse-screening analogue under increasing exposure schedules.

These results are integrated into the main text and Appendix A.8.

## 5. Figure and repository consistency

- Paper figure: `experiment_outputs/fig_scale_comparison.png`.
- Repository figure: `figures/fig_scale_comparison.png`.
- SHA256 of both figures: `042cef666a6980303c86e0f83e78b798ca2520e119853181b8efd572032a9054`.
- `scripts/17_make_paper_figures.py` regenerates the figure from the aggregate results.
- Public repository contains no user ID, personal email, local username, or institutional account string.
- Anonymous repository: `https://anonymous.4open.science/r/llm-feedback-screening-F21A`.
- Test suite: `15 passed`.

## 6. Remaining non-blocking warnings

- BibTeX reports missing publisher/address or volume/page metadata for some bibliography entries. These are formatting warnings, not manuscript failures.
- The author should visually spot-check the final PDF once before submission.

## 7. Gate decision

Pass. The manuscript, Stage 3 volume sensitivity, Stage 5 learning evidence, calibration audit, figure hashes, public artifact, and anonymous mirror are consistent and ready for final author review.
