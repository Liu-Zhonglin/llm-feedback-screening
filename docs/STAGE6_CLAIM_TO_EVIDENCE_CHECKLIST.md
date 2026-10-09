# Stage 6 Claim-to-Evidence Checklist

| Claim | Evidence | Status |
|---|---|---|
| Normal separation occurs iff \(\phi_H(1-\eta_H)<\phi_L(1-\eta_L)\) | Theorem and proof in `sec/5_results.tex` and Appendix A.2 | Verified, unchanged |
| Reverse screening is the inequality reversal | Theorem and proof; Stage 0 contract | Verified, unchanged |
| The boundary is independent of the quadratic cost form | Participation constraints; Appendix A.6 | Verified, unchanged |
| Calibration yields \(\eta_H>\eta_L\) | Stage 2 calibration audit; Appendix A.8 | Integrated |
| Calibration ordering survives 80 specifications | `calibration_variants.csv`; Appendix Table 2 | Integrated |
| History thresholds, shrinkage, and alternative classifications preserve the ordering | Stage 2 audit; Appendix Table 2 | Integrated |
| Tree-level leave-one-out and transfer checks preserve the ordering | `tree_transfer.json`; `split_generalization.json`; Appendix Table 2 | Integrated |
| Critical boundary uncertainty is reported | Fixed-label `[1.492,1.635]`; type-refit `[1.533,1.705]`; 100 bootstrap resamples | Integrated in main text |
| Continuous reliability yields the reverse-screening analogue | `continuous_sensitivity.csv`; Appendix A.8 | Integrated |
| Equal-size DPO confirmation | Stage 5 analysis; manuscript Table 1 | Verified |
| Policy-specific-pool DPO results under matched compute | Stage 5 analysis; manuscript Table 1 | Verified with corrected terminology |
| Normal minus reverse under matched budgets | Equal-size `+0.84 pp`; policy-specific pool `+0.80 pp`; both intervals exclude zero | Verified |
| Volume-proportional sensitivity | Stage 3 analysis; Appendix A.9 | Direction preserved, `+0.55 pp`, 95% CI `[-0.10,1.20]`, not statistically distinguishable |
| 0.5B scale robustness | Stage 5/appendix Table 3 | Verified |
| Verifier-noise robustness | Appendix A.10 and Figure 4 | Secondary evidence |
| Behavioral identification | Not identified; stated in main text and conclusion | Claim explicitly limited |
| Reproduction | Public artifact and anonymous mirror; processed data, scripts, aggregate results, manifests | Audited |

## Claim discipline

- The matched-budget designs are the primary learning evidence.
- The volume-proportional design is an operationalization sensitivity, not a confirmed causal volume effect.
- Calibration robustness does not identify sanction exposure or contributor behavior.
- The experiment is semi-synthetic and not a deployed-platform evaluation.
