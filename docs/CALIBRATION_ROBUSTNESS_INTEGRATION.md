# Stage 4 — Calibration and Boundary Robustness Integration

Status: PASS. Main theoretical results unchanged.

## 1. Artifact verification

Verified against the saved Stage 2–3 outputs:

- 80 calibration specifications;
- separation range 0.193–0.263;
- critical-ratio range 1.369–1.560;
- minimum-history thresholds 3, 5, 10, 20;
- tree-level leave-one-out;
- contributor-fold transfer;
- held-out-tree transfer;
- continuous-reliability sensitivity;
- fixed-label and type-refit boundary bootstrap.

The stored bootstrap CSVs contain 100 contributor-level resamples. The calibration and boundary scripts now default to 100 resamples so that a default rerun reproduces the stored intervals.

## 2. Main-text integration

Added a compact calibration-robustness paragraph to the calibration subsection:

- ordering preserved in every one of 80 specifications;
- separation range 0.193–0.263;
- tree-level leave-one-out essentially unchanged;
- contributor-fold transfer separation 0.272;
- held-out-tree transfer separation 0.147;
- calibration supports reliability but does not identify behavioral sanction responses.

Added boundary uncertainty to the mechanism simulation:

- point estimate \(r^*=1.56\);
- fixed-label 95% interval `[1.492, 1.635]`;
- type-refit 95% interval `[1.533, 1.705]`;
- type-refit interval includes mixture re-estimation and is conservative;
- neither interval identifies the unobserved exposure ratio.

## 3. Appendix integration

Added Appendix Table 2, “Calibration robustness checks,” covering:

- the 80-specification range;
- minimum-history thresholds;
- tree-level leave-one-out;
- contributor-fold transfer;
- held-out-tree transfer;
- fixed-label and type-refit boundary intervals.

Added a continuous-reliability paragraph: increasing exposure selects lower-quality contributors, while constant or decreasing exposure selects higher-quality contributors.

## 4. Claim discipline

The integration states explicitly that:

- calibration robustness supports \( \eta_H>\eta_L \);
- it does not estimate the exposure schedule \(\phi(\eta)\);
- it does not identify behavioral responses to sanctions;
- the boundary intervals do not identify the true exposure ratio.

## 5. Build audit

- PDF compiles successfully.
- Total length: 12 pages.
- Main text remains self-contained through page 8.
- No undefined references or citations.
- No overfull `\hbox` warnings.

## 6. Gate

Pass. The existing calibration and boundary evidence is now visible in the manuscript and traceable to the saved artifacts.

Next: Stage 5 is the optional mild-corruption check; Stage 6 is final narrative consolidation and submission audit.
