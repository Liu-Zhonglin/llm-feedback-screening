# Calibration and Boundary Robustness

This document records the robustness evidence integrated into the manuscript.

## Calibration specifications

Across 80 specifications that vary:

- minimum contributor history (3, 5, 10, 20);
- reliability shrinkage;
- Gaussian-mixture covariance structure;
- median and quantile type assignment;
- prompt/tree adjustment;

the ordering \(\eta_H>\eta_L\) is preserved in every specification.

| Specification | Separation \(\eta_H-\eta_L\) | Critical ratio |
|---|---:|---:|
| 80-specification range | 0.193–0.263 | 1.369–1.560 |
| Minimum history 3 | 0.251–0.263 | 1.475–1.560 |
| Minimum history 5 | 0.230–0.246 | 1.440–1.461 |
| Minimum history 10 | 0.213–0.234 | 1.396–1.420 |
| Minimum history 20 | 0.193–0.209 | 1.369–1.386 |
| Tree-level leave-one-out | 0.257 | 1.559 |
| Contributor-fold transfer | 0.272 | — |
| Held-out-tree transfer | 0.147 | — |

## Boundary uncertainty

The critical exposure ratio is:

\[
r^*=\frac{1-\eta_L}{1-\eta_H}.
\]

A contributor-level bootstrap with 100 resamples gives:

| Bootstrap unit | 95% interval for \(r^*\) |
|---|---:|
| Fixed estimated types | [1.492, 1.635] |
| Mixture refit on each resample | [1.533, 1.705] |

The type-refit interval is the conservative bound because it includes type-assignment uncertainty. Neither interval identifies the unobserved exposure ratio.

## Continuous reliability

Replacing the binary type with the empirical reliability distribution gives the continuous analogue:

- constant or decreasing exposure schedules select higher-quality contributors;
- increasing exposure schedules select lower-quality contributors.

The full grid is in `pilot_results/boundary_sensitivity/continuous_sensitivity.csv`.

## Scope

These checks support the reliability calibration and the qualitative reverse-screening mechanism. They do not identify contributors' behavioral responses to sanctions or the unobserved exposure schedule.
