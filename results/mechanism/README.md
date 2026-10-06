# Calibrated Mechanism Simulation

This directory contains the Stage 2 mechanism-regime simulation used in the paper.

## Reproduce

From the repository root:

```bash
source .venv/bin/activate
python -m pytest -q tests/test_mechanism.py
python scripts/16_plot_mechanism_regime.py
```

## Inputs

- `results/types/summary.json`: calibrated `eta_H`, `eta_L`, contributor counts, and confidence intervals.
- The mechanism parameters are simulated scenario inputs, not estimates from observed platform enforcement.

## Outputs

- `fig_mechanism_regime.png`: paper-ready four-panel regime, participation, intensity, and composition figure.
- `regime_map.csv`: grid over `phi_H` and `phi_L` with normal/reverse labels.
- `exposure_crossing.csv`: participation and composition as high-type exposure crosses the boundary.
- `intensity_scenarios.csv`: participation and composition as `P rho` varies under fixed reward slack.
- `mechanism_regime_summary.json`: parameter values, boundary ratio, and scenario definitions.

## Interpretation boundary

The simulation is a mechanism diagnostic. It shows how the analytical screening boundary maps to participation and accepted-feedback composition under transparent counterfactual inputs. It does not estimate sanction exposure, appeals, enforcement behavior, or deployed-platform costs.
