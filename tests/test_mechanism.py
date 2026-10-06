import pandas as pd

from revision3.mechanism import (
    boundary_phi_H,
    build_policies,
    expected_accepted_metrics,
    participation_threshold,
    screening_regime,
)


def test_screening_boundary_direction() -> None:
    policies = build_policies(
        eta_H=0.6,
        eta_L=0.3,
        rho=0.5,
        penalty=0.4,
        participation_cost=0.2,
        phi_H=0.3,
        phi_L=0.8,
        strict_slack=0.001,
    )
    names = {policy.name for policy in policies}
    assert "normal_screening" in names
    assert "reverse_screening" not in names


def test_reverse_screening_direction() -> None:
    policies = build_policies(
        eta_H=0.6,
        eta_L=0.3,
        rho=0.5,
        penalty=0.4,
        participation_cost=0.2,
        phi_H=0.9,
        phi_L=0.1,
        strict_slack=0.001,
    )
    names = {policy.name for policy in policies}
    assert "reverse_screening" in names
    assert "normal_screening" not in names


def test_threshold_increases_with_exposure() -> None:
    low = participation_threshold(
        eta=0.5,
        phi=0.2,
        penalty=0.4,
        rho=0.5,
        participation_cost=0.1,
    )
    high = participation_threshold(
        eta=0.5,
        phi=0.8,
        penalty=0.4,
        rho=0.5,
        participation_cost=0.1,
    )
    assert high > low


def test_boundary_crosses_with_high_type_exposure() -> None:
    eta_H = 0.54
    eta_L = 0.28
    phi_L = 0.8
    phi_H_star = boundary_phi_H(eta_H=eta_H, eta_L=eta_L, phi_L=phi_L)
    assert screening_regime(
        eta_H=eta_H,
        eta_L=eta_L,
        phi_H=0.9 * phi_H_star,
        phi_L=phi_L,
    ) == "normal"
    assert screening_regime(
        eta_H=eta_H,
        eta_L=eta_L,
        phi_H=1.1 * phi_H_star,
        phi_L=phi_L,
    ) == "reverse"
    assert screening_regime(
        eta_H=eta_H,
        eta_L=eta_L,
        phi_H=phi_H_star,
        phi_L=phi_L,
    ) == "boundary"


def test_composition_tracks_screening_direction() -> None:
    common = dict(
        eta_H=0.54,
        eta_L=0.28,
        lambda_high=0.5,
        rho=0.3,
        verifier_tpr=0.8,
        verifier_fpr=0.1,
        unverified_harmful_weight=0.25,
    )
    normal_share, normal_helpfulness = expected_accepted_metrics(
        **common,
        high_participation=1.0,
        low_participation=0.0,
    )
    reverse_share, reverse_helpfulness = expected_accepted_metrics(
        **common,
        high_participation=0.0,
        low_participation=1.0,
    )
    pooled_share, pooled_helpfulness = expected_accepted_metrics(
        **common,
        high_participation=1.0,
        low_participation=1.0,
    )
    assert normal_share == 1.0
    assert reverse_share == 0.0
    assert 0.0 < pooled_share < 1.0
    assert normal_helpfulness > pooled_helpfulness > reverse_helpfulness
