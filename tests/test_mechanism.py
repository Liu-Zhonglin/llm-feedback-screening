import pandas as pd

from revision3.mechanism import build_policies, participation_threshold


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
