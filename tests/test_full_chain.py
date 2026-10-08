import pandas as pd

from revision3.full_chain import build_experiment_manifest, summarize_policy_grid


def test_manifest_has_two_designs():
    summary = pd.DataFrame(
        {
            "policy": ["oracle", "normal_screening", "reverse_screening", "pooling_verified", "pooling_unverified"],
            "participation_rate": [1.0, 0.5, 0.5, 1.0, 1.0],
            "accepted_volume": [1000, 500, 250, 900, 950],
            "accepted_high_share": [0.5, 1.0, 0.0, 0.5, 0.5],
            "accepted_helpful_share": [0.9, 0.6, 0.3, 0.45, 0.4],
        }
    )
    manifest = build_experiment_manifest(summary)
    assert (manifest["equal_size_target"] == 2000).all()
    normal = manifest.loc[manifest.policy == "normal_screening", "endogenous_target"].iloc[0]
    reverse = manifest.loc[manifest.policy == "reverse_screening", "endogenous_target"].iloc[0]
    assert normal == 2000
    assert reverse < normal
