import numpy as np
import pandas as pd

from revision3.contributors import estimate_contributor_types


def test_estimate_contributor_types_recovers_ordering() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        high = user < 20
        for item in range(6):
            rows.append(
                {
                    "user_id": f"u{user}",
                    "helpful": int(rng.random() < (0.9 if high else 0.2)),
                    "rank": 0 if high else 1,
                    "review_count": 3,
                }
            )
    frame = pd.DataFrame(rows)
    result = estimate_contributor_types(
        frame,
        min_items=3,
        seed=42,
        bootstrap_repetitions=20,
        confidence_level=0.95,
    )
    assert result.summary["eta_H"] > result.summary["eta_L"]
    assert result.summary["n_contributors_classified"] == 40
