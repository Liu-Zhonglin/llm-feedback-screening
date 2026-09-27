import pandas as pd

from revision3.dpo_policy import derive_policy_pairs


def _config():
    return {
        "mechanism": {
            "lambda_high": 0.5,
            "verification_rate": 0.0,
            "verifier_tpr": 0.8,
            "verifier_fpr": 0.1,
            "unverified_harmful_weight": 0.25,
        }
    }


def test_reverse_screening_flips_low_type_preferences() -> None:
    pairs = pd.DataFrame(
        [
            {
                "pair_id": "p",
                "tree_id": "t",
                "chosen_type": "L",
                "prompt": "q",
                "chosen": "better",
                "rejected": "worse",
            }
        ]
    )
    items = pd.DataFrame(
        [
            {"type_name": "H", "helpful": 1.0},
            {"type_name": "L", "helpful": 0.0},
        ]
    )
    feedback = derive_policy_pairs(
        pairs,
        items,
        _config(),
        policy="reverse_screening",
        seed=7,
        n_examples=10,
    )
    assert (feedback["helpful"] == 0).all()
    assert (feedback["chosen"] == "worse").all()
    assert (feedback["rejected"] == "better").all()
    assert (feedback["weight"] == 0.25).all()
