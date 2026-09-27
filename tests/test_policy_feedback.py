import pandas as pd

from revision3.preferences import derive_policy_feedback


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


def test_oracle_policy_uses_chosen_responses() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_id": "p", "tree_id": "t", "chosen_type": "L", "prompt": "q", "chosen": "good", "rejected": "bad"},
        ]
    )
    items = pd.DataFrame(
        [
            {"type_name": "H", "helpful": 0.9},
            {"type_name": "L", "helpful": 0.1},
        ]
    )
    feedback = derive_policy_feedback(
        pairs, items, _config(), policy="oracle", seed=1, n_examples=10
    )
    assert (feedback["response"] == "good").all()
    assert (feedback["helpful"] == 1).all()


def test_cross_prompt_harm_uses_other_prompt_response() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_id": "p1", "tree_id": "t1", "chosen_type": "L", "prompt": "q1", "chosen": "good1", "rejected": "bad1"},
            {"pair_id": "p2", "tree_id": "t2", "chosen_type": "L", "prompt": "q2", "chosen": "good2", "rejected": "bad2"},
        ]
    )
    items = pd.DataFrame([{"type_name": "H", "helpful": 1.0}, {"type_name": "L", "helpful": 0.0}])
    feedback = derive_policy_feedback(
        pairs,
        items,
        _config(),
        policy="reverse_screening",
        seed=3,
        n_examples=20,
        harm_mode="cross_prompt",
    )
    assert (feedback["helpful"] == 0).all()
    assert set(feedback["response"]).issubset({"good1", "good2"})
