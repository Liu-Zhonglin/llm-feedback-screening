import pandas as pd
import torch

from revision3.dpo_policy import derive_policy_pairs, optimizer_steps_for_examples, processed_batch_size


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


def test_optimizer_steps_for_example_budget() -> None:
    assert optimizer_steps_for_examples(2000, 2) == 1000
    assert optimizer_steps_for_examples(1882, 2) == 941
    assert optimizer_steps_for_examples(2001, 2) == 1001


def test_optimizer_steps_reject_invalid_inputs() -> None:
    import pytest

    with pytest.raises(ValueError):
        optimizer_steps_for_examples(0, 2)
    with pytest.raises(ValueError):
        optimizer_steps_for_examples(10, 0)


def test_processed_batch_size_uses_weight_tensor() -> None:
    batch = {
        "chosen": {"input_ids": torch.zeros((3, 5), dtype=torch.long)},
        "weights": torch.ones(3),
    }
    assert processed_batch_size(batch) == 3
