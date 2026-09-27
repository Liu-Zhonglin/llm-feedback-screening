from datasets import Dataset

from revision3.preferences import build_preference_pairs, deterministic_split
import pandas as pd


def test_build_preference_pairs_from_ranked_siblings() -> None:
    dataset = Dataset.from_list(
        [
            {"message_id": "p", "parent_id": None, "role": "prompter", "text": "Question", "deleted": False, "synthetic": False, "rank": None, "lang": "en", "user_id": "u0", "message_tree_id": "t"},
            {"message_id": "a", "parent_id": "p", "role": "assistant", "text": "Better", "deleted": False, "synthetic": False, "rank": 0, "lang": "en", "user_id": "u1", "message_tree_id": "t"},
            {"message_id": "b", "parent_id": "p", "role": "assistant", "text": "Worse", "deleted": False, "synthetic": False, "rank": 1, "lang": "en", "user_id": "u2", "message_tree_id": "t"},
        ]
    )
    items = pd.DataFrame(
        [
            {"message_id": "a", "type_name": "H"},
            {"message_id": "b", "type_name": "L"},
        ]
    )
    pairs = build_preference_pairs(dataset, items)
    assert len(pairs) == 1
    assert pairs.iloc[0]["chosen"] == "Better"
    assert pairs.iloc[0]["rejected"] == "Worse"
    assert pairs.iloc[0]["chosen_type"] == "H"


def test_deterministic_split_has_no_tree_leakage() -> None:
    pairs = pd.DataFrame(
        {
            "tree_id": [f"t{i // 2}" for i in range(20)],
            "pair_id": [f"p{i}" for i in range(20)],
        }
    )
    train, validation, test = deterministic_split(pairs, seed=1)
    assert set(train["tree_id"]).isdisjoint(set(validation["tree_id"]))
    assert set(train["tree_id"]).isdisjoint(set(test["tree_id"]))
    assert set(validation["tree_id"]).isdisjoint(set(test["tree_id"]))
