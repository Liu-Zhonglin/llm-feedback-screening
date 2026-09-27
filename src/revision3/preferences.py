from __future__ import annotations

from collections import defaultdict

import numpy as np
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset


ROLE_PREFIX = {
    "prompter": "User",
    "assistant": "Assistant",
}


def build_prompt_lookup(dataset: Dataset) -> dict[str, str]:
    by_id = {str(row["message_id"]): row for row in dataset}
    cache: dict[str, str] = {}

    def build(message_id: str | None) -> str:
        if message_id is None:
            return ""
        if message_id in cache:
            return cache[message_id]
        row = by_id[message_id]
        parent = build(str(row["parent_id"])) if row["parent_id"] is not None else ""
        role = ROLE_PREFIX.get(str(row["role"]), str(row["role"]).title())
        current = f"{role}: {str(row['text']).strip()}"
        text = f"{parent}\n\n{current}" if parent else current
        cache[message_id] = text
        return text

    for message_id in by_id:
        build(message_id)
    return cache


def build_preference_pairs(
    dataset: Dataset,
    item_table: pd.DataFrame,
    *,
    max_prompt_chars: int = 4000,
) -> pd.DataFrame:
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        if (
            row["role"] == "assistant"
            and row["parent_id"] is not None
            and not row["deleted"]
            and not row["synthetic"]
            and row["rank"] is not None
            and row["lang"] == "en"
            and str(row["text"]).strip()
        ):
            children[str(row["parent_id"])].append(row)

    type_lookup = (
        item_table.dropna(subset=["type_name"])
        .set_index("message_id")["type_name"]
        .astype(str)
        .to_dict()
    )
    prompt_lookup = build_prompt_lookup(dataset)
    records: list[dict[str, Any]] = []

    for parent_id, candidates in children.items():
        ranked = sorted(candidates, key=lambda row: (int(row["rank"]), str(row["message_id"])))
        if len(ranked) < 2:
            continue
        chosen = ranked[0]
        rejected = ranked[-1]
        if int(chosen["rank"]) == int(rejected["rank"]):
            continue

        prompt = prompt_lookup.get(parent_id, "")
        if not prompt:
            continue
        prompt = prompt[-max_prompt_chars:]
        records.append(
            {
                "pair_id": f"{chosen['message_id']}::{rejected['message_id']}",
                "parent_id": parent_id,
                "tree_id": str(chosen["message_tree_id"]),
                "prompt": prompt,
                "chosen": str(chosen["text"]).strip(),
                "rejected": str(rejected["text"]).strip(),
                "chosen_user_id": str(chosen["user_id"]),
                "rejected_user_id": str(rejected["user_id"]),
                "chosen_type": type_lookup.get(str(chosen["message_id"])),
                "rejected_type": type_lookup.get(str(rejected["message_id"])),
                "rank_gap": int(rejected["rank"]) - int(chosen["rank"]),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No preference pairs could be constructed from OpenAssistant.")
    frame = frame.drop_duplicates("pair_id").reset_index(drop=True)
    return frame


def deterministic_split(
    pairs: pd.DataFrame,
    *,
    seed: int,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trees = pairs["tree_id"].drop_duplicates().to_numpy()
    rng = pd.Series(trees).sample(frac=1.0, random_state=seed).to_numpy()
    n_train = int(len(rng) * train_fraction)
    n_validation = int(len(rng) * validation_fraction)
    train_trees = set(rng[:n_train])
    validation_trees = set(rng[n_train : n_train + n_validation])
    test_trees = set(rng[n_train + n_validation :])
    return (
        pairs.loc[pairs["tree_id"].isin(train_trees)].reset_index(drop=True),
        pairs.loc[pairs["tree_id"].isin(validation_trees)].reset_index(drop=True),
        pairs.loc[pairs["tree_id"].isin(test_trees)].reset_index(drop=True),
    )


POLICY_TYPE_MODE = {
    "oracle": "mixed",
    "normal_screening": "H",
    "reverse_screening": "L",
    "pooling_verified": "mixed",
    "pooling_unverified": "mixed",
}



def corrupt_tokens(text: str, salt: int, rng: np.random.Generator, replace_fraction: float = 0.6) -> str:
    words = text.split()
    if not words:
        return str(salt)
    n_replace = max(1, int(round(len(words) * replace_fraction)))
    positions = rng.choice(len(words), size=min(n_replace, len(words)), replace=False)
    for position in positions:
        words[int(position)] = f"[[{rng.integers(0, 100000)}]]"
    return " ".join(words)


def derive_policy_feedback(
    pairs: pd.DataFrame,
    item_table: pd.DataFrame,
    config: dict[str, Any],
    *,
    policy: str,
    seed: int,
    n_examples: int,
    harm_mode: str = "rejected_response",
) -> pd.DataFrame:
    if policy not in POLICY_TYPE_MODE:
        raise ValueError(f"Unknown policy: {policy}")
    if harm_mode not in {"rejected_response", "cross_prompt", "token_corruption"}:
        raise ValueError(f"Unknown harm mode: {harm_mode}")
    mechanism = config["mechanism"]
    helper = (
        item_table.dropna(subset=["type_name"])
        .groupby("type_name", as_index=False)["helpful"]
        .mean()
        .set_index("type_name")["helpful"]
        .astype(float)
        .to_dict()
    )
    required = {"H", "L"} - set(helper)
    if required:
        raise ValueError(f"Missing empirical helpfulness for types: {sorted(required)}")

    usable = pairs.dropna(subset=["chosen_type"]).copy()
    usable = usable.loc[usable["chosen_type"].isin(["H", "L"])].reset_index(drop=True)
    if usable.empty:
        raise ValueError("No preference pairs have classified contributors.")

    by_type = {
        type_name: usable.loc[usable["chosen_type"] == type_name].reset_index(drop=True)
        for type_name in ("H", "L")
    }
    rng = np.random.default_rng(seed)
    type_mode = POLICY_TYPE_MODE[policy]
    verified_policy = policy in {"normal_screening", "reverse_screening", "pooling_verified"}
    rho = float(mechanism["verification_rate"]) if verified_policy else 0.0
    lambda_high = float(mechanism["lambda_high"])
    tpr = float(mechanism["verifier_tpr"])
    fpr = float(mechanism["verifier_fpr"])
    harmful_weight = float(mechanism["unverified_harmful_weight"])

    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, n_examples * 100)
    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        if type_mode == "mixed":
            type_name = "H" if rng.random() < lambda_high else "L"
        else:
            type_name = type_mode
        pool = by_type[type_name]
        if pool.empty:
            continue
        pair = pool.iloc[int(rng.integers(0, len(pool)))]
        helpful = bool(rng.random() < helper[type_name]) if policy != "oracle" else True
        if helpful:
            target = str(pair["chosen"])
        elif harm_mode == "rejected_response":
            target = str(pair["rejected"])
        elif harm_mode == "cross_prompt":
            other = usable.iloc[int(rng.integers(0, len(usable)))]
            target = str(other["chosen"])
        else:
            target = corrupt_tokens(str(pair["rejected"]), len(usable.columns) + int(rng.integers(0, 1_000_000)), rng)
        verified = bool(rng.random() < rho)
        if verified and helpful and rng.random() < fpr:
            continue
        if verified and not helpful and rng.random() < tpr:
            continue
        weight = 1.0 if helpful or verified else harmful_weight
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "tree_id": pair["tree_id"],
                "policy": policy,
                "harm_mode": harm_mode,
                "type_name": type_name,
                "helpful": int(helpful),
                "verified": int(verified),
                "weight": weight,
                "prompt": str(pair["prompt"]),
                "response": target,
                "source_chosen": str(pair["chosen"]),
                "source_rejected": str(pair["rejected"]),
            }
        )
    if len(rows) < n_examples:
        raise RuntimeError(
            f"Only generated {len(rows)} accepted examples for {policy}; "
            f"requested {n_examples} after {attempts} attempts."
        )
    return pd.DataFrame(rows)
