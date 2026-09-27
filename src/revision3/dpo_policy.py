from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from revision3.dpo_eval import DPODataCollator, PreferenceDataset, move_batch, sequence_log_prob
from revision3.sft import build_model_and_tokenizer


POLICY_TYPE_MODE = {
    "oracle": "mixed",
    "normal_screening": "H",
    "reverse_screening": "L",
    "pooling_verified": "mixed",
    "pooling_unverified": "mixed",
}


def derive_policy_pairs(
    pairs: Any,
    item_table: Any,
    config: dict[str, Any],
    *,
    policy: str,
    seed: int,
    n_examples: int,
) -> Any:
    if policy not in POLICY_TYPE_MODE:
        raise ValueError(f"Unknown policy: {policy}")
    mechanism = config["mechanism"]
    helper = (
        item_table.dropna(subset=["type_name"])
        .groupby("type_name", as_index=False)["helpful"]
        .mean()
        .set_index("type_name")["helpful"]
        .astype(float)
        .to_dict()
    )
    if {"H", "L"} - set(helper):
        raise ValueError("Missing empirical helpfulness for H or L.")

    usable = pairs.dropna(subset=["chosen_type"]).copy()
    usable = usable.loc[usable["chosen_type"].isin(["H", "L"])].reset_index(drop=True)
    if usable.empty:
        raise ValueError("No classified preference pairs are available.")
    by_type = {
        type_name: usable.loc[usable["chosen_type"] == type_name].reset_index(drop=True)
        for type_name in ("H", "L")
    }

    type_mode = POLICY_TYPE_MODE[policy]
    verified_policy = policy in {"normal_screening", "reverse_screening", "pooling_verified"}
    rho = float(mechanism["verification_rate"]) if verified_policy else 0.0
    tpr = float(mechanism["verifier_tpr"])
    fpr = float(mechanism["verifier_fpr"])
    harmful_weight = float(mechanism["unverified_harmful_weight"])
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(1000, n_examples * 100)

    while len(rows) < n_examples and attempts < max_attempts:
        attempts += 1
        type_name = (
            "H" if rng.random() < float(mechanism["lambda_high"]) else "L"
        ) if type_mode == "mixed" else type_mode
        pool = by_type[type_name]
        if pool.empty:
            continue
        pair = pool.iloc[int(rng.integers(0, len(pool)))]
        helpful = True if policy == "oracle" else bool(rng.random() < helper[type_name])
        verified = bool(rng.random() < rho)
        if verified and helpful and rng.random() < fpr:
            continue
        if verified and not helpful and rng.random() < tpr:
            continue

        if helpful:
            chosen = str(pair["chosen"])
            rejected = str(pair["rejected"])
            weight = 1.0
        else:
            chosen = str(pair["rejected"])
            rejected = str(pair["chosen"])
            weight = 1.0 if verified else harmful_weight
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "tree_id": pair["tree_id"],
                "policy": policy,
                "type_name": type_name,
                "helpful": int(helpful),
                "verified": int(verified),
                "weight": weight,
                "prompt": str(pair["prompt"]),
                "chosen": chosen,
                "rejected": rejected,
                "correct_chosen": str(pair["chosen"]),
                "correct_rejected": str(pair["rejected"]),
            }
        )
    if len(rows) < n_examples:
        raise RuntimeError(
            f"Generated only {len(rows)} pairs for {policy}; requested {n_examples}."
        )
    return __import__("pandas").DataFrame.from_records(rows)


def train_dpo_policy(
    train_frame: Any,
    config: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, tokenizer, device = build_model_and_tokenizer(config)
    model.config.use_cache = False
    dataset = PreferenceDataset(train_frame, tokenizer, int(config["model"]["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["model"].get("num_workers", 0)),
        collate_fn=DPODataCollator(int(tokenizer.pad_token_id)),
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["model"]["learning_rate"]),
    )
    beta = float(config["model"]["beta"])
    model.train()
    step = 0
    history = []
    while step < max_steps:
        for batch in loader:
            batch = move_batch(batch, device)
            chosen = sequence_log_prob(model, batch["chosen"], normalize=True)
            rejected = sequence_log_prob(model, batch["rejected"], normalize=True)
            with torch.no_grad(), model.disable_adapter():
                ref_chosen = sequence_log_prob(model, batch["chosen"], normalize=True)
                ref_rejected = sequence_log_prob(model, batch["rejected"], normalize=True)
            margin = (chosen - rejected) - (ref_chosen - ref_rejected)
            weights = batch["weights"]
            per_example = -F.logsigmoid(beta * margin)
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if step % 10 == 0:
                history.append({"step": step, "loss": float(loss.detach().cpu())})
                print(f"step={step} loss={float(loss.detach().cpu()):.4f}", flush=True)
            step += 1
            if step >= max_steps:
                break
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    return {
        "device": str(device),
        "seed": seed,
        "max_steps": max_steps,
        "train_examples": len(train_frame),
        "history": history,
    }
