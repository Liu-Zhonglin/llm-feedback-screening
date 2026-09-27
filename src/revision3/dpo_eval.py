from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, frame: Any, tokenizer: Any, max_length: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        prompt_ids = self.tokenizer(
            str(row["prompt"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length // 2,
        )["input_ids"]
        chosen_ids = self.tokenizer(
            str(row["chosen"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        rejected_ids = self.tokenizer(
            str(row["rejected"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        eos = self.tokenizer.eos_token_id
        chosen_ids = (chosen_ids + [eos])[: self.max_length - len(prompt_ids)]
        rejected_ids = (rejected_ids + [eos])[: self.max_length - len(prompt_ids)]
        return {
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
            "weight": float(row.get("weight", 1.0)),
        }


class DPODataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def _collate(self, examples: list[dict[str, Any]], response_key: str) -> dict[str, torch.Tensor]:
        sequences = []
        labels = []
        for example in examples:
            prompt_ids = example["prompt_ids"]
            response_ids = example[response_key]
            sequences.append(prompt_ids + response_ids)
            labels.append([-100] * len(prompt_ids) + response_ids)
        max_length = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        padded_labels = []
        for sequence, label in zip(sequences, labels):
            padding = max_length - len(sequence)
            input_ids.append([self.pad_token_id] * padding + sequence)
            attention_mask.append([0] * padding + [1] * len(sequence))
            padded_labels.append([-100] * padding + label)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "chosen": self._collate(examples, "chosen_ids"),
            "rejected": self._collate(examples, "rejected_ids"),
            "weights": torch.tensor([example["weight"] for example in examples], dtype=torch.float32),
        }


def sequence_log_prob(
    model: Any,
    batch: dict[str, torch.Tensor],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    token_log_probs = F.log_softmax(logits, dim=-1)
    selected = token_log_probs.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    total = (selected * mask).sum(dim=-1)
    if normalize:
        return total / mask.sum(dim=-1).clamp_min(1)
    return total


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {name: tensor.to(device) for name, tensor in value.items()}
        else:
            moved[key] = value.to(device)
    return moved


@torch.no_grad()
def evaluate_frame(
    model: Any,
    frame: Any,
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> tuple[dict[str, float], Any]:
    if frame.empty:
        return (
            {"n_pairs": 0, "accuracy": float("nan"), "mean_margin": float("nan"), "loss": float("nan")},
            frame.assign(chosen_score=[], rejected_score=[], margin=[]),
        )
    dataset = PreferenceDataset(frame, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DPODataCollator(int(tokenizer.pad_token_id)),
    )
    model.eval()
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    margins: list[float] = []
    losses: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        chosen = sequence_log_prob(model, batch["chosen"], normalize=True)
        rejected = sequence_log_prob(model, batch["rejected"], normalize=True)
        margin = chosen - rejected
        chosen_scores.extend(chosen.cpu().tolist())
        rejected_scores.extend(rejected.cpu().tolist())
        margins.extend(margin.cpu().tolist())
        losses.append(float(-F.logsigmoid(margin).mean().cpu()))
    predictions = frame.reset_index(drop=True).copy()
    predictions["chosen_score"] = chosen_scores
    predictions["rejected_score"] = rejected_scores
    predictions["margin"] = margins
    metrics = {
        "n_pairs": int(len(frame)),
        "accuracy": float(np.mean(np.asarray(margins) > 0)),
        "mean_margin": float(np.mean(margins)),
        "mean_chosen_score": float(np.mean(chosen_scores)),
        "mean_rejected_score": float(np.mean(rejected_scores)),
        "loss": float(np.mean(losses)),
    }
    return metrics, predictions
