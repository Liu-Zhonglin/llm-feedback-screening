from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset as HFDataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


class SFTDataset(torch.utils.data.Dataset):
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
            max_length=max(1, self.max_length // 2),
        )["input_ids"]
        response_ids = self.tokenizer(
            str(row["response"]),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        response_ids = response_ids[: self.max_length - len(prompt_ids)]
        if not response_ids:
            raise ValueError("Empty response after tokenization.")
        return {
            "input_ids": prompt_ids + response_ids,
            "labels": [-100] * len(prompt_ids) + response_ids,
            "weight": float(row.get("weight", 1.0)),
        }


class SFTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        labels = []
        attention_mask = []
        weights = []
        for example in examples:
            padding = max_length - len(example["input_ids"])
            input_ids.append([self.pad_token_id] * padding + example["input_ids"])
            labels.append([-100] * padding + example["labels"])
            attention_mask.append([0] * padding + [1] * len(example["input_ids"]))
            weights.append(example["weight"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
        }


def weighted_causal_loss(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    mask = labels.ne(-100)
    per_example = (token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1)
    return (per_example * batch["weights"]).mean() / batch["weights"].mean().clamp_min(1e-8)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(name, torch.float32)


def build_model_and_tokenizer(config: dict[str, Any]) -> tuple[Any, Any, torch.device]:
    model_config = config["model"]
    model_name = str(model_config["name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = resolve_device()
    default_attention = "sdpa" if device.type == "cuda" else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=str(model_config.get("attn_implementation", default_attention)),
        dtype=resolve_dtype(str(model_config.get("dtype", "float32"))),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_dropout=float(model_config["lora_dropout"]),
        target_modules=list(model_config["target_modules"]),
    )
    model = get_peft_model(model, lora)
    if bool(model_config.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    return model, tokenizer, device


def train_sft(
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
    dataset = SFTDataset(train_frame, tokenizer, int(config["model"]["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["model"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        collate_fn=SFTCollator(int(tokenizer.pad_token_id)),
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["model"]["learning_rate"]),
    )
    model.train()
    history = []
    step = 0
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            loss = weighted_causal_loss(model, batch)
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


def load_sft_model(adapter: str | Path, config: dict[str, Any]) -> tuple[Any, Any, torch.device]:
    model, tokenizer, device = build_model_and_tokenizer(config)
    if str(adapter) != "base":
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict

        adapter_model = load_file(Path(adapter) / "adapter_model.safetensors", device="cpu")
        set_peft_model_state_dict(model, adapter_model)
    model.to(device)
    return model, tokenizer, device


@torch.no_grad()
def evaluate_sft(
    model: Any,
    tokenizer: Any,
    frame: Any,
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    dataset = SFTDataset(frame, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=SFTCollator(int(tokenizer.pad_token_id)),
    )
    losses = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        losses.append(float(loss.detach().cpu()))
    return {
        "n_examples": int(len(frame)),
        "response_nll": float(np.mean(losses)) if losses else float("nan"),
    }
