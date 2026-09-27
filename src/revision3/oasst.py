from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd
from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class FeedbackItem:
    message_id: str
    parent_id: str | None
    tree_id: str
    user_id: str
    text: str
    rank: int
    helpful: int
    quality_label: float | None
    helpfulness_label: float | None
    review_count: int
    sibling_count: int


def _label_value(labels: Any, name: str) -> float | None:
    if not isinstance(labels, dict):
        return None
    names = labels.get("name") or []
    values = labels.get("value") or []
    for label_name, value in zip(names, values):
        if label_name == name:
            return float(value)
    return None


def load_oasst(config: dict[str, Any]) -> Dataset:
    dataset_config = config["dataset"]
    dataset = load_dataset(
        dataset_config["name"],
        split=dataset_config["split"],
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected a single Hugging Face Dataset split.")
    return dataset


def extract_feedback_items(dataset: Dataset, config: dict[str, Any]) -> pd.DataFrame:
    language = config["dataset"].get("language", "en")
    min_review_count = int(config["dataset"].get("min_review_count", 2))

    sibling_counts: Counter[str] = Counter()
    for row in dataset:
        if (
            row["role"] == "assistant"
            and row["parent_id"] is not None
            and not row["deleted"]
            and not row["synthetic"]
            and row["rank"] is not None
        ):
            sibling_counts[str(row["parent_id"])] += 1

    records: list[dict[str, Any]] = []
    for row in dataset:
        if row["role"] != "assistant":
            continue
        if language and row["lang"] != language:
            continue
        if row["deleted"] or row["synthetic"] or row["rank"] is None:
            continue
        if not row["review_result"]:
            continue
        if int(row["review_count"] or 0) < min_review_count:
            continue
        text = str(row["text"] or "").strip()
        if not text:
            continue

        rank = int(row["rank"])
        item = FeedbackItem(
            message_id=str(row["message_id"]),
            parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
            tree_id=str(row["message_tree_id"]),
            user_id=str(row["user_id"]),
            text=text,
            rank=rank,
            helpful=int(rank == 0),
            quality_label=_label_value(row["labels"], "quality"),
            helpfulness_label=_label_value(row["labels"], "helpfulness"),
            review_count=int(row["review_count"] or 0),
            sibling_count=sibling_counts[str(row["parent_id"])],
        )
        records.append(item.__dict__)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No eligible OpenAssistant feedback items were found.")
    frame["created_date"] = frame["message_id"].map(
        {str(row["message_id"]): row["created_date"] for row in dataset}
    )
    return frame


def summarize_feedback_items(frame: pd.DataFrame) -> dict[str, Any]:
    contributor_counts = frame.groupby("user_id").size()
    return {
        "n_items": int(len(frame)),
        "n_contributors": int(frame["user_id"].nunique()),
        "n_contributors_ge3": int((contributor_counts >= 3).sum()),
        "items_contributors_ge3": int(contributor_counts[contributor_counts >= 3].sum()),
        "helpful_rate": float(frame["helpful"].mean()),
        "rank_counts": {str(k): int(v) for k, v in frame["rank"].value_counts().sort_index().items()},
    }
