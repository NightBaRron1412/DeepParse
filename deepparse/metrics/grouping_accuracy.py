"""Grouping accuracy metric."""
from __future__ import annotations

from typing import Sequence


def grouping_accuracy(true_group_ids: Sequence[str], predicted_group_ids: Sequence[str]) -> float:
    if len(true_group_ids) != len(predicted_group_ids):
        raise ValueError("Mismatched lengths for GA computation")
    if not true_group_ids:
        return 0.0
    correct = sum(1 for a, b in zip(true_group_ids, predicted_group_ids) if a == b)
    return correct / len(true_group_ids)
