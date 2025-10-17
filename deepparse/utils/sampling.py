"""Deterministic sampling utilities for k log selection."""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Sequence

from .regex_library import classify_token


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def deterministic_sample(logs: Sequence[str], k: int) -> List[str]:
    """Select k diverse logs using token-class fingerprints.

    The algorithm computes a signature based on canonical regex classes, then performs
    deterministic reservoir sampling biased towards unique signatures.
    """

    if k >= len(logs):
        return list(logs)

    buckets = {}
    for idx, line in enumerate(logs):
        tokens = line.split()
        signature = ",".join(filter(None, (classify_token(tok) or tok for tok in tokens[:4])))
        buckets.setdefault(signature, []).append((idx, line))

    selected: List[str] = []
    for signature in sorted(buckets):
        lines = buckets[signature]
        step = max(1, len(lines) // max(1, k // max(1, len(buckets))))
        for idx, line in lines[::step]:
            selected.append(line)
            if len(selected) >= k:
                return selected[:k]

    # fallback: deterministic remainder
    if len(selected) < k:
        remaining = [line for _, line in sorted(((stable_hash(l), l) for l in logs), key=lambda x: x[0])]
        for line in remaining:
            if line not in selected:
                selected.append(line)
            if len(selected) >= k:
                break
    return selected[:k]


def deterministic_indices(logs: Sequence[str], k: int) -> List[int]:
    sample = deterministic_sample(logs, k)
    return [logs.index(line) for line in sample]
