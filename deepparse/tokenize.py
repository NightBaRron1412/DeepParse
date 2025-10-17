"""Tokenisation utilities aligning with Drain preprocessing."""
from __future__ import annotations

import re
from typing import List, Sequence

from .utils.regex_library import classify_token

TOKEN_SPLIT = re.compile(r"\s+")


def tokenize(line: str) -> List[str]:
    return [tok for tok in TOKEN_SPLIT.split(line.strip()) if tok]


def mask_tokens(tokens: Sequence[str]) -> List[str]:
    masked: List[str] = []
    for token in tokens:
        cls = classify_token(token)
        masked.append(f"<{cls}>" if cls else token)
    return masked
