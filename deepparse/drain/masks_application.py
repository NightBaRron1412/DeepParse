"""Deterministic application of mask regexes."""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..masks_types import Mask


class MaskApplier:
    """Apply regex masks to log lines before Drain clustering."""

    def __init__(self, masks: Sequence[Mask]):
        self._compiled = [(mask, re.compile(mask.pattern)) for mask in masks]

    def apply(self, line: str) -> str:
        masked_line = line
        for mask, compiled in self._compiled:
            masked_line = compiled.sub("<*>", masked_line)
        return masked_line

    def apply_tokens(self, tokens: Iterable[str]) -> list[str]:
        joined = " ".join(tokens)
        masked = self.apply(joined)
        return masked.split()
