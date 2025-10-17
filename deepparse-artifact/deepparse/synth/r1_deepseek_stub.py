"""Offline fallback synthesizer mimicking DeepSeek-R1 behaviour."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..masks_types import Mask
from ..utils.regex_library import REGEX_CLASSES
from ..tokenize import tokenize


@dataclass
class StubConfig:
    require_core_classes: bool = True


CORE_MASKS = [
    Mask("TIMESTAMP", r"(?P<timestamp>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})", "Matches ISO timestamps"),
    Mask("IPV4", r"(?P<ip>(?:\d{1,3}\.){3}\d{1,3})", "Captures IPv4 addresses"),
    Mask("NUMBER", r"(?P<number>-?\d+(?:\.\d+)?)", "Numerical literals"),
    Mask("LOGLEVEL", r"(?P<level>TRACE|DEBUG|INFO|WARN|ERROR|FATAL)", "Standard log levels"),
]


def _infer_additional_masks(logs: Sequence[str]) -> List[Mask]:
    candidates: List[Mask] = []
    seen = set()
    for line in logs:
        for token in tokenize(line):
            if token.startswith("0x") and token not in seen:
                candidates.append(Mask("HEX", r"(?P<hex>0x[0-9a-fA-F]+)", "Hex identifiers"))
                seen.add("HEX")
            if token.startswith("/") and token not in seen:
                candidates.append(Mask("PATH", r"(?P<path>/[^\s]+)", "Unix style path"))
                seen.add("PATH")
            if token.count("-") == 4 and len(token) > 10 and token not in seen:
                candidates.append(Mask("UUID", r"(?P<uuid>[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})", "UUIDs"))
                seen.add("UUID")
    return candidates


def synthesize_offline(logs: Sequence[str], config: StubConfig | None = None) -> List[Mask]:
    cfg = config or StubConfig()
    masks = CORE_MASKS.copy()
    masks.extend(_infer_additional_masks(logs))
    if cfg.require_core_classes:
        required = {"TIMESTAMP", "IPV4", "NUMBER", "LOGLEVEL"}
        available = {mask.label for mask in masks}
        missing = required - available
        if missing:
            raise ValueError(f"Offline stub failed to synthesise required masks: {missing}")
    return masks
