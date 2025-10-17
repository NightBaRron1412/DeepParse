"""Lightweight public API mirroring the example usage from the paper."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from .drain.drain_engine import DrainEngine
from .masks_types import Mask
from .synth.r1_deepseek_stub import synthesize_offline
from .utils.regex_library import validate_regexes
from .utils.sampling import deterministic_sample

try:  # Optional heavy dependency
    from .synth.hf_deepseek_r1 import synthesize_hf
except Exception:  # pragma: no cover - optional path
    synthesize_hf = None


def _ensure_mask_objects(masks: Iterable[Mask | dict[str, str]]) -> List[Mask]:
    converted: List[Mask] = []
    for mask in masks:
        if isinstance(mask, Mask):
            converted.append(mask)
        else:
            converted.append(Mask.from_dict(mask))
    return converted


def synth_masks(
    logs: Sequence[str],
    sample_size: int = 50,
    *,
    mode: str = "offline",
    temperature: float = 0.0,
    num_beams: int = 2,
    max_length: int = 512,
    strict: bool = False,
    model_name: str | None = None,
) -> List[dict[str, str]]:
    """Synthesise regex masks from raw log lines.

    Parameters mirror the usage snippet provided in the paper.  The function is
    deterministic because sampling uses :func:`deterministic_sample` and the
    offline synthesiser is rule based.  When ``mode="hf"`` it falls back to the
    optional Hugging Face pipeline with the requested generation controls.
    """

    if not logs:
        raise ValueError("Cannot synthesise masks from an empty log sequence")

    sample = deterministic_sample(logs, min(sample_size, len(logs)))
    if mode == "offline":
        masks = synthesize_offline(sample)
    elif mode == "hf":
        if synthesize_hf is None:  # pragma: no cover - optional dependency
            raise RuntimeError("Hugging Face mode requested but transformers is unavailable")
        masks = synthesize_hf(
            sample,
            model_name=model_name or "deepseek-ai/deepseek-coder-1.3b-base",
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
        )
    else:
        raise ValueError(f"Unsupported synthesis mode: {mode}")

    validate_regexes([mask.pattern for mask in masks], strict=strict)
    return [mask.to_dict() for mask in masks]


@dataclass
class Drain:
    """Convenience wrapper exposing ``load_masks``/``parse_all`` helpers."""

    depth: int = 4
    similarity_threshold: float = 0.6

    def __post_init__(self) -> None:
        self._engine = DrainEngine(
            depth=self.depth,
            similarity_threshold=self.similarity_threshold,
            masks=[],
        )

    def load_masks(self, masks: Iterable[Mask | dict[str, str]]) -> None:
        mask_objs = _ensure_mask_objects(masks)
        self._engine = DrainEngine(
            depth=self.depth,
            similarity_threshold=self.similarity_threshold,
            masks=mask_objs,
        )

    def parse_all(self, logs: Sequence[str]) -> List[str]:
        return self._engine.parse(logs)

    def add_log(self, log: str) -> str:
        return self._engine.add_log(log).template_str()
