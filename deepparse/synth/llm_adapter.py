"""High level interface for mask synthesis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..dataset_loader import Dataset
from ..logging_utils import get_logger
from ..masks_types import Mask, MaskBundle
from ..utils.regex_library import validate_regexes
from ..utils.sampling import deterministic_sample
from .prompt_templates import MASK_SYNTH_PROMPT
from .r1_deepseek_stub import synthesize_offline

LOGGER = get_logger(__name__)


class UnsupportedModeError(ValueError):
    pass


def synthesize_masks(
    dataset: Dataset,
    k: int,
    out_path: Path,
    mode: str = "offline",
    strict: bool = False,
) -> MaskBundle:
    LOGGER.info("Synthesising masks for %s with mode=%s", dataset.name, mode)
    sample = deterministic_sample(dataset.logs, k)
    masks: Sequence[Mask]
    if mode == "offline":
        masks = synthesize_offline(sample)
    elif mode == "hf":  # pragma: no cover - optional heavy path
        from .hf_deepseek_r1 import synthesize_hf

        masks = synthesize_hf(sample)
    else:  # pragma: no cover - argument validation
        raise UnsupportedModeError(mode)

    validate_regexes([mask.pattern for mask in masks], strict=strict)
    bundle = MaskBundle(dataset=dataset.name, masks=list(masks))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle.to_json(), indent=2), encoding="utf-8")
    LOGGER.info("Wrote %d masks to %s", len(masks), out_path)
    return bundle
