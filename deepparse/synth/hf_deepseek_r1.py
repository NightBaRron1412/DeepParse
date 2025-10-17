"""Optional Hugging Face based synthesiser."""
from __future__ import annotations

import json
from typing import Sequence

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ..masks_types import Mask
from ..logging_utils import get_logger
from .prompt_templates import MASK_SYNTH_PROMPT
from ..utils.regex_library import validate_regexes

LOGGER = get_logger(__name__)


def synthesize_hf(
    logs: Sequence[str],
    *,
    model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
    temperature: float = 0.0,
    num_beams: int = 2,
    max_length: int = 512,
) -> Sequence[Mask]:
    prompt = MASK_SYNTH_PROMPT.format(logs="\n".join(logs))
    LOGGER.info("Loading Hugging Face model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    output = generator(
        prompt,
        max_new_tokens=max_length,
        num_beams=num_beams,
        temperature=temperature,
    )[0]["generated_text"]
    masks_json = json.loads(output)
    regexes = [entry["pattern"] for entry in masks_json]
    validate_regexes(regexes)
    return [Mask(label=entry["label"], pattern=entry["pattern"], justification=entry["justification"]) for entry in masks_json]
