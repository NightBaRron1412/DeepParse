"""Evaluation runner for DeepParse."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from ..dataset_loader import Dataset, load_dataset
from ..drain.drain_engine import DrainEngine
from ..logging_utils import get_logger
from ..masks_types import Mask
from ..metrics import grouping_accuracy, parsing_accuracy
from ..tokenize import mask_tokens, tokenize
from ..utils.regex_library import validate_regexes
from ..seeds import resolve_seed, set_global_seed
from ..io_paths import build_paths
from ..synth import synthesize_masks
from ..utils.yaml_loader import load_yaml

LOGGER = get_logger(__name__)


def _load_masks(path: Path) -> List[Mask]:
    data = json.loads(path.read_text(encoding="utf-8"))
    masks = [Mask(**entry) for entry in data]
    validate_regexes([mask.pattern for mask in masks])
    return masks


def _ground_truth_templates(dataset: Dataset) -> List[str]:
    templates = []
    for line in dataset.logs:
        tokens = mask_tokens(tokenize(line))
        templates.append(" ".join(tokens))
    return templates


def _group_ids(templates: Sequence[str]) -> List[str]:
    return [str(hash(template)) for template in templates]


@dataclass
class EvaluationConfig:
    base_config: Path
    datasets: Sequence[str]
    output_csv: Path
    timing_csv: Path

    @classmethod
    def from_file(cls, path: Path) -> "EvaluationConfig":
        config_data = load_yaml(path)
        base_path = Path(config_data["base_config"])
        datasets = config_data.get("datasets", [])
        output_csv = Path(config_data["output_csv"])
        timing_csv = Path(config_data.get("timing_csv", ""))
        return cls(base_config=base_path, datasets=datasets, output_csv=output_csv, timing_csv=timing_csv)


class EvaluationRunner:
    def __init__(self, config_path: Path):
        self.config = EvaluationConfig.from_file(config_path)
        base_data = load_yaml(self.config.base_config)
        self.seed = resolve_seed(base_data.get("seed"))
        self.paths = build_paths(
            dataset_dir=base_data["dataset_dir"],
            mask_dir=base_data["mask_dir"],
            output_dir=base_data["output_dir"],
            log_dir=base_data["log_dir"],
        )
        self.mode = base_data.get("mode", "offline")
        self.k = int(base_data.get("k", 50))
        self.strict = bool(base_data.get("strict", False))

    def _ensure_masks(self, dataset: Dataset) -> Path:
        mask_path = self.paths.mask_dir / f"{dataset.name}.json"
        if not mask_path.exists():
            LOGGER.info("Masks missing for %s; synthesising", dataset.name)
            synthesize_masks(dataset, self.k, mask_path, mode=self.mode, strict=self.strict)
        return mask_path

    def evaluate_dataset(self, dataset_name: str) -> Dict[str, float]:
        dataset = load_dataset(dataset_name, self.paths)
        mask_path = self._ensure_masks(dataset)
        masks = _load_masks(mask_path)
        engine = DrainEngine(masks=masks)
        predicted = engine.parse(dataset.logs)
        ground_truth = _ground_truth_templates(dataset)
        ga = grouping_accuracy(_group_ids(ground_truth), _group_ids(predicted))
        pa = parsing_accuracy(ground_truth, predicted)
        LOGGER.info("Dataset %s: GA=%.3f PA=%.3f", dataset_name, ga, pa)
        return {"dataset": dataset_name, "method": "DeepParse", "GA": ga, "PA": pa}

    def run(self) -> List[Dict[str, float | str]]:
        set_global_seed(self.seed)
        rows = [self.evaluate_dataset(name) for name in self.config.datasets]
        if rows:
            ga_avg = sum(row["GA"] for row in rows) / len(rows)
            pa_avg = sum(row["PA"] for row in rows) / len(rows)
            rows.append({"dataset": "MacroAvg", "method": "DeepParse", "GA": ga_avg, "PA": pa_avg})
        self.config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["dataset", "method", "GA", "PA"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        LOGGER.info("Wrote metrics CSV to %s", self.config.output_csv)
        return rows
