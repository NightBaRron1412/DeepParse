"""Dataset loading utilities for corrected LogHub corpora."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .io_paths import PathConfig
from .logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Dataset:
    name: str
    path: Path
    logs: Sequence[str]

    @property
    def checksum(self) -> str:
        data = "\n".join(self.logs).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


EXPECTED_FILES = ["raw.log"]


def _create_demo_dataset(path: Path) -> None:
    if (path / "raw.log").exists():
        return
    LOGGER.info("Creating bundled demo dataset at %s", path)
    path.mkdir(parents=True, exist_ok=True)
    demo_logs = [
        "2024-01-01 00:00:00 INFO Worker-1 Completed task 42 in 0.5s",
        "2024-01-01 00:00:01 INFO Worker-2 Completed task 43 in 0.7s",
        "2024-01-01 00:00:02 WARN Worker-1 Retrying task 44",
        "2024-01-01 00:00:03 ERROR Worker-3 Failed task 45 with code 500",
    ]
    (path / "raw.log").write_text("\n".join(demo_logs), encoding="utf-8")
    manifest = {
        "name": "DemoTiny",
        "logs": len(demo_logs),
        "checksum": hashlib.sha256("\n".join(demo_logs).encode("utf-8")).hexdigest(),
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_dataset(name: str, paths: PathConfig, create_demo: bool = True) -> Dataset:
    dataset_root = paths.dataset_dir / name
    if create_demo and name == "DemoTiny":
        _create_demo_dataset(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset {name} missing at {dataset_root}")

    for expected in EXPECTED_FILES:
        file_path = dataset_root / expected
        if not file_path.exists():
            raise FileNotFoundError(f"Expected file {expected} in {dataset_root}")

    logs = [line.strip() for line in (dataset_root / "raw.log").read_text(encoding="utf-8").splitlines() if line.strip()]
    dataset = Dataset(name=name, path=dataset_root, logs=logs)
    LOGGER.info("Loaded dataset %s with %d logs", name, len(logs))
    return dataset


def load_many(names: Iterable[str], paths: PathConfig) -> List[Dataset]:
    return [load_dataset(name, paths) for name in names]
