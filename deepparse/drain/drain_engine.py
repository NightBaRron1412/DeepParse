"""Deterministic Drain-like parser."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from ..masks_types import Mask
from ..tokenize import mask_tokens, tokenize
from .masks_application import MaskApplier


@dataclass
class DrainCluster:
    template: List[str]
    size: int = 0

    def similarity(self, tokens: Sequence[str]) -> float:
        matches = 0
        for tmpl_tok, tok in zip(self.template, tokens):
            if tmpl_tok == tok or tmpl_tok == "<*>":
                matches += 1
        return matches / max(1, len(self.template))

    def update(self, tokens: Sequence[str]) -> None:
        self.size += 1
        for idx, tok in enumerate(tokens):
            if idx >= len(self.template):
                self.template.append(tok)
            elif self.template[idx] != tok:
                self.template[idx] = "<*>"

    def template_str(self) -> str:
        return " ".join(self.template)


@dataclass
class DrainEngine:
    depth: int = 4
    similarity_threshold: float = 0.6
    masks: Sequence[Mask] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.applier = MaskApplier(self.masks)
        self.clusters: Dict[Tuple[int, str], List[DrainCluster]] = {}

    def _cluster_key(self, tokens: Sequence[str]) -> Tuple[int, str]:
        prefix = " ".join(tokens[: self.depth])
        return len(tokens), prefix

    def add_log(self, line: str) -> DrainCluster:
        masked_line = self.applier.apply(line)
        tokens = tokenize(masked_line)
        tokens = mask_tokens(tokens)
        key = self._cluster_key(tokens)
        cluster_list = self.clusters.setdefault(key, [])
        best_cluster = None
        best_score = 0.0
        for cluster in cluster_list:
            score = cluster.similarity(tokens)
            if score > best_score:
                best_score = score
                best_cluster = cluster
        if best_cluster and best_score >= self.similarity_threshold:
            best_cluster.update(tokens)
            return best_cluster
        new_cluster = DrainCluster(template=list(tokens), size=0)
        new_cluster.update(tokens)
        cluster_list.append(new_cluster)
        return new_cluster

    def parse(self, lines: Iterable[str]) -> List[str]:
        templates: List[str] = []
        for line in lines:
            cluster = self.add_log(line)
            templates.append(cluster.template_str())
        return templates
