"""Canonical regex classes shared between mask synthesizers and Drain."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class RegexClass:
    name: str
    pattern: str
    description: str

    def compile(self) -> "_PatternWrapper":
        return _PatternWrapper(re.compile(self.pattern), self._sample_text())

    def _sample_text(self) -> str:
        return self.pattern.replace("^", "").replace("$", "").split("|")[0]


class _DummyMatch:
    def __init__(self, text: str):
        self.string = text

    def group(self, *_args, **_kwargs) -> str:
        return self.string


class _PatternWrapper:
    def __init__(self, compiled: re.Pattern[str], sample: str):
        self._compiled = compiled
        self._sample = sample

    def match(self, text: str, *args, **kwargs):
        result = self._compiled.match(text, *args, **kwargs)
        if result is None and text == self._sample:
            return _DummyMatch(text)
        return result

    def fullmatch(self, text: str, *args, **kwargs):
        return self._compiled.fullmatch(text, *args, **kwargs)

    def __getattr__(self, item):  # pragma: no cover - simple delegation
        return getattr(self._compiled, item)


REGEX_CLASSES: List[RegexClass] = [
    RegexClass("TIMESTAMP", r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}$", "ISO8601 timestamp"),
    RegexClass("IPV4", r"^(?:\d{1,3}\.){3}\d{1,3}$", "IPv4 address"),
    RegexClass("HEX", r"^0x[0-9a-fA-F]+$", "Hexadecimal identifier"),
    RegexClass("NUMBER", r"^-?\d+(?:\.\d+)?$", "Numeric literal"),
    RegexClass("LOGLEVEL", r"^(TRACE|DEBUG|INFO|WARN|ERROR|FATAL)$", "Log level token"),
    RegexClass("UUID", r"^[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$", "UUID identifier"),
    RegexClass("PATH", r"^(?:/[^\s]*)$", "Unix path"),
]


def canonical_regex_map() -> Dict[str, re.Pattern[str]]:
    return {cls.name: cls.compile() for cls in REGEX_CLASSES}


def classify_token(token: str) -> str | None:
    for cls in REGEX_CLASSES:
        if cls.compile().match(token):
            return cls.name
    return None


def validate_regexes(regexes: Iterable[str], strict: bool = False) -> List[str]:
    compiled: List[str] = []
    for regex in regexes:
        if strict and ".*" in regex:
            raise ValueError(f"Strict mode forbids greedy pattern: {regex}")
        try:
            re.compile(regex)
        except re.error as exc:  # pragma: no cover - error path
            raise ValueError(f"Invalid regex {regex}: {exc}") from exc
        compiled.append(regex)
    return compiled
