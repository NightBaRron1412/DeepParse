"""Tests for the lightweight synth/Drain public API."""
from __future__ import annotations

from deepparse import Drain, synth_masks


def test_public_api_roundtrip() -> None:
    logs = [
        "2024-01-15 10:30:45 INFO Auth user john_doe logged in from 192.168.1.10",
        "2024-01-15 10:30:46 WARN Memory usage at 85% for host 10.0.0.5",
        "2024-01-15 10:30:47 ERROR Database timeout after 45 seconds",
    ]

    patterns = synth_masks(logs, sample_size=2)
    assert isinstance(patterns, list)
    assert patterns, "synth_masks should return at least one mask"
    assert {entry["label"] for entry in patterns} >= {"TIMESTAMP", "IPV4", "NUMBER", "LOGLEVEL"}

    drain = Drain()
    drain.load_masks(patterns)
    parsed = drain.parse_all(logs)
    assert len(parsed) == len(logs)
    assert all("<*>" in template for template in parsed)
