from deepparse.drain.drain_engine import DrainEngine
from deepparse.masks_types import Mask


def test_drain_applies_masks_before_parsing():
    masks = [Mask(label="NUMBER", pattern=r"\d+", justification="numbers")]
    engine = DrainEngine(masks=masks)
    logs = ["value 123", "value 456"]
    templates = engine.parse(logs)
    assert templates[0] == templates[1] == "value <*>"
