from deepparse.synth.r1_deepseek_stub import synthesize_offline


def test_offline_stub_returns_required_masks():
    logs = [
        "2024-01-01 00:00:00 INFO worker Completed job 1",
        "2024-01-01 00:00:01 WARN worker Completed job 2",
    ]
    masks = synthesize_offline(logs)
    labels = {mask.label for mask in masks}
    assert {"TIMESTAMP", "IPV4", "NUMBER", "LOGLEVEL"}.issubset(labels)
    second_masks = synthesize_offline(logs)
    assert [mask.pattern for mask in masks] == [mask.pattern for mask in second_masks]
