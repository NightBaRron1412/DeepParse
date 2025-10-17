from deepparse.utils.regex_library import REGEX_CLASSES, classify_token, validate_regexes


def test_classify_token_matches_timestamp():
    assert classify_token("2024-01-01 00:00:00") == "TIMESTAMP"


def test_validate_regexes_strict_blocks_greedy():
    try:
        validate_regexes([r".*"], strict=True)
    except ValueError as exc:
        assert "Strict mode" in str(exc)
    else:  # pragma: no cover - ensures failure path
        raise AssertionError("Strict validation should raise on greedy pattern")


def test_regex_classes_compile():
    for regex_class in REGEX_CLASSES:
        assert regex_class.compile().match(regex_class.pattern.replace("^", "").replace("$", "").split("|")[0]) is not None
