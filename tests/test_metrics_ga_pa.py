from deepparse.metrics import grouping_accuracy, parsing_accuracy


def test_grouping_accuracy_simple():
    true = ["a", "a", "b"]
    pred = ["a", "b", "b"]
    assert grouping_accuracy(true, pred) == 2 / 3


def test_parsing_accuracy_exact():
    true = ["foo", "bar"]
    pred = ["foo", "baz"]
    assert parsing_accuracy(true, pred) == 0.5
