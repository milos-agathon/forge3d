import pytest
from forge3d.style import _evaluate_filter

@pytest.mark.parametrize("value", ["motorway","residential",None])
def test_expression_matches_legacy(value):
    props={"class":value}
    assert _evaluate_filter(["==",["get","class"],"motorway"],props) == _evaluate_filter(["==","class","motorway"],props)

def test_malformed_expression_is_diagnostic():
    with pytest.warns(RuntimeWarning,match="Invalid style filter"):
        assert not _evaluate_filter(["==",["get"],"motorway"],{})


@pytest.mark.parametrize("expression", [[["bad"],["get","class"]], ["==",["get",[]],"x"], ["==",["get","class"]]])
def test_malformed_expression_structure_is_diagnostic(expression):
    with pytest.warns(RuntimeWarning,match="Invalid style filter"):
        assert not _evaluate_filter(expression,{})


@pytest.mark.parametrize("value", ["motorway", "residential"])
def test_boolean_combinators_with_legacy_and_expression_filters(value):
    props = {"class": value}
    legacy = ["==", "class", "motorway"]
    expression = ["==", ["get", "class"], "motorway"]
    for op in ("all", "any", "none"):
        expected = value != "motorway" if op == "none" else value == "motorway"
        assert _evaluate_filter([op, legacy], props) is expected
        assert _evaluate_filter([op, expression], props) is expected
