import pytest
from forge3d.style import _evaluate_filter, _filter_properties

@pytest.mark.parametrize("expression,expected", [(["<","pop",10],True),(["<=","pop",5],True),([">","pop",10],False),([">=","pop",5],True),(["!in","class","motorway"],False),(["!has","class"],False),(["==","$type","LineString"],True),(["==","$id",7],True),(["in","class","motorway"],True)])
def test_legacy_operators(expression,expected):
    props=_filter_properties({"id":7,"geometry":{"type":"MultiLineString"},"properties":{"pop":5,"class":"motorway"}})
    assert _evaluate_filter(expression,props) is expected

def test_unknown_operator_is_diagnostic():
    with pytest.warns(RuntimeWarning,match="unsupported operator"):
        assert not _evaluate_filter(["unknown","class"],{})


@pytest.mark.parametrize("expression", [
    ["bogus-op", "class", "x"],
    ["all", ["bogus-op", "class", "x"]],
    ["==", ["bogus-op", "class"], "x"],
])
def test_support_report_diagnoses_unknown_filter(expression):
    from forge3d.style import validate_style_support, parse_style
    style = {"version": 8, "sources": {}, "layers": [{"id":"roads", "type":"line", "filter":expression}]}
    for value in (style, parse_style(style)):
        report = validate_style_support(value)
        assert report.status != "ok"
        diagnostic = next(d for d in report.diagnostics if d.code == "unsupported_style_filter")
        assert diagnostic.layer_id == "roads"
        assert "bogus-op" in diagnostic.details["reason"]
