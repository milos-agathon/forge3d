import pytest

import forge3d as f3d
from forge3d.diagnostics import (
    Diagnostic,
    LayerSummary,
    RenderFailurePolicy,
    SeverityPolicy,
    SupportMatrixEntry,
    ValidationReport,
)


def test_diagnostic_preserves_required_fields():
    diag = Diagnostic(
        code="missing_glyphs",
        severity="warning",
        message="2 glyphs are missing from the active atlas.",
        remediation="Load an atlas with the missing glyphs or change label text.",
        support_level="underdeveloped",
        layer_id="labels.cities",
        object_id="label:zurich",
        details={"missing_glyphs": ["e_acute", "sharp_s"], "count": 2},
    )

    assert diag.code == "missing_glyphs"
    assert diag.severity == "warning"
    assert diag.layer_id == "labels.cities"
    assert diag.object_id == "label:zurich"
    assert diag.to_dict() == {
        "code": "missing_glyphs",
        "severity": "warning",
        "message": "2 glyphs are missing from the active atlas.",
        "remediation": "Load an atlas with the missing glyphs or change label text.",
        "support_level": "underdeveloped",
        "layer_id": "labels.cities",
        "object_id": "label:zurich",
        "details": {"count": 2, "missing_glyphs": ["e_acute", "sharp_s"]},
    }


@pytest.mark.parametrize("severity", ["info", "warning", "error", "fatal"])
def test_diagnostic_accepts_required_severities(severity):
    diag = Diagnostic(
        code="diagnostic_inventory",
        severity=severity,
        message="inventory entry",
        remediation="No action required.",
    )

    assert diag.severity == severity


@pytest.mark.parametrize("support_level", [
    "supported",
    "underdeveloped",
    "missing",
    "Pro-gated",
    "placeholder/fallback",
    "experimental",
    "unsupported",
    "non-goal",
])
def test_diagnostic_accepts_prd_support_levels(support_level):
    diag = Diagnostic(
        code="support_status",
        severity="info",
        message="support classification",
        remediation="Read the support matrix.",
        support_level=support_level,
    )

    assert diag.support_level == support_level


def test_diagnostic_rejects_unknown_severity_and_support_level():
    with pytest.raises(ValueError, match="Unknown diagnostic severity"):
        Diagnostic(
            code="bad",
            severity="debug",
            message="bad severity",
            remediation="Use a supported severity.",
        )

    with pytest.raises(ValueError, match="Unknown support level"):
        Diagnostic(
            code="bad",
            severity="warning",
            message="bad support level",
            remediation="Use PRD Appendix B terms.",
            support_level="partial",
        )


def test_validation_report_rejects_unknown_support_summary_levels():
    with pytest.raises(ValueError, match="Unknown support level"):
        ValidationReport(supported_features={"diagnostics": "partial"})

    with pytest.raises(ValueError, match="Unknown support level"):
        ValidationReport(unsupported_features={"style.full_mapbox_spec": "mostly"})


def test_validation_report_derives_status_and_policy_blocking():
    warning_report = ValidationReport(
        diagnostics=[
            Diagnostic(
                code="missing_glyphs",
                severity="warning",
                message="Missing glyphs can affect label output.",
                remediation="Load an atlas with the missing glyphs.",
            )
        ]
    )

    assert warning_report.status == "warning"
    assert warning_report.has_errors is False
    assert warning_report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING) is False
    assert warning_report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING) is True

    error_report = ValidationReport(
        diagnostics=[
            Diagnostic(
                code="crs_mismatch",
                severity="error",
                message="Layer CRS differs from scene CRS.",
                remediation="Provide a transform or matching CRS.",
                support_level="unsupported",
                layer_id="roads",
            )
        ]
    )

    assert error_report.status == "error"
    assert error_report.has_errors is True
    assert error_report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert error_report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING) is True


def test_layer_summary_and_support_matrix_entry_contracts():
    layer = LayerSummary(
        layer_id="roads",
        layer_type="vector",
        support_level="underdeveloped",
        diagnostic_codes=["unsupported_style_field", "missing_glyphs"],
        object_count=12,
        bounds=(0.0, 1.0, 2.0, 3.0),
        memory_estimate_bytes=4096,
        details={"source": "roads.geojson"},
    )
    matrix_entry = SupportMatrixEntry(
        area="style",
        capability="line layers",
        support_level="supported",
        scope="local/provided features",
        limitations=["No streamed MVT rendering"],
        diagnostic_codes=["unsupported_style_field"],
        remediation="Use supported paint/layout fields.",
        evidence=["tests/test_diagnostics_style_support.py"],
    )

    assert layer.to_dict()["diagnostic_codes"] == [
        "missing_glyphs",
        "unsupported_style_field",
    ]
    assert matrix_entry.to_dict()["support_level"] == "supported"
    assert matrix_entry.to_dict()["scope"] == "local/provided features"


def test_severity_policy_and_public_exports():
    assert SeverityPolicy.status_for([]) == "ok"
    assert SeverityPolicy.status_for(["info", "warning"]) == "warning"
    assert SeverityPolicy.status_for(["warning", "error"]) == "error"
    assert SeverityPolicy.status_for(["fatal", "warning"]) == "fatal"

    assert f3d.Diagnostic is Diagnostic
    assert f3d.ValidationReport is ValidationReport
    for export_name in [
        "Diagnostic",
        "ValidationReport",
        "LayerSummary",
        "SupportMatrixEntry",
        "RenderFailurePolicy",
        "SeverityPolicy",
    ]:
        assert export_name in f3d.__all__
