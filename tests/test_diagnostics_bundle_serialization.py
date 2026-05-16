import json

import pytest

from forge3d.diagnostics import Diagnostic, LayerSummary, ValidationReport


def _diagnostic(code, severity, layer_id=None, object_id=None, details=None):
    return Diagnostic(
        code=code,
        severity=severity,
        message=f"{code} message",
        remediation=f"{code} remediation",
        support_level="unsupported" if severity in {"error", "fatal"} else "underdeveloped",
        layer_id=layer_id,
        object_id=object_id,
        details=details or {},
    )


def test_diagnostic_from_dict_roundtrip_preserves_bundle_ready_fields():
    payload = {
        "code": "unsupported_style_field",
        "severity": "warning",
        "message": "Unsupported paint fields were ignored by older paths.",
        "remediation": "Remove unsupported fields or use a supported style subset.",
        "support_level": "unsupported",
        "layer_id": "style.roads",
        "object_id": None,
        "details": {"fields": ["line-gradient"], "source": {"kind": "paint"}},
    }

    diag = Diagnostic.from_dict(payload)

    assert diag.to_dict() == payload


def test_validation_report_serialization_is_lossless_and_deterministic():
    report = ValidationReport(
        diagnostics=[
            _diagnostic("missing_glyphs", "warning", "labels", "city-2"),
            _diagnostic("crs_mismatch", "error", "roads", None),
            _diagnostic("experimental_feature", "warning", "labels", "line-1"),
            _diagnostic("diagnostic_inventory", "info", None, None),
        ],
        layer_summaries=[
            LayerSummary(
                layer_id="labels",
                layer_type="labels",
                support_level="underdeveloped",
                diagnostic_codes=["experimental_feature", "missing_glyphs"],
                object_count=2,
                details={"source": "fixture"},
            ),
            LayerSummary(
                layer_id="roads",
                layer_type="vector",
                support_level="unsupported",
                diagnostic_codes=["crs_mismatch"],
                object_count=1,
            ),
        ],
        estimated_gpu_memory_bytes=2048,
        supported_features={"diagnostics": "supported"},
        unsupported_features={"streamed_mvt": "non-goal", "vt_mask": "missing"},
    )

    payload = report.to_dict()
    restored = ValidationReport.from_dict(payload)

    assert restored.to_dict() == payload
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == json.dumps(
        restored.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_validation_report_orders_diagnostics_by_stable_sort_key():
    report = ValidationReport(
        diagnostics=[
            _diagnostic("missing_glyphs", "warning", "z-labels", "2"),
            _diagnostic("missing_glyphs", "warning", "a-labels", "1"),
            _diagnostic("crs_mismatch", "error", "roads", None),
            _diagnostic("fatal_path", "fatal", "terrain", None),
            _diagnostic("diagnostic_inventory", "info", None, None),
        ]
    )

    assert [
        (diag["severity"], diag["code"], diag["layer_id"], diag["object_id"])
        for diag in report.to_dict()["diagnostics"]
    ] == [
        ("fatal", "fatal_path", "terrain", None),
        ("error", "crs_mismatch", "roads", None),
        ("warning", "missing_glyphs", "a-labels", "1"),
        ("warning", "missing_glyphs", "z-labels", "2"),
        ("info", "diagnostic_inventory", None, None),
    ]


def test_diagnostic_details_are_json_serializable_and_sorted():
    diag = Diagnostic(
        code="label_rejection_summary",
        severity="warning",
        message="Labels were rejected during placement.",
        remediation="Inspect rejected label reasons.",
        support_level="underdeveloped",
        details={"z": [{"b": 2, "a": 1}], "a": {"y": 2, "x": 1}},
    )

    details = diag.to_dict()["details"]
    assert list(details.keys()) == ["a", "z"]
    assert list(details["a"].keys()) == ["x", "y"]
    assert list(details["z"][0].keys()) == ["a", "b"]

    with pytest.raises(TypeError, match="details must be JSON-serializable"):
        Diagnostic(
            code="bad_details",
            severity="info",
            message="bad details",
            remediation="Use JSON-serializable details.",
            details={"bad": object()},
        )


def test_empty_report_is_bundle_ready_ok_status():
    report = ValidationReport()

    assert report.status == "ok"
    assert report.render_blocked() is False
    assert report.to_dict() == {
        "status": "ok",
        "diagnostics": [],
        "layer_summaries": [],
        "estimated_gpu_memory_bytes": None,
        "supported_features": {},
        "unsupported_features": {},
        "render_blocked": False,
    }
