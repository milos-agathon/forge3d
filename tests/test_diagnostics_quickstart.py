import json

from forge3d.diagnostics import (
    RenderFailurePolicy,
    ValidationReport,
    crs_mismatch_diagnostic,
    estimated_gpu_memory_diagnostic,
    experimental_feature_diagnostic,
    label_rejection_summary_diagnostic,
    missing_glyphs_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
    python_public_3dtiles_incomplete_diagnostic,
    unsupported_style_field_diagnostic,
    unsupported_style_layer_type_diagnostic,
    vt_unsupported_family_diagnostic,
)
from forge3d.style import validate_style_support


def test_quickstart_warning_policy_scenario_is_executable():
    report = ValidationReport(
        diagnostics=[missing_glyphs_diagnostic(["é"], layer_id="labels")]
    )

    assert report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING) is False
    assert report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING) is True


def test_quickstart_required_inventory_serializes_stably():
    report = ValidationReport(
        diagnostics=[
            crs_mismatch_diagnostic("EPSG:4326", "EPSG:3857", layer_id="roads"),
            missing_glyphs_diagnostic(["é"], layer_id="labels"),
            unsupported_style_field_diagnostic("roads", ["line-gradient"]),
            unsupported_style_layer_type_diagnostic("heat", "heatmap"),
            pro_gated_path_diagnostic("native CityJSON import", layer_id="buildings"),
            placeholder_fallback_diagnostic("zero geometry fallback", layer_id="buildings"),
            experimental_feature_diagnostic("curved labels", layer_id="labels"),
            vt_unsupported_family_diagnostic("normal", layer_id="terrain.vt"),
            python_public_3dtiles_incomplete_diagnostic(layer_id="tiles.3d"),
            estimated_gpu_memory_diagnostic(4096, 2048, layer_id="terrain"),
            label_rejection_summary_diagnostic({"collision": 1}, layer_id="labels"),
        ]
    )

    payload = report.to_dict()
    restored = ValidationReport.from_dict(payload).to_dict()

    assert restored == payload
    assert json.dumps(payload, sort_keys=True, separators=(",", ":")) == json.dumps(
        restored,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_quickstart_style_validation_scenario_emits_structured_diagnostics():
    report = validate_style_support(
        {
            "version": 8,
            "layers": [
                {"id": "roads", "type": "line", "paint": {"line-gradient": ["get", "speed"]}},
                {"id": "heat", "type": "heatmap"},
            ],
        }
    )

    assert [diag.code for diag in report.diagnostics] == [
        "unsupported_style_layer_type",
        "unsupported_style_field",
    ]
