from forge3d.diagnostics import (
    RenderFailurePolicy,
    ValidationReport,
    experimental_feature_diagnostic,
    placeholder_fallback_diagnostic,
    pro_gated_path_diagnostic,
)


def test_warning_policy_does_not_turn_errors_into_success():
    report = ValidationReport(
        diagnostics=[
            pro_gated_path_diagnostic("native building import", layer_id="buildings"),
            placeholder_fallback_diagnostic("zero geometry fallback", layer_id="buildings"),
        ]
    )

    assert report.status == "error"
    assert report.has_errors is True
    assert report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING) is True


def test_experimental_warning_can_continue_or_block_by_policy():
    report = ValidationReport(
        diagnostics=[experimental_feature_diagnostic("line labels", layer_id="labels")]
    )

    assert report.status == "warning"
    assert report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING) is False
    assert report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING) is True
