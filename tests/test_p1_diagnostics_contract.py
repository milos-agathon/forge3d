from __future__ import annotations


def test_p1_feature_local_diagnostic_factories_are_structured_and_serializable():
    from forge3d.diagnostics import (
        Diagnostic,
        ValidationReport,
        missing_external_asset_diagnostic,
        missing_label_field_diagnostic,
        unicode_coverage_gap_diagnostic,
        unavailable_terrain_sampler_diagnostic,
        unsupported_tile_feature_diagnostic,
        unsupported_tile_format_diagnostic,
    )

    diagnostics = [
        missing_label_field_diagnostic("name", layer_id="labels.cities", object_id="city-1"),
        unicode_coverage_gap_diagnostic(["\u2603"], layer_id="labels.cities", object_id="city-2"),
        unsupported_tile_format_diagnostic("cmpt", layer_id="tiles.local", object_id="root"),
        unsupported_tile_feature_diagnostic("EXT_mesh_gpu_instancing", layer_id="tiles.local", object_id="tile-1"),
        missing_external_asset_diagnostic("raster_overlay", layer_id="ortho", path="missing/ortho.tif"),
        unavailable_terrain_sampler_diagnostic(layer_id="labels.cities", object_id="city-3"),
    ]

    assert [diagnostic.code for diagnostic in diagnostics] == [
        "missing_label_field",
        "unicode_coverage_gap",
        "unsupported_tile_format",
        "unsupported_tile_feature",
        "missing_external_asset",
        "unavailable_terrain_sampler",
    ]
    assert all(isinstance(diagnostic, Diagnostic) for diagnostic in diagnostics)
    assert all(diagnostic.layer_id for diagnostic in diagnostics)
    assert all(diagnostic.remediation for diagnostic in diagnostics)

    report = ValidationReport(diagnostics=diagnostics)
    restored = ValidationReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()


def test_p1_warning_diagnostics_obey_fail_on_warning_policy():
    from forge3d.diagnostics import RenderFailurePolicy, ValidationReport, unicode_coverage_gap_diagnostic

    report = ValidationReport(
        diagnostics=[
            unicode_coverage_gap_diagnostic(["\u2603"], layer_id="labels.cities", object_id="city-2")
        ]
    )

    assert report.status == "warning"
    assert not report.render_blocked(RenderFailurePolicy.CONTINUE_ON_WARNING)
    assert report.render_blocked(RenderFailurePolicy.FAIL_ON_WARNING)


def test_prd_p0_diagnostic_inventory_is_structured_serializable_and_actionable():
    from forge3d.diagnostics import (
        REQUIRED_DIAGNOSTIC_CODES,
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

    diagnostics = [
        crs_mismatch_diagnostic("EPSG:32632", "EPSG:4326", layer_id="labels.cities", object_id="city-1"),
        missing_glyphs_diagnostic(["\u2603"], layer_id="labels.cities", object_id="city-2"),
        unsupported_style_field_diagnostic("style.roads", ["line-gradient"], section="paint"),
        unsupported_style_layer_type_diagnostic("style.heat", "heatmap"),
        pro_gated_path_diagnostic("native_cityjson", layer_id="buildings.city", object_id="b-1"),
        placeholder_fallback_diagnostic("zero_geometry_buildings", layer_id="buildings.empty", object_id="b-0"),
        experimental_feature_diagnostic("curved_label_rendering", layer_id="labels.roads", object_id="road-1"),
        vt_unsupported_family_diagnostic("normal", layer_id="terrain", object_id="vt.normal"),
        python_public_3dtiles_incomplete_diagnostic(layer_id="tiles.local", object_id="tile-1"),
        estimated_gpu_memory_diagnostic(2048, 1024, layer_id="scene", object_id="gpu-budget"),
        label_rejection_summary_diagnostic({"collision": 2}, layer_id="labels.cities"),
    ]

    assert {diagnostic.code for diagnostic in diagnostics} == set(REQUIRED_DIAGNOSTIC_CODES)
    assert all(diagnostic.severity in {"info", "warning", "error", "fatal"} for diagnostic in diagnostics)
    assert all(diagnostic.message for diagnostic in diagnostics)
    assert all(diagnostic.remediation for diagnostic in diagnostics)

    report = ValidationReport(diagnostics=diagnostics)
    restored = ValidationReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()
