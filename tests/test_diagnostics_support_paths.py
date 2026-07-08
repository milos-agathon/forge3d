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
    vt_unsupported_family_diagnostic,
)


def test_required_diagnostic_factories_cover_inventory_with_severity_and_ids():
    diagnostics = [
        crs_mismatch_diagnostic("EPSG:4326", "EPSG:3857", layer_id="roads"),
        missing_glyphs_diagnostic(["Å", "ß"], layer_id="labels", object_id="city-1"),
        pro_gated_path_diagnostic("CityJSON building import", layer_id="buildings"),
        placeholder_fallback_diagnostic("GeoJSON building fallback", layer_id="buildings"),
        experimental_feature_diagnostic("curved labels", layer_id="labels", object_id="line-7"),
        vt_unsupported_family_diagnostic("normal", layer_id="terrain.vt"),
        python_public_3dtiles_incomplete_diagnostic(layer_id="tiles.3d"),
        estimated_gpu_memory_diagnostic(estimated_bytes=4096, budget_bytes=2048, layer_id="terrain"),
        label_rejection_summary_diagnostic(
            {"collision": 2, "missing_glyph": 1},
            layer_id="labels",
        ),
    ]

    by_code = {diag.code: diag for diag in diagnostics}

    for code in REQUIRED_DIAGNOSTIC_CODES - {"unsupported_style_field", "unsupported_style_layer_type"}:
        assert code in by_code
        assert by_code[code].severity in {"warning", "error"}
        assert by_code[code].remediation

    assert by_code["crs_mismatch"].severity == "error"
    assert by_code["crs_mismatch"].layer_id == "roads"
    assert by_code["missing_glyphs"].object_id == "city-1"
    assert by_code["vt_unsupported_family"].support_level == "unsupported"
    assert by_code["placeholder_fallback"].support_level == "placeholder/fallback"


def test_required_diagnostic_inventory_report_blocks_output_affecting_errors():
    report = ValidationReport(
        diagnostics=[
            placeholder_fallback_diagnostic("3D Tiles placeholder", layer_id="buildings"),
            vt_unsupported_family_diagnostic("mask", layer_id="terrain.vt"),
        ]
    )

    assert report.status == "error"
    assert report.render_blocked("continue_on_warning") is True
    assert report.render_blocked("fail_on_warning") is True


def test_building_layer_validator_reports_placeholder_fallback():
    from forge3d.buildings import Building, BuildingLayer, validate_building_layer_support
    import numpy as np

    layer = BuildingLayer(
        name="public-buildings",
        buildings=[
            Building(
                id="b1",
                positions=np.zeros((0, 3), dtype=np.float32),
                indices=np.zeros(0, dtype=np.uint32),
            )
        ],
        source_format="geojson",
    )

    report = validate_building_layer_support(layer, layer_id="buildings.public")

    assert report.status == "error"
    assert [diag.code for diag in report.diagnostics] == ["placeholder_fallback"]
    assert report.diagnostics[0].layer_id == "buildings.public"
    assert report.diagnostics[0].object_id == "b1"


def test_public_tiles3d_validator_reports_incomplete_render_workflow(tmp_path):
    from forge3d.tiles3d import load_tileset, validate_tiles3d_support
    import json

    tileset_path = tmp_path / "tileset.json"
    tileset_path.write_text(
        json.dumps(
            {
                "asset": {"version": "1.0"},
                "geometricError": 10.0,
                "root": {
                    "boundingVolume": {"sphere": [0, 0, 0, 1]},
                    "geometricError": 1.0,
                    "content": {"uri": "tile.b3dm"},
                },
            }
        ),
        encoding="utf-8",
    )
    tileset = load_tileset(tileset_path)

    report = validate_tiles3d_support(tileset, layer_id="tiles.local")

    assert report.status == "error"
    assert [diag.code for diag in report.diagnostics] == ["python_public_3dtiles_incomplete"]
    assert report.diagnostics[0].layer_id == "tiles.local"
    assert report.layer_summaries[0].details["tile_count"] == 1


def test_terrain_vt_validator_reports_all_native_families_supported():
    from forge3d.terrain_params import (
        TerrainVTSettings,
        VTLayerFamily,
        validate_terrain_vt_support,
    )

    settings = TerrainVTSettings(
        enabled=True,
        layers=[VTLayerFamily("albedo"), VTLayerFamily("normal"), VTLayerFamily("mask")],
    )

    report = validate_terrain_vt_support(settings, layer_id="terrain.vt")

    assert report.status == "ok"
    assert report.diagnostics == ()
    assert report.supported_features == {
        "vt.albedo": "supported",
        "vt.mask": "supported",
        "vt.normal": "supported",
    }
    assert report.layer_summaries[0].details["native_supported_families"] == [
        "albedo",
        "mask",
        "normal",
    ]


def test_label_validator_reports_experimental_paths_and_missing_glyphs():
    from forge3d.diagnostics import validate_label_support

    report = validate_label_support(
        [
            {"id": "road-1", "kind": "line", "text": "A1"},
            {"id": "bend-1", "kind": "curved", "text": "Bé"},
        ],
        atlas_glyphs=set("AB1"),
        layer_id="labels.transport",
    )

    assert report.status == "warning"
    assert [diag.code for diag in report.diagnostics] == [
        "experimental_feature",
        "experimental_feature",
        "missing_glyphs",
    ]
    assert report.diagnostics[0].object_id == "bend-1"
    assert report.diagnostics[2].details["missing_glyphs"] == ["é"]


def test_label_validator_preserves_missing_glyph_object_ids_per_label():
    from forge3d.diagnostics import validate_label_support

    report = validate_label_support(
        [
            {"id": "city-a", "kind": "point", "text": "Aé"},
            {"id": "city-b", "kind": "point", "text": "Bß"},
        ],
        atlas_glyphs=set("AB"),
        layer_id="labels.cities",
    )

    assert report.status == "warning"
    assert [(diag.code, diag.object_id, diag.details["missing_glyphs"]) for diag in report.diagnostics] == [
        ("missing_glyphs", "city-a", ["é"]),
        ("missing_glyphs", "city-b", ["ß"]),
    ]
