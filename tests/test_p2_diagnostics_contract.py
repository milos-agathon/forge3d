from __future__ import annotations


def _scene_with_layer(layer):
    from forge3d.map_scene import LightingPreset, MapScene, OrbitCamera, OutputSpec, TerrainSource

    return MapScene(
        terrain=TerrainSource(path=None, crs="EPSG:4326", metadata={"source_id": "fixture-dem"}),
        camera=OrbitCamera(),
        lighting=LightingPreset(),
        layers=[layer],
        output=OutputSpec(width=64, height=64),
    )


def _diagnostics_by_code(report):
    return {diagnostic.code: diagnostic for diagnostic in report.diagnostics}


def _summary(report, layer_id: str):
    return next(summary for summary in report.layer_summaries if summary.layer_id == layer_id)


def test_p2_feature_local_diagnostic_factories_are_structured_and_serializable():
    from forge3d.diagnostics import (
        Diagnostic,
        P2_FEATURE_DIAGNOSTIC_CODES,
        ValidationReport,
        experimental_feature_diagnostic,
        missing_texture_path_diagnostic,
        missing_uvs_diagnostic,
        placeholder_fallback_diagnostic,
        pro_gated_path_diagnostic,
        unavailable_cache_lod_stats_diagnostic,
        unsupported_instancing_path_diagnostic,
        unsupported_texture_format_diagnostic,
        vt_unsupported_family_diagnostic,
    )

    diagnostics = [
        missing_texture_path_diagnostic(
            "missing/facade.png",
            layer_id="buildings.textured",
            object_id="building-1",
            material_id="facade",
        ),
        missing_uvs_diagnostic(
            layer_id="buildings.textured",
            object_id="building-1",
            material_id="facade",
        ),
        unsupported_texture_format_diagnostic(
            "ktx2",
            layer_id="buildings.textured",
            object_id="building-1",
            path="facade.ktx2",
        ),
        unavailable_cache_lod_stats_diagnostic(
            "point_cloud_layer",
            ["cache", "lod"],
            layer_id="points.large",
        ),
        unsupported_instancing_path_diagnostic(
            "building layer MapScene instancing",
            layer_id="buildings.repeated",
            object_id="instancing",
        ),
        vt_unsupported_family_diagnostic("normal", layer_id="terrain.vt", object_id="vt.normal"),
        placeholder_fallback_diagnostic("scalar building material fallback", layer_id="buildings.textured"),
        pro_gated_path_diagnostic("textured building import", layer_id="buildings.textured"),
        experimental_feature_diagnostic("advanced curved labels", layer_id="labels.roads"),
    ]

    assert {diagnostic.code for diagnostic in diagnostics}.issuperset(P2_FEATURE_DIAGNOSTIC_CODES)
    assert all(isinstance(diagnostic, Diagnostic) for diagnostic in diagnostics)
    assert all(diagnostic.severity in {"info", "warning", "error", "fatal"} for diagnostic in diagnostics)
    assert all(diagnostic.message for diagnostic in diagnostics)
    assert all(diagnostic.remediation for diagnostic in diagnostics)
    assert all(diagnostic.support_level for diagnostic in diagnostics)

    report = ValidationReport(diagnostics=diagnostics)
    restored = ValidationReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()


def test_p2_diagnostic_serialization_order_is_deterministic():
    from forge3d.diagnostics import (
        ValidationReport,
        missing_texture_path_diagnostic,
        missing_uvs_diagnostic,
        unavailable_cache_lod_stats_diagnostic,
    )

    diagnostics = [
        unavailable_cache_lod_stats_diagnostic("tiles3d_layer", ["lod", "cache"], layer_id="tiles"),
        missing_uvs_diagnostic(layer_id="buildings", object_id="b-2", material_id="roof"),
        missing_texture_path_diagnostic("missing/facade.png", layer_id="buildings", object_id="b-1"),
    ]

    first = ValidationReport(diagnostics=diagnostics).to_dict()
    second = ValidationReport(diagnostics=list(reversed(diagnostics))).to_dict()

    assert first == second
    assert [item["code"] for item in first["diagnostics"]] == [
        "missing_texture_path",
        "missing_uvs",
        "unavailable_cache_lod_stats",
    ]


def test_mapscene_reports_building_texture_diagnostics_before_render(tmp_path):
    from forge3d.map_scene import MapSceneBuildingLayer

    texture_path = tmp_path / "missing_facade.ktx2"
    layer = MapSceneBuildingLayer.from_geojson(
        tmp_path / "buildings.geojson",
        layer_id="buildings.textured",
        support_level="underdeveloped",
        geometry_count=1,
        metadata={
            "source_id": "fixture-buildings",
            "textured_materials": [
                {
                    "material_id": "facade",
                    "object_id": "building-1",
                    "albedo_texture": str(texture_path),
                    "texture_format": "ktx2",
                    "uv_available": False,
                    "scalar_fallback": True,
                }
            ],
        },
    )

    report = _scene_with_layer(layer).validate()
    diagnostics = _diagnostics_by_code(report)

    assert report.status == "error"
    assert report.render_blocked()
    assert diagnostics["missing_texture_path"].details["path"] == str(texture_path)
    assert diagnostics["missing_texture_path"].details["material_id"] == "facade"
    assert diagnostics["missing_uvs"].object_id == "building-1"
    assert diagnostics["unsupported_texture_format"].details["format"] == "ktx2"
    assert diagnostics["placeholder_fallback"].details["feature"] == "building textured material scalar fallback"
    assert _summary(report, "buildings.textured").support_level == "placeholder/fallback"


def test_mapscene_reports_cache_lod_and_instancing_availability_diagnostics():
    from forge3d.map_scene import MapSceneBuildingLayer

    layer = MapSceneBuildingLayer(
        layer_id="buildings.repeated",
        source={"source_format": "geojson", "source_id": "fixture-buildings"},
        support_level="underdeveloped",
        geometry_count=10,
        metadata={
            "source_id": "fixture-buildings",
            "unavailable_cache_lod_stats": ["cache", "lod"],
            "instancing": {
                "requested": True,
                "path": "building layer MapScene instancing",
                "support_level": "unsupported",
            },
        },
    )

    report = _scene_with_layer(layer).validate()
    diagnostics = _diagnostics_by_code(report)
    summary = _summary(report, "buildings.repeated")

    assert diagnostics["unavailable_cache_lod_stats"].support_level == "underdeveloped"
    assert diagnostics["unavailable_cache_lod_stats"].details == {
        "layer_type": "building_layer",
        "unavailable_stats": ["cache", "lod"],
    }
    assert diagnostics["unsupported_instancing_path"].support_level == "unsupported"
    assert diagnostics["unsupported_instancing_path"].details["path"] == "building layer MapScene instancing"
    assert summary.details["instancing_status"]["support_level"] == "unsupported"
    assert summary.details["unavailable_cache_lod_stats"] == ["cache", "lod"]
