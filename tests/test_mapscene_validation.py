import forge3d as f3d


def _camera() -> f3d.OrbitCamera:
    return f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=1000.0)


def _lighting() -> f3d.LightingPreset:
    return f3d.LightingPreset(name="daylight")


def _terrain(*, crs: str = "EPSG:32610", metadata=None) -> f3d.TerrainSource:
    terrain_metadata = {"width": 16, "height": 8, "asset_status": "fixture"}
    if metadata:
        terrain_metadata.update(metadata)
    return f3d.TerrainSource(
        path="fixtures/dem.tif",
        crs=crs,
        metadata=terrain_metadata,
        elevation_sampling_available=True,
    )


def _output(*, fmt: str = "png") -> f3d.OutputSpec:
    return f3d.OutputSpec(width=64, height=32, format=fmt)


def _codes(report: f3d.ValidationReport) -> list[str]:
    return [diagnostic.code for diagnostic in report.diagnostics]


def _diagnostic(report: f3d.ValidationReport, code: str):
    return next(diagnostic for diagnostic in report.diagnostics if diagnostic.code == code)


def test_validate_returns_structured_report_with_layer_summaries_and_memory_estimate():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        diagnostics_policy={"gpu_memory_budget_bytes": 128},
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                metadata={"width": 16, "height": 8, "asset_status": "fixture"},
            ),
            f3d.PointCloudLayer(
                layer_id="points",
                path="fixtures/points.las",
                crs="EPSG:32610",
                point_count=100,
                metadata={"asset_status": "fixture"},
            ),
        ],
    )

    report = scene.validate()

    assert isinstance(report, f3d.ValidationReport)
    assert scene.last_validation_report is report
    assert report.status == "error"
    assert report.render_blocked(f3d.RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert report.render_blocked(f3d.RenderFailurePolicy.FAIL_ON_WARNING) is True
    assert report.estimated_gpu_memory_bytes is not None
    assert report.estimated_gpu_memory_bytes > 128
    assert _codes(report) == ["placeholder_fallback", "estimated_gpu_memory"]
    point_diagnostic = _diagnostic(report, "placeholder_fallback")
    assert point_diagnostic.layer_id == "points"
    assert point_diagnostic.details["feature"] == "point cloud MapScene render path"
    assert _diagnostic(report, "estimated_gpu_memory").details["estimated_bytes"] == report.estimated_gpu_memory_bytes
    assert [summary.layer_id for summary in report.layer_summaries] == ["ortho", "points", "terrain"]
    assert report.supported_features["mapscene.validation"] == "supported"
    assert report.supported_features["mapscene.recipe"] == "underdeveloped"


def test_validate_reports_crs_mismatch_without_implicit_transform():
    scene = f3d.MapScene(
        terrain=_terrain(crs="EPSG:32610"),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(
                layer_id="wgs84-raster",
                path="fixtures/wgs84.tif",
                crs="EPSG:4326",
                metadata={"asset_status": "fixture"},
            )
        ],
    )

    report = scene.validate()

    assert report.status == "error"
    assert report.render_blocked(f3d.RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert report.render_blocked(f3d.RenderFailurePolicy.FAIL_ON_WARNING) is True
    diagnostic = _diagnostic(report, "crs_mismatch")
    assert diagnostic.layer_id == "wgs84-raster"
    assert diagnostic.details == {"layer_crs": "EPSG:4326", "scene_crs": "EPSG:32610"}


def test_validate_reports_missing_layer_crs_without_assuming_compatibility():
    scene = f3d.MapScene(
        terrain=_terrain(crs="EPSG:32610"),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(
                layer_id="unknown-crs-raster",
                path="fixtures/unknown-crs.tif",
                crs=None,
                metadata={"asset_status": "fixture"},
            )
        ],
    )

    report = scene.validate()

    assert report.status == "error"
    diagnostic = _diagnostic(report, "missing_crs")
    assert diagnostic.layer_id == "unknown-crs-raster"
    assert diagnostic.support_level == "unsupported"
    assert diagnostic.details == {"layer_crs": None, "scene_crs": "EPSG:32610"}
    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "unknown-crs-raster")
    assert summary.support_level == "unsupported"


def test_validate_reports_missing_terrain_crs_without_assuming_compatibility():
    scene = f3d.MapScene(
        terrain=_terrain(crs=None),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            )
        ],
    )

    report = scene.validate()

    assert report.status == "error"
    diagnostic = _diagnostic(report, "missing_crs")
    assert diagnostic.layer_id == "terrain"
    assert diagnostic.support_level == "unsupported"
    assert diagnostic.details == {"layer_crs": None, "scene_crs": None}
    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "terrain")
    assert summary.support_level == "unsupported"


def test_validate_reports_missing_glyphs_and_style_support_diagnostics():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road-1",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "roads",
                            "type": "line",
                            "paint": {"line-color": "#ffffff", "line-gradient": ["get", "speed"]},
                        },
                        {"id": "heat", "type": "heatmap", "paint": {"heatmap-radius": 12}},
                    ],
                },
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "cafe",
                        "kind": "point",
                        "text": "Café",
                        "geometry": {"type": "Point", "coordinates": (12.0, 12.0, 0.0)},
                    }
                ],
                glyph_atlas={"glyphs": list("Caf")},
            ),
        ],
    )

    report = scene.validate()
    codes = _codes(report)

    assert report.status == "error"
    assert codes == [
        "unsupported_style_layer_type",
        "label_rejection_summary",
        "missing_glyphs",
        "unsupported_style_field",
    ]
    assert _diagnostic(report, "missing_glyphs").object_id == "cafe"
    assert _diagnostic(report, "missing_glyphs").details["missing_glyphs"] == ["é"]
    assert _diagnostic(report, "unsupported_style_field").layer_id == "roads"
    assert _diagnostic(report, "unsupported_style_layer_type").layer_id == "heat"


def test_validate_uses_fatal_diagnostic_for_unsupported_output_format():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(fmt="jpg"),
    )

    report = scene.validate()

    assert report.status == "fatal"
    assert report.render_blocked(f3d.RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert report.render_blocked(f3d.RenderFailurePolicy.FAIL_ON_WARNING) is True
    diagnostic = _diagnostic(report, "unsupported_output_format")
    assert diagnostic.severity == "fatal"
    assert diagnostic.support_level == "unsupported"
    assert diagnostic.details["format"] == "jpg"


def test_validate_includes_building_geometry_count_in_gpu_memory_estimate():
    scene = f3d.MapScene(
        terrain=_terrain(metadata={"width": 8, "height": 8}),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source="fixtures/buildings.geojson",
                support_level="underdeveloped",
                geometry_count=7,
            )
        ],
    )

    report = scene.validate()

    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "buildings")
    assert summary.object_count == 7
    assert summary.memory_estimate_bytes == 7 * 96
    assert report.estimated_gpu_memory_bytes is not None
    assert report.estimated_gpu_memory_bytes >= (64 * 32 * 4) + (8 * 8 * 4) + (7 * 96)


def test_validate_reports_missing_source_identity_for_required_sources():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(path=None, crs="EPSG:32610", metadata={}),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(layer_id="ortho", path=None, crs="EPSG:32610"),
            f3d.PointCloudLayer(layer_id="points", path=None, crs="EPSG:32610", point_count=None),
        ],
    )

    report = scene.validate()
    by_layer = {diagnostic.layer_id: diagnostic for diagnostic in report.diagnostics}

    assert report.status == "error"
    assert by_layer["terrain"].code == "missing_source_identity"
    assert by_layer["terrain"].details["source_fields"] == ["path", "metadata"]
    assert by_layer["ortho"].code == "missing_source_identity"
    assert by_layer["points"].code == "missing_source_identity"
    assert {summary.layer_id: summary.support_level for summary in report.layer_summaries}["terrain"] == "unsupported"


def test_validate_reports_missing_external_assets_before_render():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="missing/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 16, "height": 8},
            elevation_sampling_available=True,
        ),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="missing/ortho.tif",
                crs="EPSG:32610",
                metadata={"width": 16, "height": 8},
            ),
            f3d.PointCloudLayer(
                layer_id="points",
                path="missing/points.las",
                crs="EPSG:32610",
                point_count=12,
            ),
        ],
    )

    report = scene.validate()
    missing = [diagnostic for diagnostic in report.diagnostics if diagnostic.code == "missing_external_asset"]

    assert report.status == "error"
    assert report.render_blocked(f3d.RenderFailurePolicy.CONTINUE_ON_WARNING) is True
    assert [diagnostic.layer_id for diagnostic in missing] == ["ortho", "points", "terrain"]
    assert missing[0].details["path"] == "missing/ortho.tif"
    assert {summary.layer_id: summary.support_level for summary in report.layer_summaries} == {
        "ortho": "unsupported",
        "points": "unsupported",
        "terrain": "unsupported",
    }


def test_validate_reports_unsupported_asset_formats_before_render():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.txt",
            crs="EPSG:32610",
            metadata={"width": 16, "height": 8, "asset_status": "fixture"},
            elevation_sampling_available=True,
        ),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.bmp",
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            ),
            f3d.PointCloudLayer(
                layer_id="points",
                path="fixtures/points.xyz",
                crs="EPSG:32610",
                point_count=12,
                metadata={"asset_status": "fixture"},
            ),
        ],
    )

    report = scene.validate()
    unsupported = [diagnostic for diagnostic in report.diagnostics if diagnostic.code == "unsupported_asset_format"]

    assert report.status == "error"
    assert [diagnostic.layer_id for diagnostic in unsupported] == ["ortho", "points", "terrain"]
    assert unsupported[0].details["path"] == "fixtures/ortho.bmp"
    assert ".tif" in unsupported[0].details["supported_extensions"]


def test_validate_reports_supported_building_source_asset_failures_before_render():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="missing-buildings",
                source="missing/buildings.geojson",
                support_level="supported",
                geometry_count=2,
                material_status="scalar_pbr_underdeveloped",
            ),
            f3d.MapSceneBuildingLayer(
                layer_id="bad-format-buildings",
                source="fixtures/buildings.txt",
                support_level="supported",
                geometry_count=2,
                material_status="scalar_pbr_underdeveloped",
                metadata={"asset_status": "fixture"},
            ),
        ],
    )

    report = scene.validate()
    by_layer = {diagnostic.layer_id: diagnostic for diagnostic in report.diagnostics}

    assert report.status == "error"
    assert by_layer["missing-buildings"].code == "missing_external_asset"
    assert by_layer["missing-buildings"].details["path"] == "missing/buildings.geojson"
    assert by_layer["bad-format-buildings"].code == "unsupported_asset_format"
    assert ".geojson" in by_layer["bad-format-buildings"].details["supported_extensions"]
    summaries = {summary.layer_id: summary.support_level for summary in report.layer_summaries}
    assert summaries["missing-buildings"] == "unsupported"
    assert summaries["bad-format-buildings"] == "unsupported"


def test_validate_reports_supported_scalar_building_render_adapter_before_render():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.MapSceneBuildingLayer(
                layer_id="buildings",
                source="fixtures/buildings.geojson",
                support_level="supported",
                geometry_count=4,
                material_status="scalar_pbr_underdeveloped",
                metadata={"asset_status": "fixture"},
            )
        ],
    )

    report = scene.validate()

    assert report.status == "ok"
    assert not any(diagnostic.code == "placeholder_fallback" for diagnostic in report.diagnostics)
    assert report.supported_features["buildings.scalar_materials"] == "supported"
    assert report.supported_features["buildings.mapscene_render"] == "supported"
    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "buildings")
    assert summary.support_level == "supported"


def test_validate_reports_vector_path_only_loader_placeholder_before_render():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                path="fixtures/roads.geojson",
                features=None,
                crs="EPSG:32610",
                metadata={"asset_status": "fixture"},
            )
        ],
    )

    report = scene.validate()
    diagnostic = _diagnostic(report, "placeholder_fallback")

    assert report.status == "error"
    assert diagnostic.layer_id == "roads"
    assert diagnostic.details["feature"] == "vector path loader"
    summary = next(summary for summary in report.layer_summaries if summary.layer_id == "roads")
    assert summary.support_level == "placeholder/fallback"


def test_validate_reports_vector_and_label_layers_without_renderable_data():
    scene = f3d.MapScene(
        terrain=_terrain(),
        camera=_camera(),
        lighting=_lighting(),
        output=_output(),
        layers=[
            f3d.VectorOverlay(layer_id="roads", crs="EPSG:32610", path=None, features=None),
            f3d.LabelLayer(layer_id="labels", labels=None, glyph_atlas={"glyphs": []}),
        ],
    )

    report = scene.validate()
    by_layer = {diagnostic.layer_id: diagnostic for diagnostic in report.diagnostics}

    assert report.status == "error"
    assert by_layer["roads"].code == "missing_renderable_data"
    assert by_layer["roads"].details["required_any_of"] == ["path", "features"]
    assert by_layer["labels"].code == "missing_renderable_data"
    assert by_layer["labels"].details["required_any_of"] == ["labels", "plan"]
