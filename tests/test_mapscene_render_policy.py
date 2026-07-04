import pytest

import forge3d as f3d


def _base_scene(*, render_policy=f3d.RenderFailurePolicy.CONTINUE_ON_WARNING, layers=(), diagnostics_policy=None):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 16, "height": 8, "asset_status": "fixture"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=500.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=40, height=24, format="png"),
        render_policy=render_policy,
        diagnostics_policy=diagnostics_policy,
        layers=list(layers),
    )


def test_render_performs_validation_and_continues_on_warnings_by_default(tmp_path):
    scene = _base_scene(diagnostics_policy={"gpu_memory_budget_bytes": 1})
    output_path = tmp_path / "warning.png"

    report = scene.render(str(output_path), allow_placeholder=True)

    assert output_path.exists()
    assert report.status == "warning"
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "warning"
    assert scene.last_validation_report.render_blocked(f3d.RenderFailurePolicy.CONTINUE_ON_WARNING) is False
    assert [diagnostic.code for diagnostic in scene.last_validation_report.diagnostics] == [
        "estimated_gpu_memory",
    ]


def test_render_fail_on_warning_blocks_and_does_not_write_png(tmp_path):
    scene = _base_scene(
        render_policy=f3d.RenderFailurePolicy.FAIL_ON_WARNING,
        diagnostics_policy={"gpu_memory_budget_bytes": 1},
    )
    output_path = tmp_path / "blocked-warning.png"

    with pytest.raises(RuntimeError, match="warning diagnostics"):
        scene.render(str(output_path))

    assert not output_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "warning"


def test_render_errors_always_block_and_do_not_write_png(tmp_path):
    scene = _base_scene(
        layers=[
            f3d.RasterOverlay(
                layer_id="wgs84",
                path="fixtures/wgs84.tif",
                crs="EPSG:4326",
                metadata={"asset_status": "fixture"},
            )
        ]
    )
    output_path = tmp_path / "blocked-error.png"

    with pytest.raises(RuntimeError, match="blocking diagnostics"):
        scene.render(str(output_path))

    assert not output_path.exists()
    assert scene.last_validation_report is not None
    assert scene.last_validation_report.status == "error"


def test_render_rejects_missing_output_path_without_noop_success():
    scene = _base_scene()
    scene.recipe.output.path = None

    with pytest.raises(ValueError, match="render path"):
        scene.render()

    assert scene.last_validation_report is None
