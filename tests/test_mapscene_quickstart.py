import importlib.util
from pathlib import Path

import forge3d as f3d


QUICKSTART = Path("specs/004-mapscene-mvp/quickstart.md")
VECTOR_EXAMPLE = Path("examples/mapscene_vector_labels.py")
BUILDING_EXAMPLE = Path("examples/mapscene_buildings_labels.py")


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mapscene_quickstart_points_to_canonical_examples():
    text = QUICKSTART.read_text(encoding="utf-8")

    assert "python examples/mapscene_terrain_raster.py --output-dir" in text
    assert "python examples/mapscene_vector_labels.py --output-dir" in text
    assert "python examples/mapscene_buildings_labels.py --output-dir" in text
    assert "viewer_ipc" not in text
    assert "raw IPC" not in text


def test_quickstart_vector_labels_scenario_is_executable(tmp_path):
    module = _load_example(VECTOR_EXAMPLE)

    first = module.run_example(tmp_path / "first")
    second = module.run_example(tmp_path / "second")

    assert first["validation_status"] == "warning"
    assert first["render_status"] == "warning"
    assert first["render_backend"] == "native/offscreen"
    assert first["accepted_label_ids"] == second["accepted_label_ids"]
    assert first["rejected_label_reasons"] == second["rejected_label_reasons"]
    assert Path(first["png_path"]).exists()


def test_quickstart_negative_diagnostics_are_available_before_render():
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path="fixtures/dem.tif",
            crs="EPSG:32610",
            metadata={"width": 8, "height": 8, "asset_status": "fixture"},
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0),
        lighting=f3d.LightingPreset(name="daylight"),
        output=f3d.OutputSpec(width=64, height=64, format="png"),
        layers=[
            f3d.RasterOverlay(
                layer_id="wrong-crs",
                path="fixtures/wgs84.tif",
                crs="EPSG:4326",
                metadata={"asset_status": "fixture"},
            ),
            f3d.VectorOverlay(
                layer_id="bad-style",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "road",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
                style={"version": 8, "layers": [{"id": "heat", "type": "heatmap"}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "cafe",
                        "text": "cafe!",
                        "geometry": {"type": "Point", "coordinates": (16.0, 16.0, 0.0)},
                    }
                ],
                glyph_atlas={"glyphs": list("cafe")},
            ),
        ],
    )

    report = scene.validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert "crs_mismatch" in codes
    assert "unsupported_style_layer_type" in codes
    assert "missing_glyphs" in codes


def test_quickstart_building_scenario_preserves_blocking_diagnostics_in_bundle(tmp_path):
    module = _load_example(BUILDING_EXAMPLE)

    payload = module.run_example(tmp_path)

    assert payload["validation_status"] == "error"
    assert payload["bundle_status"] == "error"
    assert "pro_gated_path" in payload["diagnostic_codes"]
    assert Path(payload["bundle_path"]).exists()
