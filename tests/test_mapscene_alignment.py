from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import forge3d as f3d
import forge3d.map_scene as map_scene
from forge3d.alignment import alignment_residual, reproject_dem_to_target, resample_raster_to_grid
from forge3d.crs import proj_available


pytestmark = pytest.mark.skipif(not proj_available(), reason="CRS transform backend unavailable")


def _write_raster(path: Path, data: np.ndarray, *, crs: str = "EPSG:4326") -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(-1.0, 1.0, 0.5, 0.5)
    nodata = 0 if np.issubdtype(data.dtype, np.unsignedinteger) else -9999
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_mapscene_target_crs_reprojects_inline_vector_features() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"width": 4, "height": 4, "source_id": "mercator-dem"},
        ),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        target_crs="EPSG:3857",
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:4326",
                features=[
                    {
                        "id": "road",
                        "geometry": {"type": "LineString", "coordinates": [(0.0, 0.0), (1.0, 1.0)]},
                    }
                ],
            )
        ],
    )

    layer = scene.recipe.layers[0]
    coords = layer.features[0]["geometry"]["coordinates"]
    report = scene.validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert scene.recipe.target_crs == "EPSG:3857"
    assert layer.crs == "EPSG:3857"
    assert layer.metadata["source_crs"] == "EPSG:4326"
    assert coords[0] == pytest.approx([0.0, 0.0])
    assert coords[1][0] == pytest.approx(111319.49, rel=1.0e-4)
    assert "crs_mismatch" not in codes
    assert "alignment_transform_applied" in codes
    assert report.supported_features["mapscene.alignment"] == "supported"


def test_alignment_report_records_transformed_layers() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"width": 4, "height": 4, "source_id": "mercator-dem"},
        ),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        target_crs="EPSG:3857",
        layers=[
            f3d.LabelLayer.from_features(
                [
                    {
                        "type": "Feature",
                        "id": "city",
                        "properties": {"name": "A"},
                        "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
                    }
                ],
                crs="EPSG:4326",
                layer_id="labels",
                glyph_atlas={"glyphs": ["A"]},
            )
        ],
    )

    report = scene.alignment_report()
    label = scene.recipe.layers[0].labels[0]

    assert report == f3d.alignment_report(scene)
    assert report["target_crs"] == "EPSG:3857"
    assert report["layers"][0]["layer_id"] == "labels"
    assert report["layers"][0]["transform_applied"] is True
    assert label["geometry"]["coordinates"][0] == pytest.approx(111319.49, rel=1.0e-4)


def test_reproject_dem_to_target_records_exact_grid_metadata(tmp_path: Path) -> None:
    src_path = tmp_path / "dem4326.tif"
    _write_raster(src_path, np.arange(16, dtype=np.float32).reshape(4, 4))

    result = reproject_dem_to_target(src_path, "EPSG:3857", output_path=tmp_path / "dem3857.tif")

    assert result["array"].ndim == 2
    assert Path(result["path"]).exists()
    metadata = result["metadata"]
    assert metadata["source_crs"].upper().endswith("4326")
    assert metadata["target_crs"] == "EPSG:3857"
    assert metadata["width"] == result["array"].shape[1]
    assert metadata["height"] == result["array"].shape[0]
    assert len(metadata["geotransform"]) == 6
    assert metadata["resolution"][0] > 0.0
    assert metadata["resolution"][1] > 0.0


def test_mapscene_target_crs_reprojects_geotiff_terrain_into_recipe(tmp_path: Path) -> None:
    src_path = tmp_path / "dem4326.tif"
    _write_raster(src_path, np.arange(16, dtype=np.float32).reshape(4, 4), crs="EPSG:4326")

    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            path=src_path,
            crs="EPSG:4326",
            metadata={"source_id": "dem4326"},
            elevation_sampling_available=True,
        ),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        target_crs="EPSG:3857",
    )

    metadata = scene.recipe.terrain.metadata
    report = scene.validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert scene.recipe.terrain.data is not None
    assert scene.recipe.terrain.crs == "EPSG:3857"
    assert metadata["source_crs"].upper().endswith("4326")
    assert metadata["target_crs"] == "EPSG:3857"
    assert metadata["alignment_transform_applied"] is True
    assert metadata["alignment_kind"] == "terrain_reproject"
    assert metadata["source_path"] == str(src_path)
    assert len(metadata["geotransform"]) == 6
    assert "alignment_transform_applied" in codes


def test_resample_raster_to_explicit_grid(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    src_path = tmp_path / "overlay.tif"
    _write_raster(src_path, np.arange(16, dtype=np.uint8).reshape(4, 4))
    target_grid = {
        "crs": "EPSG:4326",
        "transform": from_origin(-1.0, 1.0, 1.0, 1.0),
        "width": 2,
        "height": 2,
        "nodata": 0,
    }

    result = resample_raster_to_grid(src_path, target_grid, resampling="nearest")

    assert result["array"].shape == (2, 2)
    assert result["metadata"]["target_crs"] == "EPSG:4326"
    assert result["metadata"]["width"] == 2
    assert result["metadata"]["height"] == 2
    assert result["metadata"]["geotransform"] == pytest.approx(list(target_grid["transform"].to_gdal()))
    assert result["array"].dtype == np.dtype("uint8")


def test_mapscene_render_resamples_geotiff_overlay_to_terrain_grid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    overlay_path = tmp_path / "classes.tif"
    _write_raster(overlay_path, np.arange(4, dtype=np.uint8).reshape(2, 2), crs="EPSG:3857")
    target_transform = from_origin(100.0, 200.0, 10.0, 10.0)
    calls: dict[str, object] = {}

    def fake_resample(source, target_grid, *, output_path=None, resampling="nearest", dst_nodata=None):
        calls["source"] = str(source)
        calls["target_grid"] = dict(target_grid)
        calls["resampling"] = resampling
        calls["dst_nodata"] = dst_nodata
        return {
            "array": np.full((int(target_grid["height"]), int(target_grid["width"])), 255, dtype=np.uint8),
            "metadata": {},
            "profile": {},
            "path": None,
        }

    def fake_terrain_result(recipe, heightmap):
        rgba = np.zeros((int(recipe.output.height), int(recipe.output.width), 4), dtype=np.uint8)
        rgba[..., 3] = 255
        return map_scene._MapSceneNativeRenderResult(
            rgba=rgba,
            aov_frame=None,
            hdr_frame=None,
            metadata={"samples_used": 1, "target_samples": 1, "denoiser_used": "none", "adaptive": False},
        )

    monkeypatch.setattr(f3d.alignment, "resample_raster_to_grid", fake_resample)
    monkeypatch.setattr(map_scene, "_render_terrain_renderer_result", fake_terrain_result)
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((3, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={
                "source_id": "dem",
                "width": 4,
                "height": 3,
                "geotransform": list(target_transform.to_gdal()),
                "resolution": [10.0, 10.0],
            },
            elevation_sampling_available=True,
        ),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=8, height=6, path=str(tmp_path / "aligned-overlay.png")),
        target_crs="EPSG:3857",
        layers=[
            f3d.RasterOverlay(
                layer_id="classes",
                path=overlay_path,
                crs="EPSG:3857",
                opacity=1.0,
                metadata={"source_id": "classes", "alignment_resampling": "mode", "nodata": 0},
            )
        ],
    )

    report = scene.render()
    target_grid = calls["target_grid"]

    assert report.status == "ok"
    assert Path(scene.last_render_path).exists()
    assert calls["source"] == str(overlay_path)
    assert calls["resampling"] == "mode"
    assert calls["dst_nodata"] == 0
    assert target_grid["crs"] == "EPSG:3857"
    assert target_grid["width"] == 4
    assert target_grid["height"] == 3
    assert target_grid["transform"] == list(target_transform.to_gdal())


def test_alignment_residual_roundtrip_and_misregistered_controls() -> None:
    roundtrip = alignment_residual([(0.0, 0.0), (1.0, 1.0)], "EPSG:4326", "EPSG:3857")
    bad = alignment_residual([(1.0, 1.0)], "EPSG:4326", "EPSG:3857", expected=[(1.0, 1.0)])

    assert roundtrip["mode"] == "roundtrip"
    assert roundtrip["max"] < 1.0e-6
    assert bad["mode"] == "expected"
    assert bad["max"] > 100000.0


def test_mapscene_validation_reports_alignment_residual_and_resolution_mismatch() -> None:
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=np.zeros((4, 4), dtype=np.float32),
            crs="EPSG:3857",
            metadata={"width": 4, "height": 4, "resolution": [10.0, 10.0], "source_id": "dem"},
            elevation_sampling_available=True,
        ),
        lighting=f3d.LightingPreset(),
        output=f3d.OutputSpec(width=64, height=64),
        target_crs="EPSG:3857",
        layers=[
            f3d.RasterOverlay(
                layer_id="overlay",
                path="overlay.tif",
                crs="EPSG:3857",
                metadata={
                    "source_id": "overlay",
                    "resolution": [30.0, 30.0],
                    "alignment_residual": {
                        "max": 2.5,
                        "threshold": 1.0,
                        "units": "m",
                        "source_crs": "EPSG:4326",
                        "target_crs": "EPSG:3857",
                    },
                },
            )
        ],
    )

    report = scene.validate()
    codes = [diagnostic.code for diagnostic in report.diagnostics]

    assert "resolution_mismatch" in codes
    assert "alignment_residual" in codes


def test_swiss_viewer_uses_shared_raster_alignment_helpers() -> None:
    example_path = Path("examples/swiss_terrain_landcover_viewer.py")
    if not example_path.exists():
        pytest.skip("example 'swiss_terrain_landcover_viewer.py' is untracked/local-only")
    source = example_path.read_text(encoding="utf-8")

    assert "def ensure_dem_in_target_crs" not in source
    assert "from rasterio.warp import reproject" not in source
    assert "from rasterio.enums import Resampling" not in source
    assert "reproject_dem_to_target(" in source
    assert "resample_raster_to_grid(" in source
