from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import forge3d as f3d

from _terrain_runtime import terrain_rendering_available
from tests._ssim import ssim


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden" / "recipes"
UPDATE_GOLDENS = os.environ.get("FORGE3D_UPDATE_RECIPE_GOLDENS") == "1"
ARTIFACT_DIR = (
    Path(os.environ["FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR"])
    if os.environ.get("FORGE3D_RECIPE_GOLDEN_ARTIFACT_DIR")
    else None
)
SSIM_MIN = 0.995
MEAN_ABS_MAX = 2.0


@dataclass(frozen=True)
class RecipeGolden:
    scene_id: str
    family: str
    build: Callable[[Path], f3d.MapScene]
    expected_features: tuple[str, ...]
    bit_depth: int = 8

    @property
    def golden_path(self) -> Path:
        return GOLDEN_DIR / f"{self.scene_id}.png"

    @property
    def command(self) -> str:
        return f"pytest tests/test_recipe_goldens.py -k {self.scene_id}"


def _heightmap(size: int = 8) -> np.ndarray:
    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return (0.25 * xx + 0.75 * yy).astype(np.float32)


def _write_rgba_png(path: Path, rgba: np.ndarray) -> Path:
    f3d.numpy_to_png(path, np.ascontiguousarray(rgba, dtype=np.uint8))
    return path


def _material_map_assets(tmp_path: Path) -> dict[str, str]:
    size = 64
    coords = np.linspace(0.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)

    normal = np.zeros((size, size, 4), dtype=np.uint8)
    normal[..., 0] = np.clip(128.0 + 92.0 * np.sin(xx * np.pi * 10.0), 0.0, 255.0).astype(np.uint8)
    normal[..., 1] = np.clip(128.0 + 56.0 * np.cos(yy * np.pi * 8.0), 0.0, 255.0).astype(np.uint8)
    normal[..., 2] = 208
    normal[..., 3] = 255

    roughness = np.zeros((size, size, 4), dtype=np.uint8)
    rough = np.clip(54.0 + 174.0 * (0.5 + 0.5 * np.sin((xx + yy) * np.pi * 7.0)), 0.0, 255.0).astype(np.uint8)
    roughness[..., :3] = rough[..., None]
    roughness[..., 3] = 255

    mask = np.zeros((size, size, 4), dtype=np.uint8)
    rings = ((np.floor(xx * 8.0) + np.floor(yy * 8.0)) % 2.0).astype(np.uint8) * 255
    mask[..., :3] = rings[..., None]
    mask[..., 3] = 255

    return {
        "normal_path": str(_write_rgba_png(tmp_path / "material-normal.png", normal)),
        "roughness_path": str(_write_rgba_png(tmp_path / "material-roughness.png", roughness)),
        "mask_path": str(_write_rgba_png(tmp_path / "material-mask.png", mask)),
    }


def _base_scene(
    tmp_path: Path,
    scene_id: str,
    *,
    layers: list[object] | None = None,
    width: int = 96,
    height: int = 64,
    samples: int = 1,
    aovs: tuple[str, ...] = (),
    hdr: bool = False,
    bit_depth: int = 8,
    map_furniture: f3d.MapFurnitureLayer | None = None,
    terrain_metadata: dict[str, object] | None = None,
    lighting_settings: dict[str, object] | None = None,
) -> f3d.MapScene:
    metadata = {
        "source_id": f"{scene_id}-dem",
        "width": 8,
        "height": 8,
        "asset_status": "fixture",
        "bounds": (-122.5, 46.6, -121.9, 47.0),
    }
    if terrain_metadata:
        metadata.update(terrain_metadata)
    return f3d.MapScene(
        terrain=f3d.TerrainSource(
            data=_heightmap(),
            crs="EPSG:32610",
            metadata=metadata,
            elevation_sampling_available=True,
        ),
        camera=f3d.OrbitCamera(target=(0.0, 0.0, 0.0), distance=800.0, azimuth_deg=35.0),
        lighting=f3d.LightingPreset(name="rainier_showcase", intensity=1.15, settings=lighting_settings),
        output=f3d.OutputSpec(
            width=width,
            height=height,
            format="png",
            path=str(tmp_path / f"{scene_id}.png"),
            samples=samples,
            aovs=aovs,
            hdr=hdr,
            bit_depth=bit_depth,
        ),
        layers=layers or [],
        map_furniture=map_furniture,
        reproducibility_profile=f3d.ReproducibilityProfile(seed=2026),
    )


def _terrain_raster(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_terrain_raster",
        layers=[
            f3d.RasterOverlay(
                layer_id="ortho",
                path="fixtures/ortho.tif",
                crs="EPSG:32610",
                opacity=0.72,
                metadata={"source_id": "ortho-fixture", "width": 8, "height": 8, "asset_status": "fixture"},
            )
        ],
    )


def _vector_labels(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_vector_labels",
        layers=[
            f3d.VectorOverlay(
                layer_id="roads",
                crs="EPSG:32610",
                features=[
                    {"id": "a", "geometry": {"type": "LineString", "coordinates": [(0.1, 0.2), (0.9, 0.75)]}},
                    {"id": "b", "geometry": {"type": "LineString", "coordinates": [(0.12, 0.78), (0.88, 0.28)]}},
                ],
                width_px=4,
                line_cap="round",
                line_join="round",
                dash_array=[10, 5],
                style={"version": 8, "layers": [{"id": "roads", "type": "line", "paint": {"line-color": "#f9fafb"}}]},
            ),
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {"id": "summit", "text": "Summit", "geometry": {"type": "Point", "coordinates": (34.0, 20.0, 0.0)}},
                    {"id": "trail", "text": "Trail", "geometry": {"type": "Point", "coordinates": (68.0, 44.0, 0.0)}},
                ],
                glyph_atlas={"glyphs": sorted(set("SummitTrail"))},
            ),
        ],
    )


def _label_halo_depth(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_label_halo_depth",
        width=128,
        height=80,
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": (28.0, 26.0, 0.25)},
                        "typography": {
                            "color": [1.0, 1.0, 1.0, 1.0],
                            "halo_color": [0.02, 0.02, 0.02, 0.92],
                            "halo_width_px": 3.0,
                        },
                    },
                    {
                        "id": "summit",
                        "text": "Summit",
                        "geometry": {"type": "Point", "coordinates": (72.0, 50.0, 0.20)},
                        "typography": {
                            "color": [0.12, 0.16, 0.18, 1.0],
                            "halo_color": [1.0, 1.0, 1.0, 0.88],
                            "halo_width_px": 2.0,
                        },
                    },
                    {
                        "id": "behind",
                        "text": "Behind",
                        "geometry": {"type": "Point", "coordinates": (28.0, 26.0, 0.85)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontSummitBehind"))},
                occlusion="terrain",
                metadata={
                    "depth_occlusion": {
                        "image": np.full((8, 8), 0.5, dtype=np.float32).tolist(),
                        "source": "recipe_depth_aov",
                        "bias": 0.0,
                    }
                },
            )
        ],
    )


def _label_occlusion_ridge(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_label_occlusion_ridge",
        width=128,
        height=80,
        layers=[
            f3d.LabelLayer(
                layer_id="labels",
                labels=[
                    {
                        "id": "front",
                        "text": "Front",
                        "geometry": {"type": "Point", "coordinates": (34.0, 26.0, 0.0)},
                        "typography": {
                            "color": [1.0, 1.0, 1.0, 1.0],
                            "halo_color": [0.02, 0.02, 0.02, 0.92],
                            "halo_width_px": 3.0,
                        },
                    },
                    {
                        "id": "behind-ridge",
                        "text": "Hidden",
                        "geometry": {"type": "Point", "coordinates": (34.0, 26.0, 0.95)},
                    },
                ],
                glyph_atlas={"glyphs": sorted(set("FrontHidden"))},
                occlusion="terrain",
            )
        ],
    )


def _vector_stroke_quality(
    tmp_path: Path,
    *,
    scene_id: str = "mapscene_vector_stroke_quality",
    width: int = 128,
    height: int = 80,
) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        scene_id,
        width=width,
        height=height,
        layers=[
            f3d.VectorOverlay(
                layer_id="cartography",
                crs="EPSG:32610",
                features=[
                    {
                        "id": "hairpin",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [(0.06, 0.74), (0.30, 0.18), (0.52, 0.74), (0.74, 0.22), (0.94, 0.74)],
                        },
                    },
                    {
                        "id": "dashed-boundary",
                        "geometry": {"type": "LineString", "coordinates": [(0.08, 0.10), (0.92, 0.10)]},
                    },
                    {
                        "id": "park-with-hole",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [(0.10, 0.32), (0.38, 0.32), (0.38, 0.62), (0.10, 0.62), (0.10, 0.32)],
                                [(0.19, 0.41), (0.30, 0.41), (0.30, 0.53), (0.19, 0.53), (0.19, 0.41)],
                            ],
                        },
                    },
                ],
                width_px=6,
                line_cap="round",
                line_join="round",
                dash_array=[12, 7],
                style={
                    "version": 8,
                    "layers": [
                        {
                            "id": "cartography",
                            "type": "line",
                            "paint": {"line-color": "#f8fafc", "line-width": 6, "fill-color": "#2563eb"},
                        }
                    ],
                },
            )
        ],
    )


def _vector_stroke_quality_4x(tmp_path: Path) -> f3d.MapScene:
    return _vector_stroke_quality(
        tmp_path,
        scene_id="mapscene_vector_stroke_quality_4x",
        width=256,
        height=160,
    )


def _offline_aovs(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_offline_aovs",
        samples=4,
        aovs=("albedo", "normal", "depth"),
        hdr=True,
    )


def _buildings(tmp_path: Path) -> f3d.MapScene:
    roof_types = ("flat", "gabled", "hipped", "pyramidal")
    features = []
    for idx, roof_type in enumerate(roof_types):
        x0 = 0.08 + idx * 0.22
        x1 = x0 + 0.15
        y0 = 0.24 if idx % 2 == 0 else 0.34
        y1 = y0 + 0.30
        features.append(
            {
                "id": f"b-{roof_type}",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]],
                },
                "properties": {
                    "height": 22.0 + idx * 7.0,
                    "roof:shape": roof_type,
                    "building:material": "brick" if idx % 2 else "concrete",
                },
            }
        )
    building = f3d.MapSceneBuildingLayer(
        layer_id="buildings",
        source={"source_id": "inline-buildings", "asset_status": "fixture"},
        support_level="supported",
        geometry_count=len(features),
        material_status="scalar_pbr_underdeveloped",
        features=features,
        metadata={"source_id": "inline-buildings", "asset_status": "fixture"},
    )
    return _base_scene(tmp_path, "mapscene_buildings", layers=[building], width=128, height=88)


def _furniture(tmp_path: Path) -> f3d.MapScene:
    furniture = f3d.MapFurnitureLayer(
        title="Recipe Golden",
        legend={"items": [{"label": "Forest", "color": "#2f855a"}, {"label": "Snow", "color": "#f8fafc"}]},
        scale_bar={"length_m": 1000, "units": "km", "location": "lower_left", "geodesic": True},
        north_arrow={"location": "upper_right", "size": 34},
        graticule={"bounds": (-122.5, 46.6, -121.9, 47.0), "interval_deg": 0.2, "include_labels": True},
    )
    return _base_scene(tmp_path, "mapscene_furniture_graticule", map_furniture=furniture, width=128, height=88)


def _alignment(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(
        tmp_path,
        "mapscene_alignment_utm",
        layers=[
            f3d.VectorOverlay(
                layer_id="aligned-boundary",
                crs="EPSG:4326",
                features=[{"id": "bbox", "geometry": {"type": "LineString", "coordinates": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9)]}}],
                metadata={"crs_policy": "explicit_transform", "crs_transform": "fixture-transform"},
                width_px=3,
            )
        ],
    )


def _material_maps(tmp_path: Path) -> f3d.MapScene:
    material_maps = _material_map_assets(tmp_path)
    return _base_scene(
        tmp_path,
        "mapscene_material_maps",
        width=128,
        height=80,
        terrain_metadata={"material_maps": material_maps},
        lighting_settings={
            "albedo_mode": "material",
            "colormap_strength": 0.0,
            "exaggeration": 1.35,
        },
    )


def _png16_color(tmp_path: Path) -> f3d.MapScene:
    return _base_scene(tmp_path, "mapscene_png16_color", bit_depth=16, width=80, height=48)


RECIPE_GOLDENS = (
    RecipeGolden("mapscene_terrain_raster", "terrain_raster", _terrain_raster, ("mapscene.render_png", "mapscene.render_backend")),
    RecipeGolden("mapscene_vector_labels", "labels_vectors", _vector_labels, ("mapscene.vector_composite", "mapscene.label_composite")),
    RecipeGolden("mapscene_label_halo_depth", "labels_depth_occlusion", _label_halo_depth, ("mapscene.label_composite",)),
    RecipeGolden("mapscene_label_occlusion_ridge", "labels_depth_occlusion", _label_occlusion_ridge, ("mapscene.label_composite",)),
    RecipeGolden("mapscene_vector_stroke_quality", "vector_stroke_quality", _vector_stroke_quality, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_vector_stroke_quality_4x", "vector_stroke_quality", _vector_stroke_quality_4x, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_offline_aovs", "offline_accumulation", _offline_aovs, ("mapscene.offline_accumulation", "mapscene.aov_export")),
    RecipeGolden("mapscene_buildings", "buildings", _buildings, ("mapscene.building_composite", "mapscene.building_gpu_mesh_composite")),
    RecipeGolden("mapscene_furniture_graticule", "map_furniture", _furniture, ("mapscene.furniture_composite",)),
    RecipeGolden("mapscene_alignment_utm", "alignment_crs", _alignment, ("mapscene.vector_composite",)),
    RecipeGolden("mapscene_material_maps", "terrain_materials", _material_maps, ("mapscene.render_png",)),
    RecipeGolden("mapscene_png16_color", "output_color", _png16_color, ("mapscene.render_png_16bit",), bit_depth=16),
)


def _write_failure_artifacts(spec: RecipeGolden, actual: np.ndarray, expected: np.ndarray) -> None:
    if ARTIFACT_DIR is None:
        return
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diff = np.abs(actual[..., :3].astype(np.int16) - expected[..., :3].astype(np.int16)).astype(np.uint8)
    diff_rgba = np.concatenate([diff, np.full(diff.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_actual.png", actual)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_expected.png", expected)
    f3d.numpy_to_png(ARTIFACT_DIR / f"{spec.scene_id}_diff.png", diff_rgba)


def _assert_matches_golden(spec: RecipeGolden, actual_path: Path) -> None:
    if UPDATE_GOLDENS:
        spec.golden_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual_path, spec.golden_path)
        return

    assert spec.golden_path.exists(), (
        f"Missing recipe golden {spec.golden_path}. "
        "Regenerate with FORGE3D_UPDATE_RECIPE_GOLDENS=1."
    )
    actual = f3d.png_to_numpy(actual_path)
    expected = f3d.png_to_numpy(spec.golden_path)
    assert actual.shape == expected.shape
    mean_abs = float(np.mean(np.abs(actual[..., :3].astype(np.float32) - expected[..., :3].astype(np.float32))))
    score = ssim(actual[..., :3], expected[..., :3], data_range=255.0)
    if score < SSIM_MIN or mean_abs > MEAN_ABS_MAX:
        _write_failure_artifacts(spec, actual, expected)
    assert score >= SSIM_MIN, f"{spec.scene_id} SSIM too low: {score:.6f}"
    assert mean_abs <= MEAN_ABS_MAX, f"{spec.scene_id} mean absolute difference too high: {mean_abs:.4f}"


def test_recipe_golden_manifest_catalog_has_required_coverage() -> None:
    families = {spec.family for spec in RECIPE_GOLDENS}
    assert len(RECIPE_GOLDENS) >= 8
    assert len(families) >= 5
    assert {"terrain_raster", "labels_vectors", "offline_accumulation", "buildings", "map_furniture"} <= families


def test_recipe_golden_catalog_links_docs_gallery() -> None:
    gallery = (ROOT / "docs" / "gallery" / "index.md").read_text(encoding="utf-8")
    for spec in RECIPE_GOLDENS:
        assert spec.scene_id in gallery
        assert str(spec.golden_path.relative_to(ROOT)).replace("\\", "/") in gallery
        assert spec.command in gallery


def test_recipe_golden_gate_rejects_pixel_regression(tmp_path: Path) -> None:
    spec = RECIPE_GOLDENS[0]
    assert spec.golden_path.exists()
    corrupted = f3d.png_to_numpy(spec.golden_path).copy()
    corrupted[: corrupted.shape[0] // 2, : corrupted.shape[1] // 2, :3] = 255 - corrupted[
        : corrupted.shape[0] // 2,
        : corrupted.shape[1] // 2,
        :3,
    ]
    actual_path = tmp_path / "corrupted_recipe_golden.png"
    f3d.numpy_to_png(actual_path, corrupted)

    with pytest.raises(AssertionError):
        _assert_matches_golden(spec, actual_path)


@pytest.mark.parametrize("spec", RECIPE_GOLDENS, ids=lambda item: item.scene_id)
def test_recipe_goldens_render_and_match(tmp_path, spec: RecipeGolden) -> None:
    if not terrain_rendering_available():
        import pytest

        pytest.skip("Recipe goldens require a terrain-capable hardware-backed forge3d runtime")

    scene = spec.build(tmp_path)
    manifest = f3d.recipe_manifest(
        scene,
        golden_fixture_intent={
            "scene_id": spec.scene_id,
            "family": spec.family,
            "golden_path": str(spec.golden_path.relative_to(ROOT)).replace("\\", "/"),
            "command": spec.command,
            "backend": "gpu_terrain",
            "tolerance": {"ssim_min": SSIM_MIN, "mean_abs_max": MEAN_ABS_MAX},
        },
    )
    intent = manifest["golden_fixture_intent"]
    assert intent["scene_id"] == spec.scene_id
    assert intent["family"] == spec.family
    assert intent["backend"] == "gpu_terrain"

    report = scene.render()
    assert scene.last_render_backend == "gpu_terrain"
    for feature in spec.expected_features:
        assert report.supported_features[feature] == "supported"
    if spec.family == "buildings":
        metadata = scene.last_render_metadata or {}
        assert metadata["building_backend"] == "terrain_scatter_instanced_mesh"
        assert metadata["building_shadow_model"] == "terrain_csm_mesh_cast_receive"
    output_path = Path(scene.last_render_path or scene.recipe.output.path or "")
    assert output_path.exists()
    _assert_matches_golden(spec, output_path)
