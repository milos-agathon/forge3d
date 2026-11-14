"""Fixed-bin binary32 arithmetic, residency and D02 native rendering proof."""

import os
from fractions import Fraction

import numpy as np
import pytest

from forge3d import RasterRecipe, RasterRecipeSource
from forge3d.preparation import PreparationGrid

rasterio = pytest.importorskip("rasterio")
pytestmark = pytest.mark.skipif(
    getattr(rasterio, "__forge3d_stub__", False), reason="requires real GeoTIFF I/O"
)


def _source(root, name, data, grid, *, nodata=None):
    path = root / (name + ".tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=1,
        dtype=data.dtype,
        height=grid.height,
        width=grid.width,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    ) as ds:
        ds.write(data, 1)
    return RasterRecipeSource.from_file(path, name)


def _inputs(tmp_path, grid):
    min_normal = np.finfo(np.float32).tiny
    max_half = np.finfo(np.float32).max / np.float32(2)
    a = np.resize(
        np.array(
            [
                1,
                3,
                1,
                5,
                -1,
                -1,
                np.nan,
                np.inf,
                -999,
                max_half,
                np.finfo(np.float32).max,
                min_normal,
                0,
                2**-103,
                np.nextafter(np.float32(2**-103), np.float32(1)),
                np.nextafter(np.float32(1), np.float32(2)),
                0,
                0,
            ],
            dtype="float32",
        ),
        (grid.height, grid.width),
    )
    b = np.resize(
        np.array(
            [
                3,
                1,
                1,
                3,
                1,
                -1,
                1,
                1,
                1,
                max_half,
                -np.finfo(np.float32).max,
                0,
                0,
                2**-103,
                -(2**-103),
                -1,
                -0.0,
                -4,
            ],
            dtype="float32",
        ),
        a.shape,
    )
    mask = np.ones(a.shape, dtype="uint8")
    mask[8, 8] = 0
    sources = (
        _source(tmp_path, "a", a, grid, nodata=-999),
        _source(tmp_path, "b", b, grid),
        _source(tmp_path, "mask", mask, grid),
    )
    return sources, a, b, mask


PALETTE = [
    (160, 20, 15, 255),
    (230, 120, 60, 255),
    (30, 160, 100, 255),
    (20, 60, 150, 255),
]
BINS = [-0.5, 0, 0.5]


def _reference(a, b, enabled, bins, right):
    # Fraction arithmetic is independent of WGSL and avoids using the same
    # division implementation. f64 rounding is exact for these fixtures.
    if (
        not enabled
        or a == -999
        or any(
            v != 0 and not 2**-103 <= abs(v) <= float(np.finfo(np.float32).max) / 2
            for v in (a, b)
        )
    ):
        return None, 0
    x, y = Fraction(float(a)), Fraction(float(b))
    if x + y == 0:
        return None, 0
    num = np.float32(float(x - y))
    den = np.float32(float(x + y))
    exact = (
        float(Fraction(float(num)) / Fraction(float(den))) if num else float(num / den)
    )
    with np.errstate(over="ignore"):
        value = float(np.float32(exact))
    if not np.isfinite(value):
        return None, 0
    return value, 1 + sum(value > edge if right else value >= edge for edge in bins)


def _physical():
    if os.environ.get("FORGE3D_RUN_GPU_ANAMNESIS") != "1":
        pytest.skip("enable the physical native lane with FORGE3D_RUN_GPU_ANAMNESIS=1")
    import forge3d as f3d
    from forge3d._native import get_native_module

    probe = f3d.device_probe(os.environ.get("WGPU_BACKEND"))
    assert probe["device_type"] in {"DiscreteGpu", "IntegratedGpu"}, probe
    assert not probe.get("software_fallback", False), probe
    return get_native_module()


def test_declaration_and_grid_contract(tmp_path):
    from forge3d.observation_time import ObservationRecord

    grid = PreparationGrid("EPSG:32635", 17, 9, (570000, 10, 0, 3980000, 0, -10))
    sources, _, _, _ = _inputs(tmp_path, grid)
    source, b, mask = sources
    observation = ObservationRecord(
        "Acquisition",
        "2023-07-18T11:00:00+02:00",
        "2023-07-18T11:00:00+02:00",
        "2023-08-01T00:00:00Z",
        {"band": "B08"},
    )
    a = RasterRecipeSource.from_file(
        source.path, source.source_id, observation=observation
    )
    recipe = RasterRecipe(a, b, mask, grid, BINS, PALETTE)
    replay = RasterRecipe.from_json(recipe.to_json())
    assert replay.to_json() == recipe.to_json()
    assert replay.identity == recipe.identity
    assert replay.legend()["observed_at"] == ["2023-07-18T09:00:00Z", None]
    assert replay.to_dict()["inputs"][2]["sha256"] == mask.sha256
    assert recipe.legend()["entries"][1]["label"] == "[-0.5, 0.0)"
    assert recipe.metrics() == {}
    with pytest.raises(ValueError, match="representable"):
        recipe.set_bins([-0.1, 0, 0.5])
    moved = tmp_path / "moved.tif"
    moved.write_bytes(a.path.read_bytes())
    recipe.replace_inputs(
        RasterRecipeSource.from_file(moved, a.source_id, observation=observation),
        b,
        mask,
    )
    assert recipe.identity == replay.identity
    with pytest.raises(ValueError, match="representable"):
        recipe.set_bins([-0.5, 0, 2**53 + 1])
    moved_grid = PreparationGrid(
        grid.crs, grid.width, grid.height, (570010, *grid.geotransform[1:])
    )
    with pytest.raises(ValueError, match="Grid mismatch"):
        a._load(moved_grid, mask=False)
    cast = _source(tmp_path, "f64", np.ones((9, 17), dtype="float64"), grid)
    with pytest.raises(ValueError, match="float32"):
        cast._load(grid, mask=False)


@pytest.mark.parametrize("right", [False, True])
def test_physical_recipe_classes_and_release(tmp_path, right):
    native = _physical()
    grid = PreparationGrid("EPSG:32635", 17, 9, (570000, 10, 0, 3980000, 0, -10))
    sources, a, b, mask = _inputs(tmp_path, grid)
    recipe = RasterRecipe(*sources, grid, BINS, PALETTE, right=right)
    before = native.global_memory_metrics()["total_bytes"]

    def compare(edges):
        expected = [
            _reference(x, y, bit, edges, right)
            for x, y, bit in zip(a.flat, b.flat, mask.flat)
        ]
        expected_values = np.array(
            [np.nan if v is None else v for v, _ in expected], dtype=np.float32
        )
        expected_ids = np.array([i for _, i in expected])
        counts = recipe.counts()
        assert list(counts.values()) == np.bincount(expected_ids, minlength=5).tolist()
        assert recipe.sample(8, 8) == (None, 0)
        values, ids = recipe.export_cells()
        np.testing.assert_array_equal(ids.ravel(), expected_ids)
        np.testing.assert_array_equal(values.ravel(), expected_values)
        np.testing.assert_array_equal(
            values.ravel()[expected_ids > 0].view(np.uint32),
            expected_values[expected_ids > 0].view(np.uint32),
        )
        assert values[0, 5].view(np.uint32) == 0x80000000

    compare(BINS)
    observed = recipe.metrics()
    assert (
        observed["normalization_dispatches"]
        == observed["classification_dispatches"]
        == 1
    )
    assert observed["parameter_upload_bytes"] == 2 * 24 + 4 * 8
    assert observed["readback_bytes"] == 20 + 8 + 9 * 17 * 8
    edges = [-0.25, 0, 0.25]
    recipe.set_bins(edges)
    compare(edges)
    revised = recipe.metrics()
    assert (
        revised["normalization_dispatches"] == 1
        and revised["classification_dispatches"] == 2
    )
    assert revised["input_upload_bytes"] == observed["input_upload_bytes"]
    assert (
        revised["parameter_upload_bytes"] - observed["parameter_upload_bytes"]
        == 24 + 4 * 8
    )
    with pytest.raises(ValueError, match="outside"):
        recipe.sample(9, 0)
    a, b = b, a
    recipe.replace_inputs(
        sources[1], _source(tmp_path, "replacement", b, grid, nodata=-999), sources[2]
    )
    recipe.counts()
    assert recipe.metrics()["normalization_dispatches"] == 2
    assert recipe.metrics()["parameter_upload_bytes"] == 5 * 24 + 3 * 4 * 8
    record = recipe.to_json()
    broken = RasterRecipeSource(sources[0].path, "bad-content", "0" * 64)
    recipe.replace_inputs(broken, sources[1], sources[2])
    with pytest.raises(ValueError, match="content changed"):
        recipe.sample(0, 0)
    recipe.close()
    assert native.global_memory_metrics()["total_bytes"] == before
    assert RasterRecipe.from_json(record).legend()["sources"][1] == "replacement"


def test_physical_render_uses_resident_d02_palette(tmp_path):
    import forge3d as f3d

    _physical()
    grid = PreparationGrid("EPSG:32635", 17, 17, (570000, 10, 0, 3980000, 0, -10))
    a = np.ones((17, 17), dtype="float32")
    b = np.tile(np.repeat([3, 2, 1, 0.25], [4, 4, 4, 5]), (17, 1)).astype("float32")
    mask = np.ones_like(a, dtype="uint8")
    mask[:, :2] = 0
    sources = (
        _source(tmp_path, "left", a, grid),
        _source(tmp_path, "right", b, grid),
        _source(tmp_path, "mask", mask, grid),
    )
    recipe = RasterRecipe(*sources, grid, BINS, PALETTE)

    def scene():
        return f3d.MapScene(
            terrain=f3d.TerrainSource(
                data=a, crs=grid.crs, metadata=grid.to_dict() | {"resolution": [10, 10]}
            ),
            camera=f3d.OrbitCamera(distance=300, azimuth_deg=-90, elevation_deg=60),
            lighting=f3d.LightingPreset(sun_direction=(0.3, 0.7, 0.6)),
            output=f3d.OutputSpec(width=192, height=192),
        )

    from PIL import Image

    resident_scene, cpu_scene = scene(), scene()
    wrong_scene = scene()
    wrong_scene.recipe.target_crs = "EPSG:32634"
    rejected = tmp_path / "rejected.png"
    with pytest.raises(ValueError, match="CRS mismatch"):
        wrong_scene.render(str(rejected), raster_recipe=recipe)
    assert not rejected.exists()
    for i, edges in enumerate([BINS, [-0.25, 0, 0.25]]):
        recipe.set_bins(edges)
        resident = tmp_path / f"resident-{i}.png"
        cpu = tmp_path / f"cpu-{i}.png"
        resident_scene.render(str(resident), raster_recipe=recipe)
        assert resident_scene.last_render_metadata["thematic_legend"] == recipe.legend()
        assert (
            resident_scene.last_render_metadata["raster_recipe_identity"]
            == recipe.identity
        )
        assert recipe.metrics()["readback_bytes"] == 0
        ids = (np.digitize((a - b) / (a + b), edges) + 1).astype(np.uint8)
        ids[mask == 0] = 0
        cpu_scene.recipe.layers = [
            f3d.RasterOverlay(
                "classes",
                data=ids,
                crs=grid.crs,
                style=f3d.CategoricalRasterStyle(dict(enumerate(PALETTE, 1)), nodata=0),
            )
        ]
        cpu_scene.render(str(cpu))
        pixels = np.asarray(Image.open(resident))
        assert len(np.unique(pixels.reshape(-1, 4), axis=0)) == len(np.unique(ids))
        np.testing.assert_array_equal(pixels, np.asarray(Image.open(cpu)))
    assert recipe.metrics()["normalization_dispatches"] == 1
    assert recipe.metrics()["classification_dispatches"] == 2


def test_world_annotation_matches_resident_terrain_without_resolution(
    tmp_path, monkeypatch
):
    import forge3d as f3d
    from forge3d import map_scene as ms
    from forge3d._map_scene_projection import TerrainProjector

    _physical()
    grid = PreparationGrid("EPSG:32635", 33, 33, (570000, 100, 0, 3998000, 0, -100))
    height = np.zeros((33, 33), dtype=np.float32)
    # The negative class occupies an off-centre patch. The annotation lies
    # well inside the patch, whose position is checked in the native AOV.
    a = np.ones_like(height)
    b = np.ones_like(height)
    b[4:13, 22:31] = 5
    mask = np.ones_like(height, dtype=np.uint8)
    recipe = RasterRecipe(
        _source(tmp_path, "left", a, grid),
        _source(tmp_path, "right", b, grid),
        _source(tmp_path, "mask", mask, grid),
        grid,
        BINS,
        PALETTE,
    )
    world = grid.xy(26, 8)
    feature = {
        "type": "Feature",
        "id": "control",
        "properties": {"name": "Control"},
        "geometry": {"type": "Point", "coordinates": world},
    }
    scene = f3d.MapScene(
        terrain=f3d.TerrainSource(data=height, crs=grid.crs, metadata=grid.to_dict()),
        camera=f3d.OrbitCamera(
            distance=5200, azimuth_deg=-90, elevation_deg=16, fov_deg=58
        ),
        lighting=f3d.LightingPreset(
            settings={"cli_params": {"camera_mode": "mesh:zup:north"}}
        ),
        layers=[
            f3d.LabelLayer.from_features(
                [feature],
                crs=grid.crs,
                target_crs=grid.crs,
                terrain_sampling="none",
                occlusion="none",
                metadata={"coordinate_space": "world"},
                glyph_atlas={"glyphs": sorted(set("Control"))},
            )
        ],
        output=f3d.OutputSpec(width=256, height=256),
    )
    compiled = scene.compile_plan()
    assignment = compiled.label_plans["labels"].accepted
    assert len(assignment) == 1, compiled.label_plans["labels"].to_dict()
    projector = TerrainProjector(scene.recipe)
    assert projector.span == 3300
    x, y, _ = projector.project(world)
    np.testing.assert_allclose(assignment[0].candidate.anchor[:2], (x, y))
    original = ms._render_terrain_renderer_result
    albedo = []

    def observe(*args, **kwargs):
        result = original(*args, **kwargs)
        albedo.append(result.aov_frame.albedo())
        return result

    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    # Request one diagnostic attachment through the ordinary renderer.
    scene.recipe.output.aovs = ("albedo",)
    scene.render(str(tmp_path / "world-label.png"), raster_recipe=recipe)
    expected = np.asarray(PALETTE[0][:3]) / 255
    linear = np.where(
        expected <= 0.04045, expected / 12.92, ((expected + 0.055) / 1.055) ** 2.4
    )
    np.testing.assert_allclose(
        albedo[0][int(y), int(x), :3],
        linear,
        atol=np.spacing(linear.astype(np.float16)).max(),
    )
