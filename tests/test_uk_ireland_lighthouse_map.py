from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pyproj")
pytest.importorskip("shapely")
from shapely.geometry import box


def _load_example_module():
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples"
    module_path = example_dir / "uk_ireland_lighthouse_map.py"
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("uk_ireland_lighthouse_map", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


uk_ie = _load_example_module()


def test_terrarium_rgb_to_height_decodes_zero_and_unit_step():
    rgb = np.array(
        [
            [[128, 0, 0], [128, 1, 0]],
            [[128, 0, 128], [127, 255, 255]],
        ],
        dtype=np.uint8,
    )
    height = uk_ie.terrarium_rgb_to_height(rgb)
    assert height.shape == (2, 2)
    assert height[0, 0] == 0.0
    assert height[0, 1] == 1.0
    assert height[1, 0] == 0.5
    assert height[1, 1] == -1.0 / 256.0


def test_decode_terrarium_tile_uses_rgb_decoding():
    image = Image.fromarray(
        np.array(
            [
                [[128, 0, 0], [128, 2, 0]],
                [[127, 255, 255], [128, 0, 64]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    tile = uk_ie.decode_terrarium_tile(image)
    assert tile.dtype == np.float32
    assert np.isclose(tile[0, 0], 0.0)
    assert np.isclose(tile[0, 1], 2.0)
    assert np.isclose(tile[1, 0], -1.0 / 256.0)
    assert np.isclose(tile[1, 1], 0.25)


def test_make_square_extent_preserves_center_and_equal_sides():
    extent = uk_ie.make_square_extent([(2.0, 3.0), (10.0, 7.0), (6.0, 12.0)], padding_ratio=0.1)
    center_x = 0.5 * (extent.min_x + extent.max_x)
    center_y = 0.5 * (extent.min_y + extent.max_y)
    assert np.isclose(center_x, 6.0)
    assert np.isclose(center_y, 7.5)
    assert np.isclose(extent.max_x - extent.min_x, extent.max_y - extent.min_y)


def test_farthest_point_indices_handles_empty_and_returns_unique_points():
    assert uk_ie.farthest_point_indices(np.empty((0, 2), dtype=np.float64), 4) == []

    points = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )
    indices = uk_ie.farthest_point_indices(points, 3)
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert all(0 <= index < len(points) for index in indices)


def test_build_lighting_override_respects_gpu_light_budget():
    lighthouses = [
        uk_ie.Lighthouse(lon=float(i), lat=float(i), name=f"L{i}", tags={}, proj_x=float(i), proj_y=float(i))
        for i in range(64)
    ]
    for i, lighthouse in enumerate(lighthouses):
        lighthouse.world_x = float(i)
        lighthouse.world_y = float(i * 2)
        lighthouse.world_z = 100.0 + i
        lighthouse.sea_dx = 1.0
        lighthouse.sea_dy = 0.0

    lights = uk_ie.build_lighting_override(lighthouses)
    assert len(lights) == uk_ie.MAX_LIGHTS
    assert lights[0]["type"] == "directional"
    assert all(light["type"] == "spot" for light in lights[1:])
    assert all(light["intensity"] == uk_ie.LIGHTHOUSE_SPOT_INTENSITY for light in lights[1:])
    assert all(light["range"] == uk_ie.LIGHTHOUSE_SPOT_RANGE_M for light in lights[1:])


def test_build_relief_plate_returns_rgba_and_marks_land_only():
    heightmap = np.zeros((32, 32), dtype=np.float32)
    heightmap[8:24, 6:30] = 200.0

    plate = uk_ie.build_relief_plate(heightmap)
    assert plate.shape == (32, 32, 4)
    assert plate.dtype == np.uint8
    assert plate[16, 10, 3] > 0
    assert plate[16, 10, :3].mean() > plate[2, 2, :3].mean()
    assert plate[2, 2, 3] == 0


def test_build_relief_plate_can_hold_land_alpha_over_zero_height_holes():
    heightmap = np.zeros((24, 24), dtype=np.float32)
    heightmap[4:20, 4:20] = 120.0
    heightmap[11:13, 11:13] = 0.0
    land_mask = np.zeros((24, 24), dtype=np.float32)
    land_mask[4:20, 4:20] = 1.0

    plate = uk_ie.build_relief_plate(heightmap, land_mask=land_mask)

    assert plate[12, 12, 3] > 0


def test_build_relief_plate_responds_to_z_scale():
    yy, xx = np.mgrid[-1.0:1.0:32j, -1.0:1.0:32j].astype(np.float32)
    ridge = np.exp(-((xx * 2.4) ** 2 + (yy * 0.8) ** 2)) * 180.0
    knot = np.exp(-(((xx + 0.28) * 3.4) ** 2 + ((yy - 0.18) * 2.2) ** 2)) * 120.0
    heightmap = ridge + knot

    base = uk_ie.build_relief_plate(heightmap, z_scale=10.0)
    exaggerated = uk_ie.build_relief_plate(heightmap, z_scale=40.0)
    land = base[:, :, 3] > 0

    base_std = float(base[:, :, :3][land].std())
    exaggerated_std = float(exaggerated[:, :, :3][land].std())

    assert exaggerated_std > base_std * 1.10


def test_fit_multiline_font_respects_requested_width():
    font = uk_ie.fit_multiline_font(
        "Relief: Mapzen Terrarium\nSource: OpenStreetMap contributors",
        max_width=180,
        preferred_size=18,
        min_size=10,
        family="sans",
        spacing=4,
    )
    width, _ = uk_ie.multiline_text_size(
        "Relief: Mapzen Terrarium\nSource: OpenStreetMap contributors",
        font,
        spacing=4,
    )
    assert width <= 180


def test_mask_heightmap_to_geometry_zeroes_outside_polygon():
    heightmap = np.arange(16, dtype=np.float32).reshape(4, 4)
    extent = uk_ie.SquareExtent(0.0, 0.0, 4.0, 4.0)
    masked = uk_ie.mask_heightmap_to_geometry(heightmap, extent=extent, geometry=box(0.0, 0.0, 2.1, 4.0))
    assert np.array_equal(masked[:, :2], heightmap[:, :2])
    assert np.all(masked[:, 3] == 0.0)


def test_repair_interior_heightmap_voids_fills_land_holes():
    heightmap = np.zeros((7, 7), dtype=np.float32)
    heightmap[1:6, 1:6] = 20.0
    heightmap[3, 3] = 0.0
    land_mask = np.zeros((7, 7), dtype=bool)
    land_mask[1:6, 1:6] = True

    repaired = uk_ie.repair_interior_heightmap_voids(heightmap, land_mask)

    assert repaired[3, 3] > 0.0
    assert repaired[0, 0] == 0.0


def test_apply_northern_relief_boost_emphasizes_northern_mountains():
    heightmap = np.zeros((8, 6), dtype=np.float32)
    heightmap[1:3, 2:4] = 120.0
    heightmap[5:7, 2:4] = 120.0
    land_mask = heightmap > 0.0

    boosted = uk_ie.apply_northern_relief_boost(heightmap, land_mask)

    assert boosted[1:3, 2:4].mean() > heightmap[1:3, 2:4].mean()
    assert boosted[5:7, 2:4].mean() <= heightmap[5:7, 2:4].mean() * 1.001
    assert boosted[0, 0] == 0.0


def test_filter_lighthouses_to_geometry_keeps_only_points_inside_shape():
    lighthouses = [
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="inside", tags={}, proj_x=1.0, proj_y=1.0),
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="outside", tags={}, proj_x=5.0, proj_y=5.0),
    ]
    filtered = uk_ie.filter_lighthouses_to_geometry(lighthouses, box(0.0, 0.0, 2.0, 2.0))
    assert [light.name for light in filtered] == ["inside"]


def test_plan_map_layout_reserves_right_column():
    alpha = np.zeros((256, 256), dtype=np.uint8)
    alpha[18:236, 20:156] = 255
    layout = uk_ie.plan_map_layout(Image.fromarray(alpha, mode="L"), poster_size=512)
    assert layout.crop_box[2] < 256
    assert layout.dest_left + layout.placed_width <= int(512 * 0.81)
    assert layout.placed_height >= int(512 * 0.96)
    assert layout.dest_top <= int(512 * 0.02)


def test_filter_lighthouses_for_poster_rejects_inland_and_non_lighthouse_landmarks():
    geometry = box(0.0, 0.0, 100_000.0, 100_000.0)
    lighthouses = [
        uk_ie.Lighthouse(
            lon=0.0,
            lat=0.0,
            name="Harbor Light",
            tags={"man_made": "lighthouse", "seamark:type": "light_minor"},
            proj_x=1_500.0,
            proj_y=50_000.0,
        ),
        uk_ie.Lighthouse(
            lon=0.0,
            lat=0.0,
            name="Hanbury Wharf Lighthouse",
            tags={"man_made": "lighthouse"},
            proj_x=50_000.0,
            proj_y=50_000.0,
        ),
        uk_ie.Lighthouse(
            lon=0.0,
            lat=0.0,
            name="Huer's Hut",
            tags={"man_made": "lighthouse", "seamark:type": "landmark"},
            proj_x=900.0,
            proj_y=10_000.0,
        ),
        uk_ie.Lighthouse(
            lon=0.0,
            lat=0.0,
            name="Lowestoft Lighthouse",
            tags={"man_made": "lighthouse", "seamark:type": "landmark"},
            proj_x=1_000.0,
            proj_y=20_000.0,
        ),
    ]

    curated = uk_ie.filter_lighthouses_for_poster(lighthouses, geometry)
    assert [light.name for light in curated] == ["Harbor Light", "Lowestoft Lighthouse"]
    assert curated[0].coast_distance_m < uk_ie.LIGHTHOUSE_EXPLICIT_SEAMARK_DISTANCE_M


def test_assign_lighthouse_glow_weights_damps_dense_clusters():
    lighthouses = [
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="cluster-a", tags={}, proj_x=0.0, proj_y=0.0),
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="cluster-b", tags={}, proj_x=1_500.0, proj_y=0.0),
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="cluster-c", tags={}, proj_x=3_000.0, proj_y=1_000.0),
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="sparse-a", tags={}, proj_x=60_000.0, proj_y=0.0),
        uk_ie.Lighthouse(lon=0.0, lat=0.0, name="sparse-b", tags={}, proj_x=130_000.0, proj_y=0.0),
    ]

    uk_ie.assign_lighthouse_glow_weights(lighthouses)
    cluster_weights = [light.glow_weight for light in lighthouses[:3]]
    sparse_weights = [light.glow_weight for light in lighthouses[3:]]

    assert max(cluster_weights) < min(sparse_weights)
    assert all(0.62 <= weight <= 1.14 for weight in cluster_weights + sparse_weights)


def test_crush_landmass_tones_pushes_land_toward_dark_fill():
    terrain = np.zeros((7, 7, 4), dtype=np.uint8)
    terrain[:, :, :3] = 132
    terrain[:, :, 3] = 255
    land_alpha = np.zeros((7, 7), dtype=np.float32)
    land_alpha[2:5, 2:5] = 1.0

    crushed = uk_ie.crush_landmass_tones(terrain, land_alpha)

    assert crushed[3, 3, :3].mean() < crushed[0, 0, :3].mean()
    assert crushed[3, 3, 0] <= 90
    assert np.all(crushed[:, :, 3] == 255)


def test_compress_relief_plate_keeps_relief_felt_not_seen():
    relief = np.zeros((7, 7, 4), dtype=np.uint8)
    relief[1:6, 1:6, :3] = 180
    relief[1:6, 1:6, 3] = 110
    relief[2:5, 2:5, 3] = 240

    compressed = uk_ie.compress_relief_plate(relief)

    assert compressed[3, 3, 3] < relief[3, 3, 3] * 0.65
    assert compressed[1, 3, 3] > 0
    assert compressed[3, 3, :3].mean() < relief[3, 3, :3].mean() * 0.86


def test_render_glow_layers_biases_wash_toward_sea_side():
    lighthouse = uk_ie.Lighthouse(lon=0.0, lat=0.0, name="Beacon", tags={})
    lighthouse.pixel_x = 128.0
    lighthouse.pixel_y = 128.0
    lighthouse.sea_dx = 1.0
    lighthouse.sea_dy = 0.0
    lighthouse.glow_weight = 1.0
    layout = uk_ie.PosterMapLayout(crop_box=(0, 0, 256, 256), dest_left=0, dest_top=0, placed_width=256, placed_height=256)

    halo_far, halo_wash, halo_core, beams, pins, stars = uk_ie.render_glow_layers(256, [lighthouse], 1, layout=layout)

    east = float(halo_wash[120:136, 134:178].mean())
    west = float(halo_wash[120:136, 78:122].mean())
    core_area = int(np.count_nonzero(halo_core > 0.05))
    wash_area = int(np.count_nonzero(halo_wash > 0.01))
    far_area = int(np.count_nonzero(halo_far > 0.01))
    pin_peak = float(pins.max())
    beam_peak = float(beams.max())
    star_peak = float(stars.max())

    assert east > west * 1.25
    assert wash_area > core_area * 4
    assert far_area > wash_area * 1.25
    assert pin_peak > 0.8
    assert beam_peak > 0.08
    assert star_peak > 0.95


def test_render_glow_layers_shrinks_dense_cluster_halos_without_killing_core():
    layout = uk_ie.PosterMapLayout(crop_box=(0, 0, 256, 256), dest_left=0, dest_top=0, placed_width=256, placed_height=256)

    dense = uk_ie.Lighthouse(lon=0.0, lat=0.0, name="Dense", tags={})
    dense.pixel_x = 128.0
    dense.pixel_y = 128.0
    dense.sea_dx = 1.0
    dense.sea_dy = 0.0
    dense.glow_weight = 0.62

    sparse = uk_ie.Lighthouse(lon=0.0, lat=0.0, name="Sparse", tags={})
    sparse.pixel_x = 128.0
    sparse.pixel_y = 128.0
    sparse.sea_dx = 1.0
    sparse.sea_dy = 0.0
    sparse.glow_weight = 1.14

    _, dense_wash, dense_core, _, dense_pins, _ = uk_ie.render_glow_layers(256, [dense], 0, layout=layout)
    _, sparse_wash, sparse_core, _, sparse_pins, _ = uk_ie.render_glow_layers(256, [sparse], 0, layout=layout)

    dense_wash_area = int(np.count_nonzero(dense_wash > 0.01))
    sparse_wash_area = int(np.count_nonzero(sparse_wash > 0.01))
    dense_core_peak = float((dense_core + dense_pins).max())
    sparse_core_peak = float((sparse_core + sparse_pins).max())

    assert dense_wash_area < sparse_wash_area * 0.82
    assert dense_core_peak >= sparse_core_peak * 0.92


def test_apply_poster_finish_preserves_alpha_and_darkens_edges():
    base = np.zeros((96, 96, 4), dtype=np.uint8)
    base[:, :, :3] = 160
    base[:, :, 3] = 203

    finished = uk_ie.apply_poster_finish(Image.fromarray(base, mode="RGBA"))
    arr = np.asarray(finished, dtype=np.uint8)

    center_mean = float(arr[40:56, 40:56, :3].mean())
    corner_mean = float(np.concatenate([arr[:8, :8, :3].ravel(), arr[-8:, -8:, :3].ravel()]).mean())
    center_std = float(arr[24:72, 24:72, 0].std())
    assert center_mean > corner_mean
    assert center_std > 4.0
    assert np.all(arr[:, :, 3] == 203)
