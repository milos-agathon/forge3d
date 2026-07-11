from pathlib import Path
import ast


REPO = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = REPO / "examples" / "population_spike_worldpop" / "france_population_spikes_height_shade.py"


def test_france_roma_population_spike_example_contract() -> None:
    assert EXAMPLE_PATH.exists()

    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(EXAMPLE_PATH))
    constants = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "fra_pd_2020_1km_UNadj.tif" in constants
    assert "france_population_spikes_roma_4k.png" in constants
    assert "France  \u00b7  2020  \u00b7  1 km grid" in constants
    assert "Population density" in constants
    assert "roma" in source.lower()


def test_france_roma_lut_maps_low_cold_high_warm() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    lut = module.build_roma_lut_srgb()

    low = lut[0].astype(int)
    high = lut[-1].astype(int)
    assert low[2] > low[0], "low Roma values should read as cold blue"
    assert high[0] > high[2], "high Roma values should read as warm red"


def test_missing_france_raster_exits_with_download_guidance(tmp_path, capsys) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.DATA_PATH = tmp_path / "fra_pd_2020_1km_UNadj.tif"

    try:
        module.ensure_france_raster_available()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected missing raster preflight to exit")

    output = capsys.readouterr().out
    assert "Missing France population raster" in output
    assert "https://data.worldpop.org/" in output
    assert str(module.DATA_PATH) in output


def test_cleanup_preserves_roma_hue_order() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    lut = module.build_roma_lut_srgb()
    raw = np.zeros((3, 3, 3), dtype=np.uint8)
    raw[:, :] = lut[128]
    raw[1, 0] = lut[0]
    raw[1, 1] = lut[128]
    raw[1, 2] = lut[255]

    cleaned = module.cleanup_snapshot(raw)

    low = cleaned[1, 0].astype(int)
    mid = cleaned[1, 1].astype(int)
    high = cleaned[1, 2].astype(int)
    assert low[2] > low[0], "low Roma pixels should stay cold"
    assert high[0] > high[2], "high Roma pixels should stay warm"
    assert mid[1] >= mid[0] and mid[1] >= mid[2], "Roma midpoint should retain its green-yellow center"


def test_cleanup_maps_rendered_lilac_floor_to_dark_roma_blue() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    raw = np.full((3, 3, 3), [136, 152, 222], dtype=np.uint8)
    cleaned = module.cleanup_snapshot(raw)
    low = cleaned[1, 1].astype(int)

    assert low[2] > low[0] * 3
    assert low[0] < 20
    assert low[1] < 80


def test_cleanup_normalizes_viewer_background_for_plate_replacement() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    raw = np.full((2, 2, 3), [22, 22, 29], dtype=np.uint8)
    cleaned = module.cleanup_snapshot(raw)

    assert np.array_equal(cleaned[0, 0], np.array(module.RENDER_BG_RGB, dtype=np.uint8))


def test_france_camera_overrides_shared_poland_fit() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.configure_base_module()

    assert module.height_shade.CAMERA_RADIUS == module.FRANCE_CAMERA_RADIUS
    assert module.height_shade.CAMERA_FOV == module.FRANCE_CAMERA_FOV
    assert module.FRANCE_CAMERA_RADIUS > 2400.0


def test_france_city_labels_are_removed() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert not hasattr(module, "FRANCE_CITY_LABELS")
    assert not hasattr(module, "draw_city_labels")
    assert not hasattr(module, "project_city_to_plate")


def test_plate_fit_preserves_render_aspect_ratio() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rect = module.fit_map_rect((2654, 1688))
    _, _, width, height = rect

    assert abs((width / height) - (2654 / 1688)) < 0.01


def test_plate_fit_uses_wide_map_area() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    x, _, width, _ = module.fit_map_rect((2654, 1688))

    assert x <= 140
    assert width >= 4500


def test_france_uses_5x_raster_supersample() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    valid = np.array([[True, False, True], [True, True, False]])
    up_data, up_valid = module.supersample_population_grid(data, valid)

    assert module.RASTER_SUPERSAMPLE == 5
    assert up_data.shape == (10, 15)
    assert up_valid.shape == (10, 15)


def test_france_height_dem_uses_refined_spike_scale() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    data = np.array([[0.0, 100.0], [500.0, 1000.0]], dtype=np.float32)
    valid = np.ones_like(data, dtype=bool)

    base_height = module.height_shade.build_height_dem(data, 1000.0, valid)
    france_height = module.build_france_height_dem(data, 1000.0, valid)

    assert module.SPIKE_HEIGHT_MULTIPLIER == 5.75
    assert np.allclose(france_height, base_height * 5.75)


def test_france_layout_has_corsica_inset_area() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.INSET_AREA_X > module.MAP_AREA_X
    assert module.INSET_AREA_Y > module.MAP_AREA_Y
    assert module.INSET_AREA_W >= 480
    assert hasattr(module, "split_mainland_and_corsica")


def test_legend_is_moved_south() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.LEGEND_Y >= 69
    assert module.LEGEND_Y < 180


def test_france_plate_is_shorter_and_map_area_is_tighter() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.SNAP_SIZE == 6144
    assert module.FINAL_PLATE_W == 4800
    assert module.FINAL_PLATE_H == 3200
    assert module.MAP_AREA_X <= 70
    assert module.MAP_AREA_W >= 4600


def test_legend_labels_top_tick_as_capped() -> None:
    import importlib.util
    import sys

    import numpy as np

    spec = importlib.util.spec_from_file_location("france_population_spikes_height_shade", EXAMPLE_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    legend = module.make_roma_legend(bar_w=16, bar_h=80, clip_max=1763.0)
    arr = np.array(legend.convert("RGBA"))

    assert arr.shape[0] > 120
    assert arr.shape[1] > 80
