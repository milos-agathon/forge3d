import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent
_LOCAL_EXAMPLE_TESTS = {
    "tests/test_california_cigar_smoke_hybrid.py": "examples/california_cigar_smoke_demo.py",
    "tests/test_california_fire_smoke_effect.py": "examples/california_fire_smoke_effect.py",
    "tests/test_california_wildfire_smoke_wind.py": "examples/california_wildfire_smoke_video.py",
    "tests/test_bosnia_terrain_landcover_viewer.py": "examples/bosnia_terrain_landcover_viewer.py",
    "tests/test_bryce_canyon_storm_timelapse.py": "examples/bryce_canyon_storm_timelapse.py",
    "tests/test_bundle_cli.py": "examples/terrain_demo.py",
    "tests/test_helsinki_transit_daycycle.py": "examples/helsinki_transit_daycycle.py",
    "tests/test_humanity_globe_video.py": "examples/humanity_globe_video.py",
    "tests/test_khumbu_icefall_sentinel_timelapse.py": "examples/khumbu_icefall_sentinel_timelapse.py",
    "tests/test_label_api_docs_support.py": "examples/label_api_truth_basic.py",
    "tests/test_label_api_quickstart.py": "examples/label_api_truth_basic.py",
    "tests/test_lighting_alignment.py": "examples/terrain_demo.py",
    "tests/test_mapscene_docs.py": "examples/mapscene_terrain_raster.py",
    "tests/test_mapscene_examples.py": "examples/mapscene_terrain_raster.py",
    "tests/test_mapscene_quickstart.py": "examples/mapscene_vector_labels.py",
    "tests/test_motion_vectors.py": "examples/camera_animation_demo.py",
    "tests/test_oit_transparency.py": "examples/terrain_demo.py",
    "tests/test_osm_city_demo.py": "examples/osm_city_demo.py",
    "tests/test_perspective_projection.py": "examples/terrain_demo.py",
    "tests/test_rotterdam_solar_potential_shadow_study.py": "examples/rotterdam_solar_potential_shadow_study.py",
    "tests/test_shadow_techniques.py": "examples/terrain_demo.py",
    "tests/test_sun_ephemeris.py": "examples/terrain_demo.py",
    "tests/test_taa_toggle.py": "examples/camera_animation_demo.py",
    "tests/test_terrain_demo.py": "examples/terrain_demo.py",
    "tests/test_terrain_demo_cli_smoke.py": "examples/terrain_demo.py",
    "tests/test_terrain_demo_preset_integration.py": "examples/terrain_demo.py",
    "tests/test_turkiye_river_basins_3d.py": "examples/turkiye_river_basins_3d.py",
    "tests/test_uk_ireland_lighthouse_map.py": "examples/uk_ireland_lighthouse_map.py",
}


def pytest_ignore_collect(collection_path, config):
    del config
    try:
        rel = Path(collection_path).resolve().relative_to(_ROOT).as_posix()
    except ValueError:
        return False
    example = _LOCAL_EXAMPLE_TESTS.get(rel)
    return bool(example and not (_ROOT / example).exists())


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asyncio (requires pytest-asyncio)")


def pytest_collection_modifyitems(items):
    has_asyncio = importlib.util.find_spec("pytest_asyncio") is not None

    if has_asyncio:
        return

    skip_asyncio = pytest.mark.skip(reason="pytest-asyncio not installed; skipping asyncio tests")
    for item in items:
        if 'asyncio' in item.keywords:
            item.add_marker(skip_asyncio)
