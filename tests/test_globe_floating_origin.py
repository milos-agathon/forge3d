"""ORBIS native GlobeScene API and truthful pre-probe behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

import forge3d as f3d


RAINIER = Path(__file__).parents[1] / "assets" / "tif" / "dem_rainier.tif"
RAINIER_LON = -121.7603
RAINIER_LAT = 46.8523


def test_globe_scene_and_metrics_are_public_native_types():
    assert f3d.GlobeScene.__module__ == "forge3d._forge3d"
    assert f3d.GlobeMetrics.__module__ == "forge3d._forge3d"
    assert "GlobeScene" in f3d.__all__
    assert "GlobeMetrics" in f3d.__all__


@pytest.mark.parametrize(
    ("lon", "lat", "name", "message"),
    [
        (float("nan"), RAINIER_LAT, "Rainier", "finite"),
        (RAINIER_LON, 91.0, "Rainier", "latitude"),
        (181.0, RAINIER_LAT, "Rainier", "longitude"),
        (RAINIER_LON, RAINIER_LAT, "  ", "target_name"),
    ],
)
def test_constructor_rejects_invalid_target_before_gpu(lon, lat, name, message):
    with pytest.raises(ValueError, match=message):
        f3d.GlobeScene(str(RAINIER), lon, lat, name)


def test_source_error_names_source_and_target(tmp_path):
    missing = tmp_path / "missing-rainier.tif"
    with pytest.raises(RuntimeError) as caught:
        f3d.GlobeScene(str(missing), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    text = str(caught.value)
    assert str(missing) in text
    assert "Mount Rainier" in text
    assert str(RAINIER_LON) in text
    assert str(RAINIER_LAT) in text


def test_target_outside_source_coverage_is_diagnostic():
    with pytest.raises(ValueError) as caught:
        f3d.GlobeScene(str(RAINIER), 0.0, 0.0, "Not Rainier")
    text = str(caught.value)
    assert "Not Rainier" in text
    assert "coverage" in text.lower()
    assert str(RAINIER) in text


def test_snapshot_and_metrics_fail_closed_before_descent():
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    with pytest.raises(RuntimeError, match="snapshot.*render"):
        scene.snapshot()
    with pytest.raises(RuntimeError, match="metrics.*complete"):
        scene.metrics()


def test_constructor_accepts_declared_pathlike_source():
    scene = f3d.GlobeScene(RAINIER, RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    assert scene.source == str(RAINIER)


def test_source_covered_waypoint_outside_seeded_overview_rejects_without_mutation():
    scene = f3d.GlobeScene(RAINIER, RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    # This point is inside the Rainier COG bounds but west of the fully covered
    # overview tile selected around the constructor target.
    with pytest.raises(ValueError, match="seeded overview tile") as caught:
        scene.fly_to(-121.88, 46.85, 1_000.0)
    message = str(caught.value)
    assert "overview validation" in message
    assert "Mount Rainier" in message
    assert "-121.88" in message and "46.85" in message and "1000" in message
    with pytest.raises(ValueError, match="seeded overview tile"):
        scene.scripted_descent(
            [
                (RAINIER_LON, RAINIER_LAT, 10_000.0),
                (-121.88, 46.85, 1_000.0),
            ]
        )
    assert scene.current_position is None
    assert scene.rendered_waypoint_count == 0
    with pytest.raises(RuntimeError, match="snapshot.*render"):
        scene.snapshot()


@pytest.mark.parametrize(
    "waypoints",
    [[], [(RAINIER_LON, RAINIER_LAT, float("nan"))], [(181.0, 0.0, 1.0)]],
)
def test_custom_waypoints_are_validated_before_render(waypoints):
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    with pytest.raises(ValueError):
        scene.scripted_descent(waypoints)


def test_entire_custom_path_is_validated_before_any_render():
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    with pytest.raises(ValueError, match="coverage"):
        scene.scripted_descent(
            [
                (RAINIER_LON, RAINIER_LAT, 10_000.0),
                (0.0, 0.0, 1_000.0),
            ]
        )
    with pytest.raises(RuntimeError, match="snapshot.*render"):
        scene.snapshot()


def test_extreme_altitude_rejects_without_mutating_scene():
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    with pytest.raises(ValueError, match="altitude.*408000") as caught:
        scene.fly_to(RAINIER_LON, RAINIER_LAT, 1.0e300)
    message = str(caught.value)
    assert RAINIER.name in message
    assert "Mount Rainier" in message
    assert "validation" in message
    assert str(RAINIER_LON) in message
    assert str(RAINIER_LAT) in message
    assert "waypoint (" in message and " m)" in message
    assert scene.current_position is None
    assert scene.rendered_waypoint_count == 0
    with pytest.raises(RuntimeError, match="snapshot.*render"):
        scene.snapshot()


def test_default_descent_contract_is_logarithmic_and_exact():
    altitudes = f3d.GlobeScene.default_descent_altitudes()
    assert altitudes[0] == 408_000.0
    assert altitudes[-1] == 0.0
    assert len(altitudes) >= 16
    assert all(a > b for a, b in zip(altitudes, altitudes[1:]))
    ratios = [
        (altitudes[i] + 1.0) / (altitudes[i + 1] + 1.0)
        for i in range(len(altitudes) - 1)
    ]
    assert max(ratios) - min(ratios) < 1e-9


def test_native_source_drives_one_bounded_stream_step_and_real_renderer():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "terrain"
        / "clipmap"
        / "globe_scene.rs"
    ).read_text(encoding="utf-8")
    assert "PyCogDataset::new" in source
    assert source.count("stream_height_tiles_globe(") == 1
    assert "render_terrain_pbr_pom(" in source
    assert "self.last_frame = Some(frame.clone_ref(py))" in source
    assert "Maintain::Wait" not in source


@pytest.mark.offscreen
def test_fly_to_and_custom_descent_render_real_frames():
    if not f3d.has_gpu():
        pytest.skip("No GPU adapter is available")
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    first = scene.fly_to(RAINIER_LON, RAINIER_LAT, 10_000.0)
    assert first.size == (256, 192)
    assert first.to_numpy().shape == (192, 256, 4)
    scene.scripted_descent([(RAINIER_LON, RAINIER_LAT, 1_000.0)])
    assert scene.rendered_waypoint_count == 2
    assert scene.current_position == (RAINIER_LON, RAINIER_LAT, 1_000.0)
    assert scene.snapshot().to_numpy().shape == (192, 256, 4)
    with pytest.raises(RuntimeError, match="unmeasured.*physical GPU probes"):
        scene.metrics()
