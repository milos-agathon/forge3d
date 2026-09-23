"""ORBIS native GlobeScene API and truthful pre-probe behavior."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from tests._ssim import ssim


RAINIER = Path(__file__).parents[1] / "assets" / "tif" / "dem_rainier.tif"
RAINIER_LON = -121.7603
RAINIER_LAT = 46.8523
ORBIS_GOLDEN = (
    Path(__file__).parent
    / "golden"
    / "terrain"
    / "orbis_rainier_ground.nvidia-vulkan.png"
)
ORBIS_SELECTED = os.environ.get("FORGE3D_RUN_ORBIS_GPU") == "1"
# The physical baseline has 36 RGB colors and a 93.32% non-modal fraction since
# globe shading builds its height normals in each fragment's east/north/up frame
# at metric scale (the flat Y-up normal was ~1000x over-steep and left the ground
# nearly unlit); these lower bounds retain broad rendering tolerance while
# rejecting blank frames.
ORBIS_MIN_DISTINCT_GROUND_COLORS = 16
ORBIS_MIN_NON_MODAL_GROUND_FRACTION = 0.05


def _require_orbis_acceptance() -> None:
    if not ORBIS_SELECTED:
        pytest.skip("set FORGE3D_RUN_ORBIS_GPU=1 for physical ORBIS acceptance")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_orbis_ground_content(image: np.ndarray, label: str) -> None:
    rgb = np.asarray(image[..., :3], dtype=np.uint8).reshape(-1, 3)
    colors, counts = np.unique(rgb, axis=0, return_counts=True)
    non_modal_fraction = 1.0 - float(counts.max()) / float(rgb.shape[0])
    assert colors.shape[0] >= ORBIS_MIN_DISTINCT_GROUND_COLORS, (
        f"{label} has no credible ground content: {colors.shape[0]} distinct RGB "
        f"colors, expected at least {ORBIS_MIN_DISTINCT_GROUND_COLORS}"
    )
    assert non_modal_fraction >= ORBIS_MIN_NON_MODAL_GROUND_FRACTION, (
        f"{label} has no credible ground content: {non_modal_fraction:.6f} non-modal "
        f"pixel fraction, expected at least {ORBIS_MIN_NON_MODAL_GROUND_FRACTION:.6f}"
    )


def test_orbis_ground_content_gate_rejects_blank_and_accepts_committed_golden():
    blank = np.full((192, 256, 4), [25, 25, 38, 255], dtype=np.uint8)
    with pytest.raises(AssertionError, match="ground content"):
        _assert_orbis_ground_content(blank, "blank ORBIS frame")

    expected = np.asarray(Image.open(ORBIS_GOLDEN).convert("RGBA"))
    _assert_orbis_ground_content(expected, "committed ORBIS golden")


@pytest.fixture(scope="module")
def orbis_descent_evidence():
    _require_orbis_acceptance()
    assert RAINIER.stat().st_size > 1024, "Rainier COG is an unrestored LFS pointer"
    assert ORBIS_GOLDEN.is_file(), f"missing committed ORBIS golden: {ORBIS_GOLDEN}"

    scene = f3d.GlobeScene(RAINIER, RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    metrics = scene.scripted_descent()
    assert isinstance(metrics, f3d.GlobeMetrics)
    assert scene.metrics().as_dict() == metrics.as_dict()
    actual = scene.snapshot().to_numpy()
    expected = np.asarray(Image.open(ORBIS_GOLDEN).convert("RGBA"))
    assert actual.shape == expected.shape
    _assert_orbis_ground_content(expected, "committed ORBIS golden")
    _assert_orbis_ground_content(actual, "actual ORBIS ground snapshot")

    score = float(ssim(actual[..., :3], expected[..., :3], data_range=255.0))
    mad = float(np.mean(np.abs(actual[..., :3].astype(np.float64) - expected[..., :3])))
    png_buffer = BytesIO()
    Image.fromarray(actual, mode="RGBA").save(png_buffer, format="PNG")
    actual_png = png_buffer.getvalue()
    evidence = {
        "target": {
            "name": "Mount Rainier",
            "longitude": RAINIER_LON,
            "latitude": RAINIER_LAT,
        },
        "metrics": metrics.as_dict(),
        "golden": {
            "path": ORBIS_GOLDEN.relative_to(Path(__file__).parents[1]).as_posix(),
            "sha256": _sha256(ORBIS_GOLDEN),
            "ssim": score,
            "mean_absolute_difference": mad,
        },
        "snapshot_sha256": hashlib.sha256(actual_png).hexdigest(),
    }
    artifact_dir = os.environ.get("FORGE3D_ORBIS_ARTIFACT_DIR")
    if artifact_dir:
        output = Path(artifact_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "orbis-rainier-ground.actual.png").write_bytes(actual_png)
        (output / "orbis-metrics.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return evidence


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


def test_failed_later_overview_seed_leaves_scene_state_unchanged():
    from forge3d.cog import CogDataset

    scene = f3d.GlobeScene(RAINIER, RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    dataset = CogDataset(RAINIER.resolve().as_uri())
    west, south, _east, north = dataset.bounds
    edge_lon = west + 1.0e-8
    edge_lat = (south + north) * 0.5
    with pytest.raises(ValueError, match="overview seed preparation") as caught:
        scene.fly_to(edge_lon, edge_lat, 1_000.0)
    message = str(caught.value)
    assert "Mount Rainier" in message
    assert "refusing to fabricate" in message
    with pytest.raises(ValueError, match="overview seed preparation"):
        scene.scripted_descent(
            [
                (RAINIER_LON, RAINIER_LAT, 10_000.0),
                (edge_lon, edge_lat, 1_000.0),
            ]
        )
    assert scene.current_position is None
    assert scene.rendered_waypoint_count == 0
    with pytest.raises(RuntimeError, match="snapshot.*render"):
        scene.snapshot()
    with pytest.raises(RuntimeError, match="metrics.*complete"):
        scene.metrics()


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
    with pytest.raises(ValueError, match="altitude.*9000000") as caught:
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


@pytest.mark.parametrize(
    ("texture", "error", "message"),
    [
        (np.zeros((4, 8), np.uint8), TypeError, "uint8 numpy array"),
        (np.zeros((4, 8, 3), np.float32), TypeError, "uint8 numpy array"),
        (np.zeros((4, 6, 3), np.uint8), ValueError, "width == 2 \* height"),
        (np.zeros((4, 8, 2), np.uint8), ValueError, "channels"),
    ],
)
def test_earth_texture_is_validated_before_gpu(texture, error, message):
    with pytest.raises(error, match=message):
        f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier", earth_texture=texture)


def test_oblique_camera_distance_is_bounded_before_render():
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    with pytest.raises(ValueError, match="from its target.*maximum"):
        scene.fly_to(RAINIER_LON, RAINIER_LAT, 8_000_000.0, 0.0, 60.0)
    assert scene.current_position is None
    assert scene.rendered_waypoint_count == 0


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
    # One bounded step per waypoint, one zero-upload reanchor to the elevated
    # physical-probe base, and one zero-upload step for each of the two metric
    # frames; no convergence loop exists in any call site.
    assert source.count("stream_height_tiles_globe(") == 2
    assert source.count("stream_height_tiles_globe_focus(") == 1
    assert "render_terrain_pbr_pom(" in source
    assert "self.last_frame = Some(frame.clone_ref(py))" in source
    assert "Maintain::Wait" not in source
    assert source.index("let prepared_path") < source.index("begin_orbis_descent")
    frame_loop = source.split("let frame_result = (||", 1)[1].split(
        "finish_owner_ledger_capture", 1
    )[0]
    assert "load_covered_overview(" not in frame_loop
    assert "read_covered_seed(" not in frame_loop

    streaming = (
        Path(__file__).parents[1] / "src" / "terrain" / "renderer" / "streaming.rs"
    ).read_text(encoding="utf-8")
    activation = streaming.split("fn submit_overview", 1)[1].split(
        "/// Slices height tiles", 1
    )[0]
    assert "tracked_create_buffer(" in activation
    assert "tracked_create_texture(" in activation
    assert "copy_buffer_to_texture(" in activation
    assert "copy_buffer_to_buffer(" in activation
    assert ".write_buffer(" not in activation
    assert ".write_texture(" not in activation
    assert ".create_buffer(" not in activation
    assert ".create_texture(" not in activation

    transaction = source.split("fn render_waypoint_with_overview", 1)[1].split(
        "fn active_overview", 1
    )[0]
    assert transaction.index("begin_height_streaming_overview_activation") < transaction.index(
        "match self.render_waypoint(py, waypoint)"
    )
    assert "commit_height_streaming_overview_activation" in transaction
    assert "rollback_height_streaming_overview_activation" in transaction
    assert "self.active_overview_tile = previous_tile" in transaction

    assert "overview_double_residency_bytes" in streaming
    assert "reserve_overview_rollback" in streaming
    begin = streaming.split("fn begin_overview_activation", 1)[1].split(
        "fn commit_overview_activation", 1
    )[0]
    assert begin.index("prepare_overview_rollback") < begin.index("serialize_with_overview")
    assert begin.index("prepare_overview_rollback") < begin.index("submit_overview")

    rollback = streaming.split("fn rollback_overview_activation", 1)[1].split(
        "/// Map a tile", 1
    )[0]
    assert "submit_prepared_overview_rollback" in rollback
    assert "serialize_with_overview" not in rollback
    assert "tracked_create_buffer" not in rollback
    assert "?" not in rollback


def test_orbis_review_corrections_are_bound_to_production_paths():
    root = Path(__file__).parents[1]
    capture = (root / "src/terrain/renderer/orbis_capture.rs").read_text()
    scene = (root / "src/terrain/clipmap/globe_scene.rs").read_text()
    geometry = (root / "src/terrain/renderer/geometry.rs").read_text()
    shader = (root / "src/shaders/terrain_pbr_pom.wgsl").read_text(encoding="utf-8")
    draw = (root / "src/terrain/renderer/draw/mod.rs").read_text()
    execute = (root / "src/terrain/renderer/draw/execute.rs").read_text()
    visibility = (root / "src/terrain/renderer/visibility_buffer.rs").read_text()
    streaming = (root / "src/terrain/renderer/streaming.rs").read_text()

    # Two distinct successfully submitted frames and exact selection origins.
    assert "frames.len() == 2" in capture
    assert "submission_serial > self.frames[0].submission_serial" in capture
    assert "first.selection_provenance != second.selection_provenance" in capture
    assert "mark_orbis_capture_submitted" in draw

    # The deliberate naive control rotates ENU into ECEF before f32 narrowing.
    assert "GlobeFrame::tangent_to_ecef" in capture
    assert "orbis_probe.local_to_ecef" in shader
    assert "orbis_probe.anchor_abs" in shader

    # Crack boundaries and variants are tied to the exact indirect selection.
    for resource in ("output_header", "output_tiles", "indirect_buffer", "instance_buffer"):
        assert f"lod_resources.{resource}" in capture
    assert "selected_mating_boundaries(" in capture
    assert "analyze_paired_cracks(" in capture
    assert "level.rebase_globe_vertices(&mut vertices)" in geometry
    assert "encode_indirect_globe_tracked" in geometry

    # Altitude/reanchor identity is part of geometry cache identity.
    assert "globe_center_anchor_bits" in geometry
    assert "height_streaming_globe_identity" in geometry

    # No visibility readback wait is reachable during an ORBIS descent.
    assert "if self.orbis_descent_active" in visibility
    assert execute.count("runtime_visibility_stats_enabled()") >= 3
    assert "&& self.runtime_visibility_stats_enabled()" in draw

    # Any capture aborted after later render failure closes every GPU scope.
    assert "pending.drain_validation_scopes" in capture
    assert "while self.validation_scopes > 0" in capture

    # Bounded polls and genuine streaming state changes are distinct and are
    # both counted only after a successful render.
    assert "bounded_steps" in streaming
    assert ".observe(snapshot)" in scene
    assert "bounded_poll_frames" in scene or "bounded_poll_frames" in (root / "src/terrain/clipmap/globe_scene/metrics.rs").read_text()
    assert scene.index("self.record_streaming_evidence(py, &stream_stats)?") > scene.index(
        '"render",\n            render_result'
    )

    probe = scene.split("fn run_orbis_physical_probe", 1)[1].split(
        "fn waypoint_context_message", 1
    )[0]
    assert probe.index("safe_orbis_probe_base") < probe.index("orbis_reanchor_state")
    assert probe.index("stream_height_tiles_globe") < probe.index("orbis_reanchor_state")
    assert "record_streaming_evidence" not in probe
    assert "rendered_waypoints +=" not in probe


@pytest.mark.offscreen
def test_fly_to_and_custom_descent_render_real_frames():
    _require_orbis_acceptance()
    scene = f3d.GlobeScene(str(RAINIER), RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    first = scene.fly_to(RAINIER_LON, RAINIER_LAT, 10_000.0)
    assert first.size == (256, 192)
    assert first.to_numpy().shape == (192, 256, 4)
    assert scene.rendered_waypoint_count == 1
    assert scene.current_position == (RAINIER_LON, RAINIER_LAT, 10_000.0)


@pytest.mark.offscreen
def test_adjacent_overview_tiles_and_cached_return_keep_ground_visible():
    _require_orbis_acceptance()
    scene = f3d.GlobeScene(RAINIER, RAINIER_LON, RAINIER_LAT, "Mount Rainier")
    first = scene.fly_to(RAINIER_LON, RAINIER_LAT, 1_000.0).to_numpy()
    adjacent_lon, adjacent_lat = -121.88, 46.85
    second = scene.fly_to(adjacent_lon, adjacent_lat, 1_000.0).to_numpy()
    returned = scene.fly_to(RAINIER_LON, RAINIER_LAT, 1_000.0).to_numpy()
    _assert_orbis_ground_content(first, "initial Rainier waypoint")
    _assert_orbis_ground_content(second, "adjacent Rainier waypoint")
    _assert_orbis_ground_content(returned, "cached Rainier return waypoint")
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, returned)
    assert scene.rendered_waypoint_count == 3


@pytest.mark.offscreen
def test_orbis_descent_meets_physical_metric_gates(orbis_descent_evidence):
    metrics = orbis_descent_evidence["metrics"]
    assert 0.0 <= metrics["max_vertex_jitter_px"] < 0.5
    assert metrics["jitter_sample_count"] >= 32
    assert (
        metrics["naive_max_vertex_jitter_px"]
        > 10.0 * metrics["max_vertex_jitter_px"]
    )
    assert 0 < metrics["peak_gpu_visible_bytes"] < 512 * 1024 * 1024
    assert metrics["lod_crack_pixels"] == 0
    assert metrics["crack_boundary_samples"] >= 64
    assert metrics["crack_depth_variance"] > 0.0


@pytest.mark.offscreen
def test_orbis_descent_streaming_progresses_without_blocking(orbis_descent_evidence):
    metrics = orbis_descent_evidence["metrics"]
    expected_frames = len(f3d.GlobeScene.default_descent_altitudes())
    assert metrics["rendered_frames"] == expected_frames
    assert metrics["bounded_poll_frames"] == expected_frames
    assert 1 <= metrics["streaming_progress_frames"] <= expected_frames
    assert metrics["pending_streaming_frames"] > 0
    assert metrics["coarse_fallback_frames"] > 0
    assert 0 < metrics["max_stream_uploads_per_frame"] <= 64


@pytest.mark.offscreen
def test_orbis_descent_records_exact_nvidia_vulkan_adapter(orbis_descent_evidence):
    metrics = orbis_descent_evidence["metrics"]
    assert metrics["adapter_vendor"] == 0x10DE
    assert "nvidia" in metrics["adapter_name"].casefold()
    assert metrics["adapter_backend"] == "Vulkan"
    assert metrics["adapter_device_type"] == "DiscreteGpu"
    assert metrics["software_fallback"] is False

    probe_path = os.environ.get("FORGE3D_EXPECTED_ADAPTER_PROBE")
    assert probe_path, "selected ORBIS lane must provide its adapter probe"
    probe = json.loads(Path(probe_path).read_text(encoding="utf-8"))["probe"]
    assert metrics["adapter_name"] == probe["name"]
    assert metrics["adapter_vendor"] == probe["vendor"]
    assert metrics["adapter_backend"] == probe["backend"]
    assert metrics["adapter_device_type"] == probe["device_type"]
    assert metrics["software_fallback"] == probe["software_fallback"]


@pytest.mark.offscreen
def test_orbis_ground_snapshot_matches_committed_nvidia_vulkan_golden(
    orbis_descent_evidence,
):
    golden = orbis_descent_evidence["golden"]
    assert golden["ssim"] >= 0.995
    assert golden["mean_absolute_difference"] <= 2.0


def test_orbis_evidence_writer_records_runtime_sha_and_measured_payload(tmp_path):
    from scripts.write_orbis_evidence import EvidenceError, _validate_measurements

    golden_path = tmp_path / "golden.png"
    actual_path = tmp_path / "actual.png"
    golden_path.write_bytes(b"committed golden bytes")
    actual_path.write_bytes(b"measured actual bytes")
    metrics = {
        "target": {"name": "Mount Rainier"},
        "metrics": {
            "max_vertex_jitter_px": 0.1,
            "naive_max_vertex_jitter_px": 2.0,
            "jitter_sample_count": 64,
            "peak_gpu_visible_bytes": 123456,
            "lod_crack_pixels": 0,
            "crack_boundary_samples": 128,
            "crack_depth_variance": 0.25,
            "rendered_frames": 25,
            "bounded_poll_frames": 25,
            "streaming_progress_frames": 10,
            "pending_streaming_frames": 3,
            "coarse_fallback_frames": 2,
            "max_stream_uploads_per_frame": 8,
            "adapter_name": "NVIDIA Test Adapter",
            "adapter_backend": "Vulkan",
            "adapter_vendor": 0x10DE,
            "adapter_device_type": "DiscreteGpu",
            "software_fallback": False,
        },
        "golden": {
            "path": "tests/golden/terrain/orbis_rainier_ground.nvidia-vulkan.png",
            "sha256": _sha256(golden_path),
            "ssim": 0.999,
            "mean_absolute_difference": 0.5,
        },
        "snapshot_sha256": _sha256(actual_path),
    }
    probe = {
        "requested_backend": "vulkan",
        "probe": {
            "status": "ok",
            "name": "NVIDIA Test Adapter",
            "vendor": 0x10DE,
            "backend": "Vulkan",
            "device_type": "DiscreteGpu",
            "software_fallback": False,
        },
    }
    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["streaming_progress_frames"] = 0
    with pytest.raises(EvidenceError, match="positive integer"):
        _validate_measurements(rejected)

    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["jitter_sample_count"] = 31
    with pytest.raises(EvidenceError, match="at least 32"):
        _validate_measurements(rejected)

    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["crack_boundary_samples"] = 63
    with pytest.raises(EvidenceError, match="at least 64"):
        _validate_measurements(rejected)

    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["rendered_frames"] = 24
    rejected["metrics"]["bounded_poll_frames"] = 24
    with pytest.raises(EvidenceError, match="exactly 25"):
        _validate_measurements(rejected)

    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["bounded_poll_frames"] = 24
    with pytest.raises(EvidenceError, match="exactly 25"):
        _validate_measurements(rejected)

    rejected = json.loads(json.dumps(metrics))
    rejected["metrics"]["streaming_progress_frames"] = 26
    with pytest.raises(EvidenceError, match="between 1 and 25"):
        _validate_measurements(rejected)
    assert _validate_measurements(metrics)["streaming_progress_frames"] == 10
    metrics_path = tmp_path / "metrics.json"
    probe_path = tmp_path / "probe.json"
    output_path = tmp_path / "evidence.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    probe_path.write_text(json.dumps(probe), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/write_orbis_evidence.py",
            "--repository",
            ".",
            "--metrics",
            str(metrics_path),
            "--adapter-probe",
            str(probe_path),
            "--golden",
            str(golden_path),
            "--actual-image",
            str(actual_path),
            "--command",
            "python -m pytest tests/test_globe_floating_origin.py",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).parents[1],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(output_path.read_text(encoding="utf-8"))
    expected_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True
    ).strip()
    assert evidence["repository_sha"] == expected_sha
    assert evidence["commands"] == [
        "python -m pytest tests/test_globe_floating_origin.py"
    ]
    assert evidence["adapter"] == probe["probe"]
    assert evidence["measurements"] == metrics


def test_orbis_ci_contract_is_lfs_restored_zero_skip_and_evidence_bound():
    workflow = (Path(__file__).parents[1] / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    job = workflow.split("  test-golden-images-nvidia:", 1)[1].split(
        "\n  # ============================================================================", 1
    )[0]
    assert "needs: [build-wheel-windows, terrain-golden-paths]" in job
    assert "prepare-lfs-fixtures" not in job
    assert "lfs-fixture-bundles" not in job and "python-tiffs.zip" not in job
    assert (
        "Path = 'assets/tif/dem_rainier.tif'; "
        "Sha256 = '875b243474b151175f76037acd60c2149ac2e46fba9ba2bbce0c9a6998015dd3'"
        in job
    )
    assert "FORGE3D_RUN_ORBIS_GPU: '1'" in job
    assert "run_nvidia_visual_acceptance.py --suite orbis" in job
    assert "assert_junit_zero_skips.py" in job
    assert "write_orbis_evidence.py" in job
    assert "FORGE3D_ORBIS_ARTIFACT_DIR" in job
    assert (
        '--command "python scripts/terrain_ci_probe.py --mode terrain '
        '--require-nvidia-vulkan --json `"$env:FORGE3D_EXPECTED_ADAPTER_PROBE`""'
        in job
    )
    assert (
        '--command "python scripts/run_nvidia_visual_acceptance.py --suite orbis '
        '--junit `"$junit`""'
        in job
    )
