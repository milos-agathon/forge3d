from __future__ import annotations

import math

import numpy as np
import pytest

from forge3d.animation import CameraKeyframe
from forge3d.camera_rigs import (
    TerrainClearance,
    TerrainOrbitRig,
    TerrainRailRig,
    TerrainTargetFollowRig,
)
from forge3d.terrain_scatter import TerrainScatterSource


def _eye_from_state(state) -> tuple[float, float, float]:
    assert state.target is not None
    phi = math.radians(float(state.phi_deg))
    theta = math.radians(float(state.theta_deg))
    target = tuple(float(component) for component in state.target)
    sin_theta = math.sin(theta)
    return (
        target[0] + float(state.radius) * sin_theta * math.cos(phi),
        target[1] + float(state.radius) * math.cos(theta),
        target[2] + float(state.radius) * sin_theta * math.sin(phi),
    )


def _assert_clear(animation, source: TerrainScatterSource, clearance: TerrainClearance, fps: int) -> None:
    total_frames = animation.get_frame_count(fps)
    for frame in range(total_frames):
        time = frame / fps
        state = animation.evaluate(time)
        assert state is not None
        assert state.target is not None
        assert -1e-4 <= state.target[0] <= source.terrain_width + 1e-4
        assert -1e-4 <= state.target[2] <= source.terrain_width + 1e-4
        eye = _eye_from_state(state)
        assert -1e-4 <= eye[0] <= source.terrain_width + 1e-4
        assert -1e-4 <= eye[2] <= source.terrain_width + 1e-4
        row, col = source.contract_to_pixel(eye[0], eye[2])
        safe_height = source.sample_scaled_height(row, col) + clearance.minimum_height
        assert eye[1] + 1e-4 >= safe_height


@pytest.fixture
def flat_source() -> TerrainScatterSource:
    return TerrainScatterSource(
        np.zeros((64, 64), dtype=np.float32),
        z_scale=1.0,
        terrain_width=100.0,
    )


def test_orbit_rig_bakes_deterministically(flat_source: TerrainScatterSource) -> None:
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=2.0,
        radius=20.0,
        phi_start_deg=0.0,
        phi_end_deg=90.0,
        theta_start_deg=60.0,
        fov_start_deg=50.0,
        clearance=TerrainClearance(minimum_height=2.0),
    )

    anim_a = rig.bake(flat_source, samples_per_second=10)
    anim_b = rig.bake(flat_source, samples_per_second=10)

    assert anim_a.get_frame_count(10) == 21
    assert anim_a.keyframe_count == anim_b.keyframe_count
    assert [
        (
            keyframe.time,
            keyframe.phi_deg,
            keyframe.theta_deg,
            keyframe.radius,
            keyframe.fov_deg,
            keyframe.target,
        )
        for keyframe in anim_a.get_keyframes()
    ] == [
        (
            keyframe.time,
            keyframe.phi_deg,
            keyframe.theta_deg,
            keyframe.radius,
            keyframe.fov_deg,
            keyframe.target,
        )
        for keyframe in anim_b.get_keyframes()
    ]

    midpoint = anim_a.evaluate(1.0)
    assert midpoint is not None
    assert midpoint.target == pytest.approx((50.0, 0.0, 50.0))
    assert midpoint.radius == pytest.approx(20.0, abs=1.0)
    _assert_clear(anim_a, flat_source, rig.clearance, fps=40)


def test_rail_rig_uses_constant_speed_and_lookahead(flat_source: TerrainScatterSource) -> None:
    rig = TerrainRailRig(
        path_xz=[(10.0, 10.0), (90.0, 10.0)],
        duration=2.0,
        camera_height_offset=15.0,
        look_ahead_distance=10.0,
        fov_deg=55.0,
    )

    animation = rig.bake(flat_source, samples_per_second=8)

    state_half = animation.evaluate(0.5)
    state_one = animation.evaluate(1.0)
    assert state_half is not None and state_one is not None
    assert state_half.target is not None and state_one.target is not None
    assert state_half.target[0] == pytest.approx(40.0, abs=1.0)
    assert state_one.target[0] == pytest.approx(60.0, abs=1.0)

    eye_one = _eye_from_state(state_one)
    assert eye_one[0] == pytest.approx(50.0, abs=1.0)
    assert eye_one[2] == pytest.approx(10.0, abs=1.0)
    _assert_clear(animation, flat_source, rig.clearance, fps=32)


def test_target_follow_rig_stays_behind_path_tangent(flat_source: TerrainScatterSource) -> None:
    rig = TerrainTargetFollowRig(
        target_path_xz=[(10.0, 30.0), (10.0, 90.0)],
        duration=2.0,
        radius=20.0,
        theta_deg=60.0,
        heading_offset_deg=180.0,
        fov_deg=45.0,
        clearance=TerrainClearance(minimum_height=2.0),
    )

    animation = rig.bake(flat_source, samples_per_second=8)
    state = animation.evaluate(1.0)
    assert state is not None
    assert state.target is not None

    eye = _eye_from_state(state)
    assert eye[2] < state.target[2]
    assert eye[1] > state.target[1]
    _assert_clear(animation, flat_source, rig.clearance, fps=32)


def test_clearance_lifts_eye_and_recomputes_orbit_parameters() -> None:
    source = TerrainScatterSource(
        np.full((32, 32), 20.0, dtype=np.float32),
        z_scale=1.0,
        terrain_width=100.0,
    )
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=10.0,
        phi_start_deg=0.0,
        phi_end_deg=0.0,
        theta_start_deg=90.0,
        clearance=TerrainClearance(minimum_height=5.0),
    )

    animation = rig.bake(source, samples_per_second=4)
    state = animation.evaluate(0.5)
    assert state is not None

    eye = _eye_from_state(state)
    assert eye[1] == pytest.approx(5.0, abs=1e-3)
    assert state.radius > 10.0
    _assert_clear(animation, source, rig.clearance, fps=16)


def test_orbit_rig_refines_when_verification_finds_clearance_hotspot() -> None:
    heightmap = np.zeros((64, 64), dtype=np.float32)
    heightmap[44, 31:34] = 30.0
    source = TerrainScatterSource(heightmap, z_scale=1.0, terrain_width=100.0)
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=20.0,
        phi_start_deg=0.0,
        phi_end_deg=180.0,
        theta_start_deg=90.0,
        clearance=TerrainClearance(minimum_height=5.0, max_refine_passes=8),
    )

    animation = rig.bake(source, samples_per_second=1)

    assert animation.keyframe_count > 2
    _assert_clear(animation, source, rig.clearance, fps=16)


def test_orbit_rig_preserves_large_angle_sweeps_across_bake_density(
    flat_source: TerrainScatterSource,
) -> None:
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=20.0,
        phi_start_deg=0.0,
        phi_end_deg=270.0,
        theta_start_deg=60.0,
    )

    sparse = rig.bake(flat_source, samples_per_second=1)
    dense = rig.bake(flat_source, samples_per_second=8)

    assert sparse.get_keyframes()[-1].phi_deg == pytest.approx(270.0)
    assert dense.get_keyframes()[-1].phi_deg == pytest.approx(270.0)
    assert sparse.evaluate(0.5).phi_deg == pytest.approx(135.0, abs=1.0)
    assert dense.evaluate(0.5).phi_deg == pytest.approx(135.0, abs=1.0)

    multi_turn = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=20.0,
        phi_start_deg=0.0,
        phi_end_deg=720.0,
        theta_start_deg=60.0,
    ).bake(flat_source, samples_per_second=1)
    assert multi_turn.get_keyframes()[-1].phi_deg == pytest.approx(720.0)
    assert multi_turn.evaluate(0.5).phi_deg == pytest.approx(360.0, abs=1.0)


def test_orbit_rig_dense_verification_prevents_narrow_clearance_overshoot() -> None:
    heightmap = np.zeros((64, 64), dtype=np.float32)
    for row, col, value in (
        (52, 4, 34.4708345729944),
        (49, 58, 11.881316632473617),
        (60, 19, 10.809122929711776),
        (16, 41, 14.211712986254597),
        (44, 24, 25.336562926595498),
    ):
        heightmap[max(0, row - 1):min(64, row + 2), max(0, col - 1):min(64, col + 2)] = value

    source = TerrainScatterSource(heightmap, z_scale=1.0, terrain_width=100.0)
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=25.528397973993567,
        phi_start_deg=176.51508390872843,
        phi_end_deg=317.25958154840686,
        theta_start_deg=96.6071845743734,
        theta_end_deg=88.03130490228422,
        clearance=TerrainClearance(minimum_height=5.426188480387692, max_refine_passes=8),
    )

    animation = rig.bake(source, samples_per_second=3)

    assert animation.keyframe_count > 4
    _assert_clear(animation, source, rig.clearance, fps=240)


def test_rail_rig_refines_when_target_interpolation_overshoots_corner(
    flat_source: TerrainScatterSource,
) -> None:
    rig = TerrainRailRig(
        path_xz=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)],
        duration=2.0,
        camera_height_offset=10.0,
        look_ahead_distance=0.0,
        clearance=TerrainClearance(minimum_height=1.0, max_refine_passes=8),
    )

    animation = rig.bake(flat_source, samples_per_second=4)

    assert animation.keyframe_count > 9
    _assert_clear(animation, flat_source, rig.clearance, fps=240)


def test_rail_rig_keeps_nonzero_target_at_path_end(flat_source: TerrainScatterSource) -> None:
    rig = TerrainRailRig(
        path_xz=[(10.0, 10.0), (90.0, 10.0)],
        duration=1.0,
        camera_height_offset=0.0,
        look_ahead_distance=20.0,
        target_height_offset=0.0,
        clearance=TerrainClearance(),
    )

    animation = rig.bake(flat_source, samples_per_second=4)
    final_state = animation.evaluate(1.0)

    assert final_state is not None
    assert final_state.target is not None
    eye = _eye_from_state(final_state)
    assert final_state.target[0] > eye[0]
    assert final_state.target[0] == pytest.approx(100.0)
    assert eye[0] == pytest.approx(90.0)


def test_rail_rig_keeps_forward_target_when_path_ends_on_boundary(
    flat_source: TerrainScatterSource,
) -> None:
    rig = TerrainRailRig(
        path_xz=[(10.0, 10.0), (100.0, 10.0)],
        duration=1.0,
        camera_height_offset=0.0,
        look_ahead_distance=20.0,
        target_height_offset=0.0,
        clearance=TerrainClearance(),
    )

    animation = rig.bake(flat_source, samples_per_second=4)
    final_state = animation.evaluate(1.0)

    assert final_state is not None
    assert final_state.target is not None
    eye = _eye_from_state(final_state)
    assert final_state.target[0] == pytest.approx(100.0)
    assert eye[0] < final_state.target[0]
    assert eye[0] < 100.0
    _assert_clear(animation, flat_source, rig.clearance, fps=32)


def test_rigs_validate_paths_and_bounds(flat_source: TerrainScatterSource) -> None:
    with pytest.raises(ValueError, match="unique points"):
        TerrainRailRig(
            path_xz=[(10.0, 10.0), (10.0, 10.0)],
            duration=1.0,
            camera_height_offset=5.0,
            look_ahead_distance=1.0,
        )

    rig = TerrainTargetFollowRig(
        target_path_xz=[(10.0, 10.0), (120.0, 10.0)],
        duration=1.0,
        radius=10.0,
    )
    with pytest.raises(ValueError, match="terrain bounds"):
        rig.bake(flat_source, samples_per_second=4)

    with pytest.raises(TypeError, match="TerrainClearance"):
        TerrainOrbitRig(
            target_xz=(50.0, 50.0),
            duration=1.0,
            radius=10.0,
            phi_start_deg=0.0,
            phi_end_deg=90.0,
            clearance=None,
        )


def test_rigs_reject_invalid_polar_angles() -> None:
    with pytest.raises(ValueError, match=r"\[0, 180\)"):
        TerrainOrbitRig(
            target_xz=(50.0, 50.0),
            duration=1.0,
            radius=10.0,
            phi_start_deg=0.0,
            phi_end_deg=90.0,
            theta_start_deg=180.0,
        )

    with pytest.raises(ValueError, match=r"\[0, 180\)"):
        TerrainOrbitRig(
            target_xz=(50.0, 50.0),
            duration=1.0,
            radius=10.0,
            phi_start_deg=0.0,
            phi_end_deg=90.0,
            theta_end_deg=180.0,
        )

    with pytest.raises(ValueError, match=r"\[0, 180\)"):
        TerrainTargetFollowRig(
            target_path_xz=[(10.0, 10.0), (20.0, 10.0)],
            duration=1.0,
            radius=10.0,
            theta_deg=180.0,
        )


def test_replace_keyframes_accepts_rig_generated_keyframes(flat_source: TerrainScatterSource) -> None:
    rig = TerrainOrbitRig(
        target_xz=(50.0, 50.0),
        duration=1.0,
        radius=20.0,
        phi_start_deg=0.0,
        phi_end_deg=90.0,
    )
    animation = rig.bake(flat_source, samples_per_second=4)
    keyframes = animation.get_keyframes()

    animation.replace_keyframes(
        keyframes
        + [CameraKeyframe(1.5, 120.0, 60.0, 25.0, 55.0, target=(50.0, 0.0, 50.0))]
    )

    assert animation.keyframe_count == len(keyframes) + 1
    assert animation.get_keyframes()[-1].time == pytest.approx(1.5)
