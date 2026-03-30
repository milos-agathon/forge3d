"""Terrain camera rig authoring helpers that bake to CameraAnimation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional, Sequence, Tuple

from .animation import CameraAnimation, CameraKeyframe
from .terrain_scatter import TerrainScatterSource

_EPSILON = 1e-6
_CLEARANCE_EPSILON = 1e-4


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: float, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _non_negative_float(value: float, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _polar_angle_deg(value: float, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0 or result >= 180.0:
        raise ValueError(f"{name} must be in [0, 180)")
    return result


def _coerce_xz_point(point: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{name} must be a 2D (x, z) point")
    x = _finite_float(point[0], name=f"{name}[0]")
    z = _finite_float(point[1], name=f"{name}[1]")
    return (x, z)


def _coerce_path(points: Sequence[Sequence[float]], *, name: str) -> tuple[tuple[float, float], ...]:
    if len(points) < 2:
        raise ValueError(f"{name} must contain at least 2 points")

    collapsed: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        current = _coerce_xz_point(point, name=f"{name}[{index}]")
        if not collapsed:
            collapsed.append(current)
            continue
        dx = current[0] - collapsed[-1][0]
        dz = current[1] - collapsed[-1][1]
        if math.hypot(dx, dz) > _EPSILON:
            collapsed.append(current)

    if len(collapsed) < 2:
        raise ValueError(f"{name} must contain at least 2 unique points")
    return tuple(collapsed)


def _validate_point_within_source(
    source: TerrainScatterSource,
    point: tuple[float, float],
    *,
    name: str,
) -> tuple[float, float]:
    x, z = point
    terrain_width = float(source.terrain_width)
    if x < 0.0 or x > terrain_width or z < 0.0 or z > terrain_width:
        raise ValueError(
            f"{name} must stay within [0, {terrain_width}] terrain bounds, got {(x, z)}"
        )
    return point


def _terrain_height(source: TerrainScatterSource, x: float, z: float) -> float:
    row, col = source.contract_to_pixel(x, z)
    return float(source.sample_scaled_height(row, col))


def _lerp(start: float, end: float, alpha: float) -> float:
    return float(start + (end - start) * alpha)


def _sample_times(duration: float, samples_per_second: int) -> list[float]:
    total_frames = int(math.ceil(duration * samples_per_second)) + 1
    return [min(duration, frame / samples_per_second) for frame in range(total_frames)]


def _orbit_eye(
    target: tuple[float, float, float],
    radius: float,
    phi_deg: float,
    theta_deg: float,
) -> tuple[float, float, float]:
    phi = math.radians(phi_deg)
    theta = math.radians(theta_deg)
    sin_theta = math.sin(theta)
    return (
        float(target[0] + radius * sin_theta * math.cos(phi)),
        float(target[1] + radius * math.cos(theta)),
        float(target[2] + radius * sin_theta * math.sin(phi)),
    )


def _orbit_from_eye_target(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    dx = float(eye[0] - target[0])
    dy = float(eye[1] - target[1])
    dz = float(eye[2] - target[2])
    radius = math.sqrt(dx * dx + dy * dy + dz * dz)
    if radius <= _EPSILON:
        raise ValueError("camera eye and target must not coincide")
    phi_deg = math.degrees(math.atan2(dz, dx))
    theta_deg = math.degrees(math.acos(max(-1.0, min(1.0, dy / radius))))
    return (float(phi_deg), float(theta_deg), float(radius))


def _unwrap_phi(previous_phi: float, current_phi: float) -> float:
    phi = float(current_phi)
    while phi - previous_phi > 180.0:
        phi -= 360.0
    while phi - previous_phi < -180.0:
        phi += 360.0
    return phi


def _align_angle_to_reference(angle: float, reference: float) -> float:
    aligned = float(angle)
    reference = float(reference)
    while aligned - reference > 180.0:
        aligned -= 360.0
    while aligned - reference < -180.0:
        aligned += 360.0
    return aligned


def _unwrap_keyframes(keyframes: Iterable[CameraKeyframe]) -> list[CameraKeyframe]:
    ordered = sorted(keyframes, key=lambda keyframe: keyframe.time)
    if not ordered:
        return []

    result: list[CameraKeyframe] = [ordered[0]]
    previous_phi = float(ordered[0].phi_deg)
    for keyframe in ordered[1:]:
        phi = _unwrap_phi(previous_phi, float(keyframe.phi_deg))
        updated = CameraKeyframe(
            float(keyframe.time),
            phi,
            float(keyframe.theta_deg),
            float(keyframe.radius),
            float(keyframe.fov_deg),
            target=tuple(keyframe.target) if keyframe.target is not None else None,
        )
        result.append(updated)
        previous_phi = phi
    return result


@dataclass(frozen=True)
class _PathSample:
    x: float
    z: float
    tangent_x: float
    tangent_z: float
    distance: float


class _PolylinePath:
    def __init__(self, points: Sequence[tuple[float, float]]):
        self.points = tuple(points)
        lengths: list[float] = [0.0]
        self._segment_lengths: list[float] = []
        for start, end in zip(self.points, self.points[1:]):
            seg_len = math.hypot(end[0] - start[0], end[1] - start[1])
            self._segment_lengths.append(seg_len)
            lengths.append(lengths[-1] + seg_len)
        self._lengths = tuple(lengths)
        self.total_length = float(lengths[-1])
        if self.total_length <= _EPSILON:
            raise ValueError("path length must be > 0")

    def sample_distance(self, distance: float, *, extrapolate: bool = False) -> _PathSample:
        raw_distance = float(distance)
        clamped = max(0.0, min(raw_distance, self.total_length))
        if clamped <= 0.0:
            start = self.points[0]
            end = self.points[1]
            tangent_x, tangent_z = self._segment_tangent(start, end)
            if extrapolate and raw_distance < 0.0:
                return _PathSample(
                    start[0] + tangent_x * raw_distance,
                    start[1] + tangent_z * raw_distance,
                    tangent_x,
                    tangent_z,
                    raw_distance,
                )
            return _PathSample(start[0], start[1], tangent_x, tangent_z, 0.0)
        if clamped >= self.total_length:
            start = self.points[-2]
            end = self.points[-1]
            tangent_x, tangent_z = self._segment_tangent(start, end)
            if extrapolate and raw_distance > self.total_length:
                extension = raw_distance - self.total_length
                return _PathSample(
                    end[0] + tangent_x * extension,
                    end[1] + tangent_z * extension,
                    tangent_x,
                    tangent_z,
                    raw_distance,
                )
            return _PathSample(end[0], end[1], tangent_x, tangent_z, self.total_length)

        for index, segment_length in enumerate(self._segment_lengths):
            start_distance = self._lengths[index]
            end_distance = self._lengths[index + 1]
            if clamped <= end_distance or index == len(self._segment_lengths) - 1:
                start = self.points[index]
                end = self.points[index + 1]
                alpha = 0.0 if segment_length <= _EPSILON else (
                    (clamped - start_distance) / segment_length
                )
                tangent_x, tangent_z = self._segment_tangent(start, end)
                return _PathSample(
                    x=_lerp(start[0], end[0], alpha),
                    z=_lerp(start[1], end[1], alpha),
                    tangent_x=tangent_x,
                    tangent_z=tangent_z,
                    distance=clamped,
                )

        start = self.points[-2]
        end = self.points[-1]
        tangent_x, tangent_z = self._segment_tangent(start, end)
        return _PathSample(end[0], end[1], tangent_x, tangent_z, self.total_length)

    @staticmethod
    def _segment_tangent(
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= _EPSILON:
            raise ValueError("path contains a degenerate segment")
        return (dx / length, dz / length)


def _offset_path_sample(sample: _PathSample, lateral_offset: float) -> tuple[float, float]:
    lateral_x = -sample.tangent_z
    lateral_z = sample.tangent_x
    return (
        float(sample.x + lateral_x * lateral_offset),
        float(sample.z + lateral_z * lateral_offset),
    )


@dataclass(frozen=True)
class TerrainClearance:
    minimum_height: float = 0.0
    max_refine_passes: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_height",
            _non_negative_float(self.minimum_height, name="minimum_height"),
        )
        passes = int(self.max_refine_passes)
        if passes < 0:
            raise ValueError("max_refine_passes must be >= 0")
        object.__setattr__(self, "max_refine_passes", passes)


def _coerce_clearance(clearance: TerrainClearance) -> TerrainClearance:
    if not isinstance(clearance, TerrainClearance):
        raise TypeError("clearance must be a TerrainClearance")
    return clearance


class _BaseTerrainRig:
    clearance: TerrainClearance
    duration: float

    def _validate_source(self, source: TerrainScatterSource) -> None:
        if not isinstance(source, TerrainScatterSource):
            raise TypeError("source must be a TerrainScatterSource")

    def _sample_keyframe(self, source: TerrainScatterSource, time: float) -> CameraKeyframe:
        raise NotImplementedError

    def _normalize_keyframes(self, keyframes: Iterable[CameraKeyframe]) -> list[CameraKeyframe]:
        return _unwrap_keyframes(keyframes)

    def _validate_animation_sample(
        self,
        animation: CameraAnimation,
        source: TerrainScatterSource,
        time: float,
    ) -> Optional[str]:
        state = animation.evaluate(time)
        if state is None:
            return "animation evaluation returned None"
        if state.target is None:
            return "target-aware rig animation lost its target during evaluation"
        target = tuple(float(component) for component in state.target)
        terrain_width = float(source.terrain_width)
        if (
            target[0] < -_CLEARANCE_EPSILON
            or target[0] > terrain_width + _CLEARANCE_EPSILON
            or target[2] < -_CLEARANCE_EPSILON
            or target[2] > terrain_width + _CLEARANCE_EPSILON
        ):
            return f"camera target left terrain bounds at time={time:.3f}"
        eye = _orbit_eye(
            target,
            float(state.radius),
            float(state.phi_deg),
            float(state.theta_deg),
        )
        if (
            eye[0] < -_CLEARANCE_EPSILON
            or eye[0] > terrain_width + _CLEARANCE_EPSILON
            or eye[2] < -_CLEARANCE_EPSILON
            or eye[2] > terrain_width + _CLEARANCE_EPSILON
        ):
            return f"camera eye left terrain bounds at time={time:.3f}"
        safe_height = _terrain_height(source, eye[0], eye[2]) + self.clearance.minimum_height
        if eye[1] + _CLEARANCE_EPSILON < safe_height:
            return f"camera eye violated clearance at time={time:.3f}"
        return None

    def _bake_keyframes(
        self,
        source: TerrainScatterSource,
        samples_per_second: int,
    ) -> list[CameraKeyframe]:
        return self._normalize_keyframes(
            self._sample_keyframe(source, time)
            for time in _sample_times(self.duration, samples_per_second)
        )

    def _verification_times(
        self,
        animation: CameraAnimation,
        samples_per_second: int,
    ) -> list[float]:
        # Dense verification is cheap compared to manual flyover debugging.
        # A coarse 4x bake-rate sweep misses narrow Catmull-Rom overshoots.
        verify_sps = max(samples_per_second * 32, 240)
        times = {round(time, 9): time for time in _sample_times(self.duration, verify_sps)}
        for keyframe in animation.get_keyframes():
            time = float(keyframe.time)
            times.setdefault(round(time, 9), time)
        return sorted(times.values())

    def _refine_animation(
        self,
        animation: CameraAnimation,
        source: TerrainScatterSource,
        samples_per_second: int,
    ) -> CameraAnimation:
        failing_times: list[float] = []

        for _ in range(self.clearance.max_refine_passes + 1):
            verify_times = self._verification_times(animation, samples_per_second)
            failing_times = [
                time
                for time in verify_times
                if self._validate_animation_sample(animation, source, time) is not None
            ]
            if not failing_times:
                return animation

            existing_times = {
                round(float(keyframe.time), 9)
                for keyframe in animation.get_keyframes()
            }
            inserted = False
            keyframes = list(animation.get_keyframes())
            for time in failing_times:
                rounded = round(float(time), 9)
                if rounded in existing_times:
                    continue
                keyframes.append(self._sample_keyframe(source, float(time)))
                existing_times.add(rounded)
                inserted = True

            if not inserted:
                message = self._validate_animation_sample(animation, source, failing_times[0])
                raise ValueError(message or "failed to refine rig animation")

            animation.replace_keyframes(self._normalize_keyframes(keyframes))

        verify_times = self._verification_times(animation, samples_per_second)
        failing_times = [
            time
            for time in verify_times
            if self._validate_animation_sample(animation, source, time) is not None
        ]
        if not failing_times:
            return animation
        final_time = failing_times[0]
        message = self._validate_animation_sample(animation, source, final_time)
        raise ValueError(message or "failed to satisfy clearance constraints after refinement")

    def bake(
        self,
        source: TerrainScatterSource,
        *,
        samples_per_second: int = 60,
    ) -> CameraAnimation:
        self._validate_source(source)
        samples_per_second = int(samples_per_second)
        if samples_per_second <= 0:
            raise ValueError("samples_per_second must be > 0")

        animation = CameraAnimation()
        animation.replace_keyframes(self._bake_keyframes(source, samples_per_second))
        return self._refine_animation(animation, source, samples_per_second)


@dataclass(frozen=True)
class TerrainOrbitRig(_BaseTerrainRig):
    target_xz: tuple[float, float]
    duration: float
    radius: float
    phi_start_deg: float
    phi_end_deg: float
    theta_start_deg: float = 45.0
    theta_end_deg: Optional[float] = None
    radius_end: Optional[float] = None
    fov_start_deg: float = 55.0
    fov_end_deg: Optional[float] = None
    target_height_offset: float = 0.0
    clearance: TerrainClearance = field(default_factory=TerrainClearance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_xz", _coerce_xz_point(self.target_xz, name="target_xz"))
        object.__setattr__(self, "duration", _positive_float(self.duration, name="duration"))
        object.__setattr__(self, "radius", _positive_float(self.radius, name="radius"))
        object.__setattr__(self, "clearance", _coerce_clearance(self.clearance))
        object.__setattr__(self, "phi_start_deg", _finite_float(self.phi_start_deg, name="phi_start_deg"))
        object.__setattr__(self, "phi_end_deg", _finite_float(self.phi_end_deg, name="phi_end_deg"))
        object.__setattr__(
            self,
            "theta_start_deg",
            _polar_angle_deg(self.theta_start_deg, name="theta_start_deg"),
        )
        if self.theta_end_deg is not None:
            object.__setattr__(
                self,
                "theta_end_deg",
                _polar_angle_deg(self.theta_end_deg, name="theta_end_deg"),
            )
        if self.radius_end is not None:
            object.__setattr__(
                self,
                "radius_end",
                _positive_float(self.radius_end, name="radius_end"),
            )
        object.__setattr__(self, "fov_start_deg", _positive_float(self.fov_start_deg, name="fov_start_deg"))
        if self.fov_end_deg is not None:
            object.__setattr__(
                self,
                "fov_end_deg",
                _positive_float(self.fov_end_deg, name="fov_end_deg"),
            )
        object.__setattr__(
            self,
            "target_height_offset",
            _finite_float(self.target_height_offset, name="target_height_offset"),
        )

    def _normalize_keyframes(self, keyframes: Iterable[CameraKeyframe]) -> list[CameraKeyframe]:
        ordered = sorted(keyframes, key=lambda keyframe: keyframe.time)
        if not ordered:
            return []

        result: list[CameraKeyframe] = []
        for keyframe in ordered:
            alpha = 0.0 if self.duration <= _EPSILON else min(max(float(keyframe.time) / self.duration, 0.0), 1.0)
            reference_phi = _lerp(self.phi_start_deg, self.phi_end_deg, alpha)
            phi = _align_angle_to_reference(float(keyframe.phi_deg), reference_phi)
            result.append(
                CameraKeyframe(
                    float(keyframe.time),
                    phi,
                    float(keyframe.theta_deg),
                    float(keyframe.radius),
                    float(keyframe.fov_deg),
                    target=tuple(keyframe.target) if keyframe.target is not None else None,
                )
            )
        return result

    def _sample_keyframe(self, source: TerrainScatterSource, time: float) -> CameraKeyframe:
        target_xz = _validate_point_within_source(source, self.target_xz, name="target_xz")
        alpha = 0.0 if self.duration <= _EPSILON else min(max(time / self.duration, 0.0), 1.0)
        radius_end = self.radius if self.radius_end is None else self.radius_end
        theta_end = self.theta_start_deg if self.theta_end_deg is None else self.theta_end_deg
        fov_end = self.fov_start_deg if self.fov_end_deg is None else self.fov_end_deg

        target = (
            target_xz[0],
            _terrain_height(source, target_xz[0], target_xz[1]) + self.target_height_offset,
            target_xz[1],
        )
        eye = _orbit_eye(
            target,
            _lerp(self.radius, radius_end, alpha),
            _lerp(self.phi_start_deg, self.phi_end_deg, alpha),
            _lerp(self.theta_start_deg, theta_end, alpha),
        )
        eye = _apply_clearance(source, eye, clearance=self.clearance)
        phi_deg, theta_deg, radius = _orbit_from_eye_target(eye, target)
        return CameraKeyframe(
            time,
            phi_deg,
            theta_deg,
            radius,
            _lerp(self.fov_start_deg, fov_end, alpha),
            target=target,
        )


@dataclass(frozen=True)
class TerrainRailRig(_BaseTerrainRig):
    path_xz: tuple[tuple[float, float], ...]
    duration: float
    camera_height_offset: float
    look_ahead_distance: float
    lateral_offset: float = 0.0
    target_height_offset: float = 0.0
    fov_deg: float = 55.0
    clearance: TerrainClearance = field(default_factory=TerrainClearance)

    def __init__(
        self,
        path_xz: Sequence[Sequence[float]],
        duration: float,
        camera_height_offset: float,
        look_ahead_distance: float,
        lateral_offset: float = 0.0,
        target_height_offset: float = 0.0,
        fov_deg: float = 55.0,
        clearance: TerrainClearance = TerrainClearance(),
    ) -> None:
        object.__setattr__(self, "path_xz", _coerce_path(path_xz, name="path_xz"))
        object.__setattr__(self, "duration", _positive_float(duration, name="duration"))
        object.__setattr__(
            self,
            "camera_height_offset",
            _finite_float(camera_height_offset, name="camera_height_offset"),
        )
        object.__setattr__(
            self,
            "look_ahead_distance",
            _non_negative_float(look_ahead_distance, name="look_ahead_distance"),
        )
        object.__setattr__(self, "lateral_offset", _finite_float(lateral_offset, name="lateral_offset"))
        object.__setattr__(
            self,
            "target_height_offset",
            _finite_float(target_height_offset, name="target_height_offset"),
        )
        object.__setattr__(self, "fov_deg", _positive_float(fov_deg, name="fov_deg"))
        object.__setattr__(self, "clearance", _coerce_clearance(clearance))

    def _sample_keyframe(self, source: TerrainScatterSource, time: float) -> CameraKeyframe:
        for index, point in enumerate(self.path_xz):
            _validate_point_within_source(source, point, name=f"path_xz[{index}]")
        path = _PolylinePath(self.path_xz)
        alpha = 0.0 if self.duration <= _EPSILON else min(max(time / self.duration, 0.0), 1.0)
        distance = path.total_length * alpha
        eye_sample = path.sample_distance(distance)

        eye_x, eye_z = _offset_path_sample(eye_sample, self.lateral_offset)
        _validate_point_within_source(source, (eye_x, eye_z), name="rail camera eye")

        terrain_width = float(source.terrain_width)
        target_sample = path.sample_distance(distance + self.look_ahead_distance, extrapolate=True)
        target_x = float(min(max(target_sample.x, 0.0), terrain_width))
        target_z = float(min(max(target_sample.z, 0.0), terrain_width))
        if math.hypot(target_x - eye_x, target_z - eye_z) <= _EPSILON:
            cell_size = terrain_width / max(max(source.height, source.width) - 1, 1)
            fallback_distance = max(
                self.look_ahead_distance,
                path.total_length / max(len(path.points) - 1, 1),
                cell_size,
            )
            forward_sample = path.sample_distance(distance + fallback_distance, extrapolate=True)
            target_x = float(min(max(forward_sample.x, 0.0), terrain_width))
            target_z = float(min(max(forward_sample.z, 0.0), terrain_width))
            if math.hypot(target_x - eye_x, target_z - eye_z) <= _EPSILON:
                # Preserve the forward boundary target and back the eye off slightly
                # rather than flipping the shot back toward the start of the rail.
                eye_backoff = min(distance, cell_size)
                if eye_backoff > 0.0:
                    candidate_eye = _offset_path_sample(
                        path.sample_distance(distance - eye_backoff),
                        self.lateral_offset,
                    )
                    _validate_point_within_source(source, candidate_eye, name="rail camera eye")
                    if math.hypot(target_x - candidate_eye[0], target_z - candidate_eye[1]) > _EPSILON:
                        eye_x, eye_z = candidate_eye
            if math.hypot(target_x - eye_x, target_z - eye_z) <= _EPSILON:
                raise ValueError(
                    "rail camera target collapsed onto the camera eye; adjust the path or height offsets"
                )
        _validate_point_within_source(source, (target_x, target_z), name="rail camera target")

        target = (
            target_x,
            _terrain_height(source, target_x, target_z) + self.target_height_offset,
            target_z,
        )
        eye = (
            eye_x,
            _terrain_height(source, eye_x, eye_z) + self.camera_height_offset,
            eye_z,
        )
        eye = _apply_clearance(source, eye, clearance=self.clearance)
        phi_deg, theta_deg, radius = _orbit_from_eye_target(eye, target)
        return CameraKeyframe(
            time,
            phi_deg,
            theta_deg,
            radius,
            self.fov_deg,
            target=target,
        )


@dataclass(frozen=True)
class TerrainTargetFollowRig(_BaseTerrainRig):
    target_path_xz: tuple[tuple[float, float], ...]
    duration: float
    radius: float
    theta_deg: float = 45.0
    heading_offset_deg: float = 180.0
    target_height_offset: float = 0.0
    fov_deg: float = 55.0
    clearance: TerrainClearance = field(default_factory=TerrainClearance)

    def __init__(
        self,
        target_path_xz: Sequence[Sequence[float]],
        duration: float,
        radius: float,
        theta_deg: float = 45.0,
        heading_offset_deg: float = 180.0,
        target_height_offset: float = 0.0,
        fov_deg: float = 55.0,
        clearance: TerrainClearance = TerrainClearance(),
    ) -> None:
        object.__setattr__(
            self,
            "target_path_xz",
            _coerce_path(target_path_xz, name="target_path_xz"),
        )
        object.__setattr__(self, "duration", _positive_float(duration, name="duration"))
        object.__setattr__(self, "radius", _positive_float(radius, name="radius"))
        object.__setattr__(self, "theta_deg", _polar_angle_deg(theta_deg, name="theta_deg"))
        object.__setattr__(
            self,
            "heading_offset_deg",
            _finite_float(heading_offset_deg, name="heading_offset_deg"),
        )
        object.__setattr__(
            self,
            "target_height_offset",
            _finite_float(target_height_offset, name="target_height_offset"),
        )
        object.__setattr__(self, "fov_deg", _positive_float(fov_deg, name="fov_deg"))
        object.__setattr__(self, "clearance", _coerce_clearance(clearance))

    def _sample_keyframe(self, source: TerrainScatterSource, time: float) -> CameraKeyframe:
        for index, point in enumerate(self.target_path_xz):
            _validate_point_within_source(source, point, name=f"target_path_xz[{index}]")
        path = _PolylinePath(self.target_path_xz)
        alpha = 0.0 if self.duration <= _EPSILON else min(max(time / self.duration, 0.0), 1.0)
        distance = path.total_length * alpha
        target_sample = path.sample_distance(distance)

        target = (
            target_sample.x,
            _terrain_height(source, target_sample.x, target_sample.z) + self.target_height_offset,
            target_sample.z,
        )
        heading_deg = math.degrees(math.atan2(target_sample.tangent_z, target_sample.tangent_x))
        desired_phi = heading_deg + self.heading_offset_deg
        eye = _orbit_eye(target, self.radius, desired_phi, self.theta_deg)
        _validate_point_within_source(source, (eye[0], eye[2]), name="follow camera eye")
        eye = _apply_clearance(source, eye, clearance=self.clearance)
        phi_deg, theta_deg, radius = _orbit_from_eye_target(eye, target)
        return CameraKeyframe(
            time,
            phi_deg,
            theta_deg,
            radius,
            self.fov_deg,
            target=target,
        )


def _apply_clearance(
    source: TerrainScatterSource,
    eye: tuple[float, float, float],
    *,
    clearance: TerrainClearance,
) -> tuple[float, float, float]:
    _validate_point_within_source(source, (eye[0], eye[2]), name="camera eye")
    safe_height = _terrain_height(source, eye[0], eye[2]) + clearance.minimum_height
    if eye[1] < safe_height:
        return (eye[0], safe_height, eye[2])
    return eye


__all__ = [
    "TerrainClearance",
    "TerrainOrbitRig",
    "TerrainRailRig",
    "TerrainTargetFollowRig",
]
