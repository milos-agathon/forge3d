# python/forge3d/animation.py
"""
Camera animation module for keyframe-based camera paths.

Provides:
    CameraAnimation     - Keyframe animation with cubic interpolation
    CameraKeyframe      - Inspectable camera keyframe storage
    CameraState         - Interpolated camera state at a given time

Example:
    >>> from forge3d.animation import CameraAnimation
    >>> anim = CameraAnimation()
    >>> anim.add_keyframe(time=0.0, phi=0, theta=45, radius=5000, fov=60)
    >>> anim.add_keyframe(time=5.0, phi=180, theta=30, radius=3000, fov=60)
    >>> state = anim.evaluate(2.5)
    >>> print(f"phi={state.phi_deg}, theta={state.theta_deg}")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

from ._native import get_native_module

_NATIVE = get_native_module()


def _coerce_target(
    target: Optional[Tuple[float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    if target is None:
        return None
    if len(target) != 3:
        raise ValueError("target must be a 3-tuple when provided")
    return (float(target[0]), float(target[1]), float(target[2]))


# Re-export native classes if available
if _NATIVE is not None and all(
    hasattr(_NATIVE, name)
    for name in ("CameraAnimation", "CameraKeyframe", "CameraState")
):
    CameraAnimation = _NATIVE.CameraAnimation
    CameraKeyframe = _NATIVE.CameraKeyframe
    CameraState = _NATIVE.CameraState
else:
    @dataclass(frozen=True)
    class CameraKeyframe:
        """A single camera keyframe."""

        time: float
        phi_deg: float
        theta_deg: float
        radius: float
        fov_deg: float
        target: Optional[Tuple[float, float, float]] = None

        def __init__(
            self,
            time: float,
            phi: float,
            theta: float,
            radius: float,
            fov: float,
            target: Optional[Tuple[float, float, float]] = None,
        ) -> None:
            object.__setattr__(self, "time", float(time))
            object.__setattr__(self, "phi_deg", float(phi))
            object.__setattr__(self, "theta_deg", float(theta))
            object.__setattr__(self, "radius", float(radius))
            object.__setattr__(self, "fov_deg", float(fov))
            object.__setattr__(self, "target", _coerce_target(target))

        def __repr__(self) -> str:
            return (
                "CameraKeyframe("
                f"time={self.time:.2f}, phi={self.phi_deg:.2f}, theta={self.theta_deg:.2f}, "
                f"radius={self.radius:.2f}, fov={self.fov_deg:.2f}, target={self.target})"
            )

    @dataclass(frozen=True)
    class CameraState:
        """Interpolated camera state at a given time."""

        phi_deg: float
        theta_deg: float
        radius: float
        fov_deg: float
        target: Optional[Tuple[float, float, float]] = None

        def __repr__(self) -> str:
            return (
                "CameraState("
                f"phi={self.phi_deg:.2f}, theta={self.theta_deg:.2f}, "
                f"radius={self.radius:.2f}, fov={self.fov_deg:.2f}, target={self.target})"
            )

    class CameraAnimation:
        """Camera animation with keyframe storage and cubic Hermite interpolation."""

        def __init__(self):
            self._keyframes: list[CameraKeyframe] = []

        def add_keyframe(
            self,
            time: float,
            phi: float,
            theta: float,
            radius: float,
            fov: float,
            target: Optional[Tuple[float, float, float]] = None,
        ) -> None:
            """Add a keyframe. Keyframes are sorted by time automatically."""
            self._keyframes.append(
                CameraKeyframe(time, phi, theta, radius, fov, target=target)
            )
            self._keyframes.sort(key=lambda keyframe: keyframe.time)

        # Alias for compatibility with native API
        def add_keyframe_py(
            self,
            time: float,
            phi: float,
            theta: float,
            radius: float,
            fov: float,
            target: Optional[Tuple[float, float, float]] = None,
        ) -> None:
            self.add_keyframe(time, phi, theta, radius, fov, target=target)

        def get_keyframes(self) -> list[CameraKeyframe]:
            """Return a copy of the stored keyframes."""
            return list(self._keyframes)

        def replace_keyframes(self, keyframes: Iterable[CameraKeyframe]) -> None:
            """Replace all keyframes from an iterable of CameraKeyframe objects."""
            updated = list(keyframes)
            for keyframe in updated:
                if not isinstance(keyframe, CameraKeyframe):
                    raise TypeError(
                        "replace_keyframes expects an iterable of CameraKeyframe objects"
                    )
            self._keyframes = sorted(updated, key=lambda keyframe: keyframe.time)

        def clear_keyframes(self) -> None:
            """Remove all keyframes."""
            self._keyframes.clear()

        @property
        def duration(self) -> float:
            """Get animation duration in seconds."""
            if not self._keyframes:
                return 0.0
            return self._keyframes[-1].time

        @property
        def keyframe_count(self) -> int:
            """Get number of keyframes."""
            return len(self._keyframes)

        def get_frame_count(self, fps: int) -> int:
            """Get total frame count for given fps."""
            if self.duration <= 0 or fps <= 0:
                return 0
            import math

            return int(math.ceil(self.duration * fps)) + 1

        @staticmethod
        def _cubic_hermite(p0: float, p1: float, p2: float, p3: float, t: float) -> float:
            t2 = t * t
            t3 = t2 * t
            h1 = -0.5 * t3 + t2 - 0.5 * t
            h2 = 1.5 * t3 - 2.5 * t2 + 1.0
            h3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            h4 = 0.5 * t3 - 0.5 * t2
            return h1 * p0 + h2 * p1 + h3 * p2 + h4 * p3

        @classmethod
        def _interpolate_target(
            cls,
            k0: CameraKeyframe,
            k1: CameraKeyframe,
            k2: CameraKeyframe,
            k3: CameraKeyframe,
            t: float,
        ) -> Optional[Tuple[float, float, float]]:
            if k1.target is None or k2.target is None:
                return None
            p0 = k0.target if k0.target is not None else k1.target
            p1 = k1.target
            p2 = k2.target
            p3 = k3.target if k3.target is not None else k2.target
            return (
                cls._cubic_hermite(p0[0], p1[0], p2[0], p3[0], t),
                cls._cubic_hermite(p0[1], p1[1], p2[1], p3[1], t),
                cls._cubic_hermite(p0[2], p1[2], p2[2], p3[2], t),
            )

        def evaluate(self, time: float) -> Optional[CameraState]:
            """Evaluate camera state at given time using cubic Hermite interpolation."""
            if not self._keyframes:
                return None

            n = len(self._keyframes)
            if n == 1:
                keyframe = self._keyframes[0]
                return CameraState(
                    keyframe.phi_deg,
                    keyframe.theta_deg,
                    keyframe.radius,
                    keyframe.fov_deg,
                    target=keyframe.target,
                )

            first_time = float(self._keyframes[0].time)
            last_time = float(self._keyframes[-1].time)
            time = max(first_time, min(float(time), last_time))

            idx = 0
            for i, keyframe in enumerate(self._keyframes):
                if keyframe.time > time:
                    idx = max(0, i - 1)
                    break
                idx = i

            if idx >= n - 1:
                idx = n - 2

            k1 = self._keyframes[idx]
            k2 = self._keyframes[idx + 1]
            k0 = self._keyframes[idx - 1] if idx > 0 else k1
            k3 = self._keyframes[idx + 2] if idx + 2 < n else k2

            segment_duration = k2.time - k1.time
            t = (time - k1.time) / segment_duration if segment_duration > 0 else 0.0

            return CameraState(
                phi_deg=self._cubic_hermite(
                    k0.phi_deg, k1.phi_deg, k2.phi_deg, k3.phi_deg, t
                ),
                theta_deg=self._cubic_hermite(
                    k0.theta_deg, k1.theta_deg, k2.theta_deg, k3.theta_deg, t
                ),
                radius=self._cubic_hermite(k0.radius, k1.radius, k2.radius, k3.radius, t),
                fov_deg=self._cubic_hermite(
                    k0.fov_deg, k1.fov_deg, k2.fov_deg, k3.fov_deg, t
                ),
                target=self._interpolate_target(k0, k1, k2, k3, t),
            )

        # Alias for native compatibility
        def evaluate_py(self, time: float) -> Optional[CameraState]:
            return self.evaluate(time)

        def __repr__(self) -> str:
            return (
                f"CameraAnimation(keyframes={self.keyframe_count}, "
                f"duration={self.duration:.2f}s)"
            )


class RenderConfig:
    """Configuration for offline animation rendering."""

    def __init__(
        self,
        output_dir: str = "./frames",
        fps: int = 30,
        width: int = 1920,
        height: int = 1080,
        filename_prefix: str = "frame",
        frame_digits: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.width = width
        self.height = height
        self.filename_prefix = filename_prefix
        self.frame_digits = frame_digits

    def frame_path(self, frame: int) -> Path:
        """Generate frame path for given frame number."""
        filename = f"{self.filename_prefix}_{frame:0{self.frame_digits}d}.png"
        return self.output_dir / filename

    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class RenderProgress:
    """Progress information for render callbacks."""

    def __init__(self, frame: int, total_frames: int, time: float, output_path: Path):
        self.frame = frame
        self.total_frames = total_frames
        self.time = time
        self.output_path = output_path

    @property
    def percent(self) -> float:
        """Get progress as percentage (0.0 to 1.0)."""
        if self.total_frames == 0:
            return 0.0
        return self.frame / self.total_frames

    def __repr__(self) -> str:
        return f"RenderProgress({self.frame}/{self.total_frames}, {self.percent*100:.1f}%)"


__all__ = [
    "CameraAnimation",
    "CameraKeyframe",
    "CameraState",
    "RenderConfig",
    "RenderProgress",
]
