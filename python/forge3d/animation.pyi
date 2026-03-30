from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


class CameraKeyframe:
    @property
    def time(self) -> float: ...
    @property
    def phi_deg(self) -> float: ...
    @property
    def theta_deg(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def fov_deg(self) -> float: ...
    @property
    def target(self) -> Optional[Tuple[float, float, float]]: ...
    def __init__(
        self,
        time: float,
        phi: float,
        theta: float,
        radius: float,
        fov: float,
        target: Optional[Tuple[float, float, float]] = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...


class CameraState:
    @property
    def phi_deg(self) -> float: ...
    @property
    def theta_deg(self) -> float: ...
    @property
    def radius(self) -> float: ...
    @property
    def fov_deg(self) -> float: ...
    @property
    def target(self) -> Optional[Tuple[float, float, float]]: ...
    def __repr__(self) -> str: ...


class CameraAnimation:
    @property
    def duration(self) -> float: ...
    @property
    def keyframe_count(self) -> int: ...
    def __init__(self) -> None: ...
    def add_keyframe(
        self,
        time: float,
        phi: float,
        theta: float,
        radius: float,
        fov: float,
        target: Optional[Tuple[float, float, float]] = ...,
    ) -> None: ...
    def add_keyframe_py(
        self,
        time: float,
        phi: float,
        theta: float,
        radius: float,
        fov: float,
        target: Optional[Tuple[float, float, float]] = ...,
    ) -> None: ...
    def get_keyframes(self) -> list[CameraKeyframe]: ...
    def replace_keyframes(self, keyframes: list[CameraKeyframe]) -> None: ...
    def clear_keyframes(self) -> None: ...
    def get_frame_count(self, fps: int) -> int: ...
    def evaluate(self, time: float) -> Optional[CameraState]: ...
    def evaluate_py(self, time: float) -> Optional[CameraState]: ...
    def __repr__(self) -> str: ...


class RenderConfig:
    output_dir: Path
    fps: int
    width: int
    height: int
    filename_prefix: str
    frame_digits: int
    def __init__(
        self,
        output_dir: str = ...,
        fps: int = ...,
        width: int = ...,
        height: int = ...,
        filename_prefix: str = ...,
        frame_digits: int = ...,
    ) -> None: ...
    def frame_path(self, frame: int) -> Path: ...
    def ensure_output_dir(self) -> None: ...


class RenderProgress:
    frame: int
    total_frames: int
    time: float
    output_path: Path
    percent: float
    def __init__(self, frame: int, total_frames: int, time: float, output_path: Path) -> None: ...
    def __repr__(self) -> str: ...


__all__ = [
    "CameraAnimation",
    "CameraKeyframe",
    "CameraState",
    "RenderConfig",
    "RenderProgress",
]
