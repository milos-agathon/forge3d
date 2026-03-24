from pathlib import Path
from typing import Any, Callable, Optional, Sequence


class ViewerError(Exception): ...


class ViewerHandle:
    def __init__(
        self,
        process: Any,
        host: str,
        port: int,
        timeout: float = ...,
        cleanup_paths: Optional[list[Path]] = ...,
    ) -> None: ...
    def send_ipc(self, cmd: dict[str, Any]) -> dict[str, Any]: ...
    def get_stats(self) -> dict[str, Any]: ...
    def get_terrain_volumetrics_report(self) -> dict[str, Any]: ...
    def load_obj(self, path: str | Path) -> None: ...
    def load_gltf(self, path: str | Path) -> None: ...
    def load_terrain(self, path: str | Path) -> None: ...
    def load_overlay(
        self,
        name: str,
        path: str | Path,
        extent: Optional[tuple[float, float, float, float]] = ...,
        opacity: Optional[float] = ...,
        z_order: Optional[int] = ...,
    ) -> None: ...
    def load_point_cloud(
        self,
        path: str | Path,
        point_size: float = ...,
        max_points: int = ...,
        color_mode: Optional[str] = ...,
    ) -> None: ...
    def set_point_cloud_params(
        self,
        point_size: Optional[float] = ...,
        visible: Optional[bool] = ...,
        color_mode: Optional[str] = ...,
    ) -> None: ...
    def set_transform(
        self,
        translation: Optional[tuple[float, float, float]] = ...,
        rotation_quat: Optional[tuple[float, float, float, float]] = ...,
        scale: Optional[tuple[float, float, float]] = ...,
    ) -> None: ...
    def set_camera_lookat(
        self,
        eye: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float] = ...,
    ) -> None: ...
    def set_fov(self, deg: float) -> None: ...
    def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None: ...
    def set_ibl(self, path: str | Path, intensity: float = ...) -> None: ...
    def set_z_scale(self, value: float) -> None: ...
    def set_terrain_scatter(self, batches: list[dict[str, Any]]) -> None: ...
    def clear_terrain_scatter(self) -> None: ...
    def set_orbit_camera(
        self,
        phi_deg: float,
        theta_deg: float,
        radius: float,
        fov_deg: Optional[float] = ...,
    ) -> None: ...
    def snapshot(
        self,
        path: str | Path,
        width: Optional[int] = ...,
        height: Optional[int] = ...,
    ) -> None: ...
    def render_animation(
        self,
        animation: Any,
        output_dir: str | Path,
        fps: int = ...,
        width: Optional[int] = ...,
        height: Optional[int] = ...,
        progress_callback: Optional[Callable[[int, int], None]] = ...,
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "ViewerHandle": ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
    @property
    def port(self) -> int: ...
    @property
    def is_running(self) -> bool: ...


def set_msaa(samples: int) -> int: ...


def open_viewer_async(
    width: int = ...,
    height: int = ...,
    title: str = ...,
    obj_path: Optional[str | Path] = ...,
    gltf_path: Optional[str | Path] = ...,
    terrain_path: Optional[str | Path] = ...,
    fov_deg: float = ...,
    timeout: float = ...,
    ipc_host: str = ...,
    ipc_port: int = ...,
) -> ViewerHandle: ...


def open_viewer(
    width: int = ...,
    height: int = ...,
    title: str = ...,
    obj_path: Optional[str | Path] = ...,
    gltf_path: Optional[str | Path] = ...,
    fov_deg: float = ...,
    snapshot_path: Optional[str] = ...,
    snapshot_width: Optional[int] = ...,
    snapshot_height: Optional[int] = ...,
    initial_commands: Optional[Sequence[str]] = ...,
) -> int: ...
