"""Jupyter widgets for viewer control and notebook-friendly terrain previews."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .helpers.offscreen import rgba_to_png_bytes

__all__ = ["ViewerWidget", "widgets_available"]


def _missing_ipywidgets_message() -> str:
    return "Jupyter widgets require ipywidgets. Install with: pip install forge3d[jupyter]"


try:
    import ipywidgets as _widgets
except ImportError as _WIDGETS_IMPORT_ERROR:  # pragma: no cover - optional dependency
    _widgets = None
else:  # pragma: no branch
    _WIDGETS_IMPORT_ERROR = None


def widgets_available() -> bool:
    """Return ``True`` when the optional ipywidgets dependency is installed."""
    return _widgets is not None


def _load_height_array(src: Any) -> np.ndarray:
    if isinstance(src, np.ndarray):
        data = np.asarray(src, dtype=np.float32)
    else:
        path = Path(src)
        suffix = path.suffix.lower()
        if suffix == ".npy":
            data = np.asarray(np.load(path), dtype=np.float32)
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError(
                    "Loading terrain previews from image paths requires Pillow. "
                    "Install with: pip install pillow"
                ) from exc
            data = np.asarray(Image.open(path), dtype=np.float32)
            if data.ndim == 3:
                data = data[..., 0]

    if data.ndim != 2:
        raise ValueError("Inline preview source must resolve to a 2D height array")
    if data.size == 0:
        raise ValueError("Inline preview source must not be empty")

    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros_like(data, dtype=np.float32)

    fill_value = float(np.nanmin(data[finite]))
    return np.where(finite, data, fill_value).astype(np.float32)


def _palette_rgb(name: Optional[str]) -> np.ndarray:
    palette_name = str(name or "terrain")
    try:
        from .colormaps import get as get_colormap

        cmap = get_colormap(palette_name)
        rgb = np.asarray(cmap.rgba[:, :3], dtype=np.float32)
        return np.clip(rgb, 0.0, 1.0) ** (1.0 / 2.2)
    except Exception:
        stops = np.array(
            [
                [0.11, 0.16, 0.23],
                [0.19, 0.36, 0.28],
                [0.54, 0.50, 0.33],
                [0.74, 0.67, 0.46],
                [0.92, 0.91, 0.88],
            ],
            dtype=np.float32,
        )
        x = np.linspace(0.0, 1.0, stops.shape[0], dtype=np.float32)
        t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        return np.column_stack([np.interp(t, x, stops[:, i]) for i in range(3)]).astype(np.float32)


def _sample_palette(rgb_lut: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.clip(values * float(rgb_lut.shape[0] - 1), 0.0, float(rgb_lut.shape[0] - 1))
    lo = np.floor(idx).astype(np.int32)
    hi = np.clip(lo + 1, 0, rgb_lut.shape[0] - 1)
    frac = (idx - lo)[..., None]
    return rgb_lut[lo] * (1.0 - frac) + rgb_lut[hi] * frac


def _hillshade(heightmap: np.ndarray, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    dy, dx = np.gradient(heightmap.astype(np.float32))
    slope_x = -dx
    slope_y = -dy

    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    light = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        ],
        dtype=np.float32,
    )

    normal = np.dstack((-slope_x, np.ones_like(heightmap, dtype=np.float32), -slope_y))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True) + 1e-8
    shade = normal @ light
    return np.clip(0.18 + 0.82 * shade, 0.0, 1.0)


def _rotate_heightmap(heightmap: np.ndarray, phi_deg: float) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        return heightmap

    finite = np.isfinite(heightmap)
    min_value = float(heightmap[finite].min()) if finite.any() else 0.0
    max_value = float(heightmap[finite].max()) if finite.any() else 1.0
    span = max(max_value - min_value, 1e-6)
    normalized = ((heightmap - min_value) / span * 65535.0).astype(np.uint16)
    image = Image.fromarray(normalized, mode="I;16")
    rotated = image.rotate(-float(phi_deg), resample=Image.BILINEAR, expand=False)
    rotated_array = np.asarray(rotated, dtype=np.float32) / 65535.0
    return rotated_array * span + min_value


def _render_dem_preview(
    src: Any,
    *,
    width: int,
    height: int,
    camera_phi: float = 35.0,
    camera_theta: float = 55.0,
    camera_distance: float = 1.2,
    lighting_azimuth: float = 315.0,
    lighting_elevation: float = 35.0,
    palette: Optional[str] = "terrain",
    **_: Any,
) -> np.ndarray:
    """Render a lightweight pseudo-3D terrain preview from a height array."""
    terrain = _load_height_array(src)
    terrain = _rotate_heightmap(terrain, camera_phi)

    h_src, w_src = terrain.shape
    max_dim = max(h_src, w_src)
    if max_dim > 320:
        step = int(np.ceil(max_dim / 320.0))
        terrain = terrain[::step, ::step]

    finite = np.isfinite(terrain)
    min_value = float(terrain[finite].min()) if finite.any() else 0.0
    max_value = float(terrain[finite].max()) if finite.any() else 1.0
    span = max(max_value - min_value, 1e-6)
    normalized = np.clip((terrain - min_value) / span, 0.0, 1.0)

    shade = _hillshade(normalized, lighting_azimuth, lighting_elevation)
    palette_rgb = _sample_palette(_palette_rgb(palette), normalized)

    canvas = np.zeros((int(height), int(width), 4), dtype=np.uint8)
    sky_top = np.array([143, 186, 233], dtype=np.float32)
    sky_bottom = np.array([236, 245, 255], dtype=np.float32)
    blend = np.linspace(0.0, 1.0, int(height), dtype=np.float32)[:, None]
    canvas[..., :3] = (sky_top * (1.0 - blend) + sky_bottom * blend).astype(np.uint8)[:, None, :]
    canvas[..., 3] = 255

    rows, cols = normalized.shape
    x_samples = np.linspace(0.0, cols - 1.0, int(width), dtype=np.float32)
    hidden_y = np.full(int(width), int(height), dtype=np.int32)

    theta = np.clip(float(camera_theta), 10.0, 85.0)
    distance = max(float(camera_distance), 0.2)
    relief_scale = (height * 0.55) * np.sin(np.deg2rad(theta)) / (0.8 + distance)
    depth_scale = (height * 0.68) / max(rows - 1, 1)
    horizon = height * 0.20 + (85.0 - theta) * 0.8

    for row in range(rows):
        row_color = palette_rgb[row]
        row_shade = shade[row]
        row_height = normalized[row]

        color_interp = np.column_stack(
            [np.interp(x_samples, np.arange(cols, dtype=np.float32), row_color[:, channel]) for channel in range(3)]
        )
        shade_interp = np.interp(x_samples, np.arange(cols, dtype=np.float32), row_shade)
        height_interp = np.interp(x_samples, np.arange(cols, dtype=np.float32), row_height)

        lit_rgb = np.clip(color_interp * (0.35 + 0.75 * shade_interp[:, None]), 0.0, 1.0)
        y_line = horizon + row * depth_scale - height_interp * relief_scale

        for sx in range(int(width)):
            sy = int(np.clip(y_line[sx], 0, height - 1))
            if sy < hidden_y[sx]:
                canvas[sy:hidden_y[sx], sx, :3] = (lit_rgb[sx] * 255.0).astype(np.uint8)
                hidden_y[sx] = sy

    return canvas


if _widgets is None:  # pragma: no cover - only executed without ipywidgets
    class _InlinePreview:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_missing_ipywidgets_message()) from _WIDGETS_IMPORT_ERROR

    class ViewerWidget:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(_missing_ipywidgets_message()) from _WIDGETS_IMPORT_ERROR

else:
    class _InlinePreview(_widgets.VBox):  # type: ignore[misc]
        """Internal inline fallback preview used when the live viewer is unavailable."""

        def __init__(
            self,
            src: Any,
            *,
            render_fn: Optional[Any] = None,
            width: int = 800,
            height: int = 600,
            auto_render: bool = True,
            **render_kwargs: Any,
        ) -> None:
            self._src = src
            self._render_fn = render_fn or _render_dem_preview
            self._width = int(width)
            self._height = int(height)
            self._render_kwargs = dict(render_kwargs)
            self._last_image: Optional[np.ndarray] = None
            self._last_error: Optional[Exception] = None
            self._render_inflight = False
            self._suspend_observers = False

            self.camera_phi = _widgets.FloatSlider(
                value=float(self._render_kwargs.pop("camera_phi", 35.0)),
                min=0.0,
                max=360.0,
                step=1.0,
                description="Camera φ",
                continuous_update=False,
            )
            self.camera_theta = _widgets.FloatSlider(
                value=float(self._render_kwargs.pop("camera_theta", 55.0)),
                min=10.0,
                max=85.0,
                step=1.0,
                description="Camera θ",
                continuous_update=False,
            )
            self.camera_distance = _widgets.FloatSlider(
                value=float(self._render_kwargs.pop("camera_distance", 1.2)),
                min=0.2,
                max=3.0,
                step=0.05,
                description="Distance",
                continuous_update=False,
            )
            self.lighting_azimuth = _widgets.FloatSlider(
                value=float(self._render_kwargs.pop("lighting_azimuth", 315.0)),
                min=0.0,
                max=360.0,
                step=1.0,
                description="Sun Az",
                continuous_update=False,
            )
            self.lighting_elevation = _widgets.FloatSlider(
                value=float(self._render_kwargs.pop("lighting_elevation", 35.0)),
                min=5.0,
                max=85.0,
                step=1.0,
                description="Sun El",
                continuous_update=False,
            )
            self.palette = _widgets.Text(
                value=str(self._render_kwargs.pop("palette", "terrain")),
                description="Palette",
            )
            self._status = _widgets.HTML(value="<b>Render:</b> idle")
            self._image = _widgets.Image(format="png")
            self._image.layout.width = f"{self._width}px"
            self._image.layout.height = f"{self._height}px"

            for control in (
                self.camera_phi,
                self.camera_theta,
                self.camera_distance,
                self.lighting_azimuth,
                self.lighting_elevation,
                self.palette,
            ):
                control.observe(self._on_control_change, names="value")

            controls = _widgets.Tab(
                children=[
                    _widgets.VBox([self.camera_phi, self.camera_theta, self.camera_distance]),
                    _widgets.VBox([self.lighting_azimuth, self.lighting_elevation, self.palette]),
                ]
            )
            controls.set_title(0, "Camera")
            controls.set_title(1, "Lighting")

            super().__init__(children=[self._status, self._image, controls])

            if auto_render:
                self.request_render()

        def get_params(self) -> Dict[str, Any]:
            """Return the current render parameters."""
            params = {
                "width": self._width,
                "height": self._height,
                "camera_phi": float(self.camera_phi.value),
                "camera_theta": float(self.camera_theta.value),
                "camera_distance": float(self.camera_distance.value),
                "lighting_azimuth": float(self.lighting_azimuth.value),
                "lighting_elevation": float(self.lighting_elevation.value),
                "palette": self.palette.value or "terrain",
            }
            params.update(self._render_kwargs)
            return params

        def update_params(self, **kwargs: Any) -> None:
            """Update persistent render parameters and re-render."""
            self._suspend_observers = True
            try:
                if "camera_phi" in kwargs:
                    self.camera_phi.value = float(kwargs.pop("camera_phi"))
                if "camera_theta" in kwargs:
                    self.camera_theta.value = float(kwargs.pop("camera_theta"))
                if "camera_distance" in kwargs:
                    self.camera_distance.value = float(kwargs.pop("camera_distance"))
                if "lighting_azimuth" in kwargs:
                    self.lighting_azimuth.value = float(kwargs.pop("lighting_azimuth"))
                if "lighting_elevation" in kwargs:
                    self.lighting_elevation.value = float(kwargs.pop("lighting_elevation"))
                if "palette" in kwargs:
                    self.palette.value = str(kwargs.pop("palette"))
                self._render_kwargs.update(kwargs)
            finally:
                self._suspend_observers = False
            self.request_render()

        def request_render(self) -> None:
            """Render immediately using the latest parameters."""
            self._render_inflight = True
            self._status.value = "<b>Render:</b> rendering"
            try:
                image = self._render_fn(self._src, **self.get_params())
                self._last_image = np.asarray(image, dtype=np.uint8)
                self._image.value = rgba_to_png_bytes(self._last_image)
                self._last_error = None
                self._status.value = "<b>Render:</b> ready"
            except Exception as exc:
                self._last_error = exc
                self._status.value = f"<b>Render:</b> failed ({exc})"
                raise
            finally:
                self._render_inflight = False

        def wait_for_idle(self, timeout: float = 2.0) -> bool:
            """Return ``True`` when no render is currently inflight."""
            del timeout
            return not self._render_inflight

        def render_png_bytes(self) -> bytes:
            """Render now and return PNG bytes for the latest preview."""
            if self._last_image is None:
                self.request_render()
            if self._last_image is None:
                raise RuntimeError("No rendered image is available")
            return rgba_to_png_bytes(self._last_image)

        def _on_control_change(self, change: Any) -> None:
            del change
            if self._suspend_observers:
                return
            self.request_render()


    class ViewerWidget(_widgets.VBox):  # type: ignore[misc]
        """Notebook-friendly wrapper around the interactive viewer subprocess."""

        def __init__(
            self,
            terrain_path: Optional[Any] = None,
            *,
            src: Optional[Any] = None,
            width: int = 1280,
            height: int = 720,
            auto_launch: bool = True,
            fallback_to_render: bool = True,
            render_fn: Optional[Any] = None,
            **render_kwargs: Any,
        ) -> None:
            self._width = int(width)
            self._height = int(height)
            self._handle = None
            self._fallback: Optional[_InlinePreview] = None
            self._temp_paths: list[Path] = []
            self._status = _widgets.HTML(value="<b>Viewer:</b> idle")
            self._snapshot_image = _widgets.Image(format="png")
            self._snapshot_image.layout.width = "100%"

            children = [self._status, self._snapshot_image]

            if auto_launch:
                try:
                    from .viewer import open_viewer_async

                    self._handle = open_viewer_async(
                        terrain_path=terrain_path,
                        width=self._width,
                        height=self._height,
                    )
                    self._status.value = f"<b>Viewer:</b> running on port {self._handle.port}"
                    try:
                        self.snapshot(width=self._width, height=self._height)
                    except Exception:
                        pass
                except Exception as exc:
                    fallback_src = src if src is not None else terrain_path
                    if fallback_to_render and fallback_src is not None:
                        self._fallback = _InlinePreview(
                            src=fallback_src,
                            render_fn=render_fn,
                            width=self._width,
                            height=self._height,
                            **render_kwargs,
                        )
                        children = [self._status, self._fallback]
                        self._status.value = (
                            f"<b>Viewer:</b> unavailable ({exc}); using inline preview fallback"
                        )
                    else:
                        self._status.value = f"<b>Viewer:</b> unavailable ({exc})"

            super().__init__(children=children)

        def send_ipc(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
            """Send a raw IPC command to the running viewer."""
            if self._handle is None:
                raise RuntimeError("Viewer process is not running")
            return self._handle.send_ipc(cmd)

        def set_camera(self, phi_deg: float, theta_deg: float, radius: float, fov_deg: Optional[float] = None) -> None:
            """Set the orbit camera on the running viewer."""
            if self._handle is None:
                raise RuntimeError("Viewer process is not running")
            self._handle.set_orbit_camera(phi_deg=phi_deg, theta_deg=theta_deg, radius=radius, fov_deg=fov_deg)

        def set_sun(self, azimuth_deg: float, elevation_deg: float) -> None:
            """Set the sun direction on the running viewer."""
            if self._handle is None:
                raise RuntimeError("Viewer process is not running")
            self._handle.set_sun(azimuth_deg=azimuth_deg, elevation_deg=elevation_deg)

        def snapshot(
            self,
            path: Optional[Any] = None,
            *,
            width: Optional[int] = None,
            height: Optional[int] = None,
        ) -> Optional[Path]:
            """Capture a snapshot and show it inline in the notebook output."""
            output_path: Optional[Path] = Path(path) if path is not None else None
            snapshot_width = int(width) if width is not None else self._width
            snapshot_height = int(height) if height is not None else self._height

            if self._handle is not None:
                if output_path is None:
                    handle, temp_path = tempfile.mkstemp(prefix="forge3d_widget_", suffix=".png")
                    Path(temp_path).unlink(missing_ok=True)
                    output_path = Path(temp_path)
                    self._temp_paths.append(output_path)
                    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(temp_path).touch()
                    os_path = str(output_path)
                else:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    os_path = str(output_path)

                self._handle.snapshot(os_path, width=snapshot_width, height=snapshot_height)
                self._snapshot_image.value = output_path.read_bytes()
                self._status.value = f"<b>Viewer:</b> snapshot saved to {output_path.name}"
                return output_path

            if self._fallback is not None:
                png_bytes = self._fallback.render_png_bytes()
                self._snapshot_image.value = png_bytes
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(png_bytes)
                self._status.value = "<b>Viewer:</b> snapshot generated from inline preview fallback"
                return output_path

            raise RuntimeError("Viewer is not available")

        def close(self) -> None:
            """Close the running viewer subprocess and clean up temporary snapshots."""
            if self._handle is not None:
                self._handle.close()
                self._handle = None
            for path in self._temp_paths:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._temp_paths.clear()
            self._status.value = "<b>Viewer:</b> closed"
            super().close()
