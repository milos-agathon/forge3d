"""P4.2b: Scale bar generation with geodetic distance computation.

Renders scale bars with proper distance labels based on geographic bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .map_plate import BBox
from .certificate import _captured_cpu_render


@dataclass
class ScaleBarConfig:
    """Configuration for scale bar rendering."""
    units: Literal["m", "km", "mi", "ft"] = "km"
    style: Literal["simple", "alternating"] = "alternating"
    geodesic: bool = True
    width_px: int = 180
    height_px: int = 25
    divisions: int = 4
    font_size: int = 12
    padding: int = 8
    bar_height: int = 8
    background: tuple[int, int, int, int] = (255, 255, 255, 200)
    bar_color_1: tuple[int, int, int, int] = (0, 0, 0, 255)
    bar_color_2: tuple[int, int, int, int] = (255, 255, 255, 255)
    label_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    border_color: tuple[int, int, int, int] = (0, 0, 0, 255)


UNIT_FACTORS = {
    "m": 1.0,
    "km": 0.001,
    "mi": 0.000621371,
    "ft": 3.28084,
}

UNIT_LABELS = {
    "m": "m",
    "km": "km",
    "mi": "mi",
    "ft": "ft",
}

NICE_INTERVALS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000]


class ScaleBar:
    """Generates scale bar images with geodetic distance."""

    def __init__(self, meters_per_pixel: float, config: ScaleBarConfig | None = None):
        """
        Args:
            meters_per_pixel: Ground distance per pixel
            config: Scale bar configuration
        """
        self.meters_per_pixel = meters_per_pixel
        self.config = config or ScaleBarConfig()

    @staticmethod
    def compute_meters_per_pixel(bbox: BBox, image_width: int, *, geodesic: bool = True) -> float:
        """Compute meters per pixel from geographic bounds.

        Uses WGS84 geodesic distance along the bounding-box center latitude by
        default, falling back to a cosine approximation when a geodesic backend
        is unavailable.
        """
        if image_width <= 0:
            raise ValueError("image_width must be positive")
        if geodesic:
            endpoints = ((float(bbox.west), float(bbox.center_lat)), (float(bbox.east), float(bbox.center_lat)))
            if str(getattr(bbox, "crs", "EPSG:4326")).upper() not in {"EPSG:4326", "WGS84", "WGS 84"}:
                try:
                    import numpy as np

                    from .crs import transform_coords

                    transformed = transform_coords(np.asarray(endpoints, dtype=np.float64), str(bbox.crs), "EPSG:4326")
                    endpoints = ((float(transformed[0, 0]), float(transformed[0, 1])), (float(transformed[1, 0]), float(transformed[1, 1])))
                except Exception:
                    return abs(float(bbox.width)) / float(image_width)
            try:
                from pyproj import Geod

                geod = Geod(ellps="WGS84")
                _az12, _az21, distance = geod.inv(endpoints[0][0], endpoints[0][1], endpoints[1][0], endpoints[1][1])
                return abs(float(distance)) / float(image_width)
            except Exception:
                pass
        lat_rad = math.radians(bbox.center_lat)
        meters_per_deg_lon = 111320.0 * math.cos(lat_rad)
        total_meters = bbox.width * meters_per_deg_lon
        return abs(total_meters) / image_width

    def _choose_nice_distance(self, max_distance: float) -> float:
        """Choose a nice round distance that fits within max_distance."""
        cfg = self.config
        unit_factor = UNIT_FACTORS[cfg.units]
        max_display = max_distance * unit_factor
        nice = NICE_INTERVALS[0]
        for interval in NICE_INTERVALS:
            if interval <= max_display:
                nice = interval
            else:
                break
        return nice / unit_factor

    @_captured_cpu_render("python.scale_bar.render", "scale_bar.cpu", draw_calls=1)
    def render(self, *, certificate: bool | str = False) -> np.ndarray:
        """Render geometry and its label through the shared native text engine."""
        from ._map_scene_render import _draw_text

        image, label, anchor = self.render_geometry()
        cfg = self.config
        _draw_text(
            image,
            label,
            anchor,
            color=cfg.label_color,
            halo=(0, 0, 0, 0),
            halo_width_px=0.0,
            font_size=float(cfg.font_size),
        )
        return image

    def render_geometry(self) -> tuple[np.ndarray, str, tuple[int, int]]:
        """Return deterministic bar geometry plus native-label placement metadata."""
        from ._map_scene_render import _text_outline_metrics

        cfg = self.config
        bar_width = cfg.width_px - 2 * cfg.padding
        max_distance = bar_width * self.meters_per_pixel
        nice_distance = self._choose_nice_distance(max_distance)
        actual_bar_px = int(nice_distance / self.meters_per_pixel)
        actual_bar_px = min(actual_bar_px, bar_width)
        unit_factor = UNIT_FACTORS[cfg.units]
        unit_label = UNIT_LABELS[cfg.units]
        display_distance = nice_distance * unit_factor
        if display_distance >= 1:
            label = f"{int(display_distance)} {unit_label}"
        else:
            label = f"{display_distance:.1f} {unit_label}"
        label_width, outline_height, bounds = _text_outline_metrics(label, float(cfg.font_size))
        label_height = max(outline_height, max(1, int(math.ceil(cfg.font_size * 1.25))))
        total_width = max(actual_bar_px + 2 * cfg.padding, label_width + 2 * cfg.padding)
        total_height = cfg.padding + cfg.bar_height + 4 + label_height + cfg.padding
        image = np.empty((total_height, total_width, 4), dtype=np.uint8)
        image[...] = cfg.background
        bar_x = cfg.padding
        bar_y = cfg.padding
        if cfg.style == "alternating" and cfg.divisions > 1:
            edges = np.linspace(bar_x, bar_x + actual_bar_px, cfg.divisions + 1, dtype=int)
            for i in range(cfg.divisions):
                color = cfg.bar_color_1 if i % 2 == 0 else cfg.bar_color_2
                image[bar_y : bar_y + cfg.bar_height, edges[i] : edges[i + 1]] = color
        else:
            image[
                bar_y : bar_y + cfg.bar_height,
                bar_x : bar_x + actual_bar_px,
            ] = cfg.bar_color_1
        x1 = bar_x + actual_bar_px
        y1 = bar_y + cfg.bar_height
        image[bar_y:y1, bar_x] = cfg.border_color
        image[bar_y:y1, max(bar_x, x1 - 1)] = cfg.border_color
        image[bar_y, bar_x:x1] = cfg.border_color
        image[max(bar_y, y1 - 1), bar_x:x1] = cfg.border_color
        label_x = max(0, bar_x + (actual_bar_px - label_width) // 2)
        if bounds is not None:
            label_x = max(0, int(round(label_x - float(bounds[0]))))
        label_y = bar_y + cfg.bar_height + 4
        return image, label, (label_x, label_y)
