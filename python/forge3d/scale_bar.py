"""P4.2b: Scale bar generation with geodetic distance computation.

Renders scale bars with proper distance labels based on geographic bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .map_plate import BBox


@dataclass
class ScaleBarConfig:
    """Configuration for scale bar rendering."""
    units: Literal["m", "km", "mi", "ft"] = "km"
    style: Literal["simple", "alternating"] = "alternating"
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
    def compute_meters_per_pixel(bbox: BBox, image_width: int) -> float:
        """Compute meters per pixel from geographic bounds.
        
        Uses WGS84 approximation: 1 degree ≈ 111320m * cos(lat) for longitude.
        """
        lat_rad = math.radians(bbox.center_lat)
        meters_per_deg_lon = 111320.0 * math.cos(lat_rad)
        total_meters = bbox.width * meters_per_deg_lon
        return total_meters / image_width

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

    def render(self) -> np.ndarray:
        """Render the scale bar to an RGBA image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return self._render_simple()

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
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", cfg.font_size)
        except OSError:
            try:
                font = ImageFont.truetype("Arial.ttf", cfg.font_size)
            except OSError:
                font = ImageFont.load_default()
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0]
        label_height = bbox[3] - bbox[1]
        total_width = max(actual_bar_px + 2 * cfg.padding, label_width + 2 * cfg.padding)
        total_height = cfg.padding + cfg.bar_height + 4 + label_height + cfg.padding
        img = Image.new("RGBA", (total_width, total_height), cfg.background)
        draw = ImageDraw.Draw(img)
        bar_x = cfg.padding
        bar_y = cfg.padding
        if cfg.style == "alternating" and cfg.divisions > 1:
            div_width = actual_bar_px // cfg.divisions
            for i in range(cfg.divisions):
                x0 = bar_x + i * div_width
                x1 = x0 + div_width
                y0 = bar_y
                y1 = bar_y + cfg.bar_height
                color = cfg.bar_color_1 if i % 2 == 0 else cfg.bar_color_2
                draw.rectangle([x0, y0, x1, y1], fill=color)
            draw.rectangle(
                [bar_x, bar_y, bar_x + actual_bar_px, bar_y + cfg.bar_height],
                outline=cfg.border_color,
                width=1,
            )
        else:
            draw.rectangle(
                [bar_x, bar_y, bar_x + actual_bar_px, bar_y + cfg.bar_height],
                fill=cfg.bar_color_1,
                outline=cfg.border_color,
                width=1,
            )
        label_x = bar_x + (actual_bar_px - label_width) // 2
        label_y = bar_y + cfg.bar_height + 4
        draw.text((label_x, label_y), label, font=font, fill=cfg.label_color)
        return np.array(img, dtype=np.uint8)

    def _render_simple(self) -> np.ndarray:
        """Simple fallback rendering without PIL text."""
        cfg = self.config
        h, w = cfg.height_px, cfg.width_px
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[..., :] = cfg.background
        bar_w = w - 2 * cfg.padding
        bar_x = cfg.padding
        bar_y = cfg.padding
        bar_h = cfg.bar_height
        img[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w, :] = cfg.bar_color_1
        return img
