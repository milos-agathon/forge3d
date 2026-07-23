"""P4.2c: North Arrow / Compass indicator for map plates.

Renders a north arrow or compass rose indicator for cartographic output.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .certificate import _captured_cpu_render


@dataclass
class NorthArrowConfig:
    """Configuration for north arrow rendering."""
    style: Literal["arrow", "compass", "simple"] = "arrow"
    size: int = 60
    rotation_deg: float = 0.0  # Rotation from true north (for rotated maps)
    color: tuple[int, int, int, int] = (0, 0, 0, 255)
    background: tuple[int, int, int, int] = (255, 255, 255, 200)
    show_n_label: bool = True
    font_size: int = 14
    border_width: int = 1
    border_color: tuple[int, int, int, int] = (0, 0, 0, 255)


class NorthArrow:
    """Generates north arrow images for map plates."""

    def __init__(self, config: NorthArrowConfig | None = None):
        self.config = config or NorthArrowConfig()

    @_captured_cpu_render("python.north_arrow.render", "north_arrow.cpu", draw_calls=1)
    def render(self, *, certificate: bool | str = False, cache: str | None = None) -> np.ndarray:
        """Render arrow geometry and the N label through native shaping."""
        _ = cache
        from ._map_scene_render import _draw_text

        image, label, anchor = self.render_geometry()
        cfg = self.config
        if label is not None and anchor is not None:
            _draw_text(
                image,
                label,
                anchor,
                color=cfg.color,
                halo=(0, 0, 0, 0),
                halo_width_px=0.0,
                font_size=float(cfg.font_size),
            )
        return image

    def render_geometry(
        self, *, cache: str | os.PathLike[str] | None = None
    ) -> tuple[np.ndarray, str | None, tuple[int, int] | None]:
        """Return deterministic numpy geometry plus optional native-label metadata."""
        _ = cache  # deterministic CPU helper; accepted for render-surface consistency
        cfg = self.config
        size = cfg.size
        padding = 8
        total_size = size + 2 * padding
        image = np.zeros((total_size, total_size, 4), dtype=np.uint8)
        cx, cy = total_size // 2, total_size // 2
        yy, xx = np.mgrid[:total_size, :total_size]
        radius = (total_size - padding) * 0.5
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        image[distance <= radius] = cfg.background
        if cfg.border_width > 0:
            border = (distance <= radius) & (distance >= radius - cfg.border_width)
            image[border] = cfg.border_color
        rot_rad = math.radians(cfg.rotation_deg)
        if cfg.style == "arrow":
            self._draw_arrow(image, cx, cy, size, rot_rad, cfg)
        elif cfg.style == "compass":
            self._draw_compass(image, cx, cy, size, rot_rad, cfg)
        else:
            self._draw_simple(image, cx, cy, size, rot_rad, cfg)
        if cfg.show_n_label:
            from ._map_scene_render import _text_anchor_for_visual_center, _text_outline_metrics

            n_offset = size // 2 - 2
            nx = cx + n_offset * math.sin(rot_rad)
            ny = cy - n_offset * math.cos(rot_rad)
            _label_width, _label_height, bounds = _text_outline_metrics("N", float(cfg.font_size))
            anchor = (
                _text_anchor_for_visual_center(nx, ny, float(cfg.font_size), bounds)
                if bounds is not None
                else (int(round(nx)), int(round(ny)))
            )
            return image, "N", anchor
        return image, None, None

    @staticmethod
    def _polygon(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int, int]) -> None:
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        inside = np.zeros(xx.shape, dtype=bool)
        previous = points[-1]
        for current in points:
            x0, y0 = previous
            x1, y1 = current
            crossing = ((y0 > yy) != (y1 > yy)) & (
                xx < (x1 - x0) * (yy - y0) / ((y1 - y0) + 1e-12) + x0
            )
            inside ^= crossing
            previous = current
        image[inside] = color

    @staticmethod
    def _line(image: np.ndarray, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int, int], width: int) -> None:
        yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        denominator = max(dx * dx + dy * dy, 1e-12)
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / denominator, 0.0, 1.0)
        distance = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        image[distance <= max(0.5, width * 0.5)] = color

    def _draw_arrow(self, image: np.ndarray, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
        """Draw classic north arrow style."""
        arrow_len = size // 2 - 8
        arrow_width = size // 6
        
        # Arrow points
        # Tip (north)
        tip_x = cx + arrow_len * math.sin(rot_rad)
        tip_y = cy - arrow_len * math.cos(rot_rad)
        
        # Base left
        base_angle_l = rot_rad + math.pi + math.atan2(arrow_width, arrow_len)
        base_dist = math.sqrt(arrow_len**2 + arrow_width**2) * 0.4
        bl_x = cx + base_dist * math.sin(base_angle_l)
        bl_y = cy - base_dist * math.cos(base_angle_l)
        
        # Base right
        base_angle_r = rot_rad + math.pi - math.atan2(arrow_width, arrow_len)
        br_x = cx + base_dist * math.sin(base_angle_r)
        br_y = cy - base_dist * math.cos(base_angle_r)
        
        # Tail (south)
        tail_x = cx - arrow_len * 0.6 * math.sin(rot_rad)
        tail_y = cy + arrow_len * 0.6 * math.cos(rot_rad)
        
        self._polygon(image, [(tip_x, tip_y), (cx, cy), (bl_x, bl_y)], cfg.color)
        self._polygon(image, [(tip_x, tip_y), (cx, cy), (br_x, br_y)], cfg.background)
        self._line(image, (tip_x, tip_y), (br_x, br_y), cfg.color, 1)
        self._line(image, (br_x, br_y), (cx, cy), cfg.color, 1)
        self._line(image, (cx, cy), (tip_x, tip_y), cfg.color, 1)
        self._line(image, (cx, cy), (tail_x, tail_y), cfg.color, 2)

    def _draw_compass(self, image: np.ndarray, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
        """Draw compass rose style with N/S/E/W."""
        arrow_len = size // 2 - 12
        
        for _i, (label, angle_offset) in enumerate([("N", 0), ("E", 90), ("S", 180), ("W", 270)]):
            angle = rot_rad + math.radians(angle_offset)
            
            # Arrow tip
            tip_x = cx + arrow_len * math.sin(angle)
            tip_y = cy - arrow_len * math.cos(angle)
            
            # Arrow base points
            base_offset = 6
            perp_angle = angle + math.pi / 2
            bl_x = cx + base_offset * math.sin(perp_angle)
            bl_y = cy - base_offset * math.cos(perp_angle)
            br_x = cx - base_offset * math.sin(perp_angle)
            br_y = cy + base_offset * math.cos(perp_angle)
            
            self._polygon(image, [(tip_x, tip_y), (bl_x, bl_y), (br_x, br_y)], cfg.color)

    def _draw_simple(self, image: np.ndarray, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
        """Draw simple line arrow."""
        arrow_len = size // 2 - 8
        
        # Main line
        tip_x = cx + arrow_len * math.sin(rot_rad)
        tip_y = cy - arrow_len * math.cos(rot_rad)
        tail_x = cx - arrow_len * 0.5 * math.sin(rot_rad)
        tail_y = cy + arrow_len * 0.5 * math.cos(rot_rad)
        
        self._line(image, (tail_x, tail_y), (tip_x, tip_y), cfg.color, 3)
        
        # Arrowhead
        head_len = 10
        head_angle = math.pi / 6
        lx = tip_x - head_len * math.sin(rot_rad - head_angle)
        ly = tip_y + head_len * math.cos(rot_rad - head_angle)
        rx = tip_x - head_len * math.sin(rot_rad + head_angle)
        ry = tip_y + head_len * math.cos(rot_rad + head_angle)
        
        self._polygon(image, [(tip_x, tip_y), (lx, ly), (rx, ry)], cfg.color)
