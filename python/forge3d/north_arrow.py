"""P4.2c: North Arrow / Compass indicator for map plates.

Renders a north arrow or compass rose indicator for cartographic output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


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

    def render(self) -> np.ndarray:
        """Render the north arrow to an RGBA image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return self._render_simple()

        cfg = self.config
        size = cfg.size
        padding = 8
        total_size = size + 2 * padding
        
        img = Image.new("RGBA", (total_size, total_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Draw background circle
        draw.ellipse(
            [padding // 2, padding // 2, total_size - padding // 2, total_size - padding // 2],
            fill=cfg.background,
            outline=cfg.border_color if cfg.border_width > 0 else None,
            width=cfg.border_width,
        )
        
        cx, cy = total_size // 2, total_size // 2
        rot_rad = math.radians(cfg.rotation_deg)
        
        if cfg.style == "arrow":
            self._draw_arrow(draw, cx, cy, size, rot_rad, cfg)
        elif cfg.style == "compass":
            self._draw_compass(draw, cx, cy, size, rot_rad, cfg)
        else:
            self._draw_simple(draw, cx, cy, size, rot_rad, cfg)
        
        # Draw "N" label
        if cfg.show_n_label:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", cfg.font_size)
            except OSError:
                try:
                    font = ImageFont.truetype("Arial Bold.ttf", cfg.font_size)
                except OSError:
                    try:
                        font = ImageFont.truetype("Arial.ttf", cfg.font_size)
                    except OSError:
                        font = ImageFont.load_default()
            
            # Position N at top of arrow
            n_offset = size // 2 - 2
            nx = cx + n_offset * math.sin(rot_rad)
            ny = cy - n_offset * math.cos(rot_rad)
            
            bbox = draw.textbbox((0, 0), "N", font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text((nx - tw // 2, ny - th // 2 - 2), "N", font=font, fill=cfg.color)
        
        return np.array(img, dtype=np.uint8)

    def _draw_arrow(self, draw, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
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
        
        # Draw filled north half (dark)
        draw.polygon([(tip_x, tip_y), (cx, cy), (bl_x, bl_y)], fill=cfg.color)
        
        # Draw outlined south half (light)
        draw.polygon([(tip_x, tip_y), (cx, cy), (br_x, br_y)], fill=cfg.background, outline=cfg.color)
        
        # Draw tail
        draw.line([(cx, cy), (tail_x, tail_y)], fill=cfg.color, width=2)

    def _draw_compass(self, draw, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
        """Draw compass rose style with N/S/E/W."""
        arrow_len = size // 2 - 12
        
        for i, (label, angle_offset) in enumerate([("N", 0), ("E", 90), ("S", 180), ("W", 270)]):
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
            
            # Draw arrow
            if label == "N":
                draw.polygon([(tip_x, tip_y), (bl_x, bl_y), (br_x, br_y)], fill=cfg.color)
            else:
                draw.polygon([(tip_x, tip_y), (bl_x, bl_y), (br_x, br_y)], fill=cfg.color, outline=cfg.color)

    def _draw_simple(self, draw, cx: int, cy: int, size: int, rot_rad: float, cfg: NorthArrowConfig):
        """Draw simple line arrow."""
        arrow_len = size // 2 - 8
        
        # Main line
        tip_x = cx + arrow_len * math.sin(rot_rad)
        tip_y = cy - arrow_len * math.cos(rot_rad)
        tail_x = cx - arrow_len * 0.5 * math.sin(rot_rad)
        tail_y = cy + arrow_len * 0.5 * math.cos(rot_rad)
        
        draw.line([(tail_x, tail_y), (tip_x, tip_y)], fill=cfg.color, width=3)
        
        # Arrowhead
        head_len = 10
        head_angle = math.pi / 6
        lx = tip_x - head_len * math.sin(rot_rad - head_angle)
        ly = tip_y + head_len * math.cos(rot_rad - head_angle)
        rx = tip_x - head_len * math.sin(rot_rad + head_angle)
        ry = tip_y + head_len * math.cos(rot_rad + head_angle)
        
        draw.polygon([(tip_x, tip_y), (lx, ly), (rx, ry)], fill=cfg.color)

    def _render_simple(self) -> np.ndarray:
        """Simple fallback rendering without PIL."""
        cfg = self.config
        size = cfg.size + 16
        img = np.zeros((size, size, 4), dtype=np.uint8)
        img[..., :] = cfg.background
        
        # Draw a simple triangle pointing up
        cx, cy = size // 2, size // 2
        for y in range(size):
            for x in range(size):
                dx, dy = x - cx, y - cy
                if dy < 0 and abs(dx) < -dy // 2:
                    img[y, x, :] = cfg.color
        
        return img
