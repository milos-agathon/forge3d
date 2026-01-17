"""P4.2a: Legend generation from colormaps.

Renders elevation/value legends with gradient bars, tick marks, and labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class LegendConfig:
    """Configuration for legend rendering."""
    orientation: Literal["vertical", "horizontal"] = "vertical"
    bar_width: int = 25
    bar_height: int = 250
    tick_count: int = 5
    tick_length: int = 8
    label_format: str = "{:.0f}"
    label_suffix: str = " m"
    font_size: int = 14
    title: str | None = None
    title_font_size: int = 16
    padding: int = 10
    background: tuple[int, int, int, int] = (255, 255, 255, 0)
    tick_color: tuple[int, int, int, int] = (0, 0, 0, 255)
    label_color: tuple[int, int, int, int] = (0, 0, 0, 255)


class Legend:
    """Generates legend images from colormaps."""

    def __init__(
        self,
        colormap_rgba: np.ndarray,
        domain: tuple[float, float],
        config: LegendConfig | None = None,
    ):
        """
        Args:
            colormap_rgba: (N, 4) float32 array of RGBA values in [0, 1]
            domain: (min_value, max_value) for the legend range
            config: Legend configuration
        """
        self.colormap_rgba = colormap_rgba
        self.domain = domain
        self.config = config or LegendConfig()

    @classmethod
    def from_colormap(cls, colormap, domain: tuple[float, float], config: LegendConfig | None = None):
        """Create Legend from a forge3d Colormap object."""
        return cls(colormap.rgba, domain, config)

    def get_ticks(self) -> list[tuple[float, str]]:
        """Return (value, label) pairs for tick positions."""
        cfg = self.config
        vmin, vmax = self.domain
        ticks = []
        for i in range(cfg.tick_count):
            t = i / (cfg.tick_count - 1) if cfg.tick_count > 1 else 0.5
            value = vmin + t * (vmax - vmin)
            label = cfg.label_format.format(value) + cfg.label_suffix
            ticks.append((value, label))
        return ticks

    def _render_gradient_bar(self) -> np.ndarray:
        """Render the colormap gradient bar."""
        cfg = self.config
        cmap = self.colormap_rgba
        n_colors = len(cmap)
        if cfg.orientation == "vertical":
            bar = np.zeros((cfg.bar_height, cfg.bar_width, 4), dtype=np.uint8)
            for y in range(cfg.bar_height):
                t = 1.0 - (y / (cfg.bar_height - 1))
                idx = int(t * (n_colors - 1))
                idx = max(0, min(n_colors - 1, idx))
                color = (cmap[idx] * 255).astype(np.uint8)
                bar[y, :] = color
        else:
            bar = np.zeros((cfg.bar_width, cfg.bar_height, 4), dtype=np.uint8)
            for x in range(cfg.bar_height):
                t = x / (cfg.bar_height - 1)
                idx = int(t * (n_colors - 1))
                idx = max(0, min(n_colors - 1, idx))
                color = (cmap[idx] * 255).astype(np.uint8)
                bar[:, x] = color
        return bar

    def render(self) -> np.ndarray:
        """Render the complete legend to an RGBA image."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return self._render_gradient_bar()

        cfg = self.config
        bar = self._render_gradient_bar()
        bar_h, bar_w = bar.shape[:2]
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", cfg.font_size)
            title_font = ImageFont.truetype("DejaVuSans.ttf", cfg.title_font_size)
        except OSError:
            try:
                font = ImageFont.truetype("Arial.ttf", cfg.font_size)
                title_font = ImageFont.truetype("Arial.ttf", cfg.title_font_size)
            except OSError:
                font = ImageFont.load_default()
                title_font = font
        ticks = self.get_ticks()
        max_label_width = 0
        label_height = 0
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        for _, label in ticks:
            bbox = draw.textbbox((0, 0), label, font=font)
            max_label_width = max(max_label_width, bbox[2] - bbox[0])
            label_height = max(label_height, bbox[3] - bbox[1])
        title_height = 0
        if cfg.title:
            bbox = draw.textbbox((0, 0), cfg.title, font=title_font)
            title_height = bbox[3] - bbox[1] + cfg.padding
        if cfg.orientation == "vertical":
            total_width = cfg.padding + bar_w + cfg.tick_length + cfg.padding + max_label_width + cfg.padding
            total_height = cfg.padding + title_height + bar_h + cfg.padding
        else:
            total_width = cfg.padding + bar_w + cfg.padding
            total_height = cfg.padding + title_height + bar_h + cfg.tick_length + cfg.padding + label_height + cfg.padding
        img = Image.new("RGBA", (total_width, total_height), cfg.background)
        draw = ImageDraw.Draw(img)
        if cfg.orientation == "vertical":
            bar_x = cfg.padding
            bar_y = cfg.padding + title_height
            bar_pil = Image.fromarray(bar)
            img.paste(bar_pil, (bar_x, bar_y))
            for i, (value, label) in enumerate(ticks):
                t = (value - self.domain[0]) / (self.domain[1] - self.domain[0])
                y = bar_y + int((1.0 - t) * (bar_h - 1))
                tick_x0 = bar_x + bar_w
                tick_x1 = tick_x0 + cfg.tick_length
                draw.line([(tick_x0, y), (tick_x1, y)], fill=cfg.tick_color, width=1)
                bbox = draw.textbbox((0, 0), label, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                label_x = tick_x1 + cfg.padding // 2
                label_y = y - lh // 2
                draw.text((label_x, label_y), label, font=font, fill=cfg.label_color)
            if cfg.title:
                bbox = draw.textbbox((0, 0), cfg.title, font=title_font)
                tw = bbox[2] - bbox[0]
                title_x = (total_width - tw) // 2
                title_y = cfg.padding // 2
                draw.text((title_x, title_y), cfg.title, font=title_font, fill=cfg.label_color)
        else:
            bar_x = cfg.padding
            bar_y = cfg.padding + title_height
            bar_pil = Image.fromarray(bar)
            img.paste(bar_pil, (bar_x, bar_y))
            for i, (value, label) in enumerate(ticks):
                t = (value - self.domain[0]) / (self.domain[1] - self.domain[0])
                x = bar_x + int(t * (bar_h - 1))
                tick_y0 = bar_y + bar_w
                tick_y1 = tick_y0 + cfg.tick_length
                draw.line([(x, tick_y0), (x, tick_y1)], fill=cfg.tick_color, width=1)
                bbox = draw.textbbox((0, 0), label, font=font)
                lw = bbox[2] - bbox[0]
                label_x = x - lw // 2
                label_y = tick_y1 + 2
                draw.text((label_x, label_y), label, font=font, fill=cfg.label_color)
            if cfg.title:
                bbox = draw.textbbox((0, 0), cfg.title, font=title_font)
                tw = bbox[2] - bbox[0]
                title_x = (total_width - tw) // 2
                title_y = cfg.padding // 2
                draw.text((title_x, title_y), cfg.title, font=title_font, fill=cfg.label_color)
        return np.array(img, dtype=np.uint8)
