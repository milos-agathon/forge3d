"""P4.2a: Legend generation from colormaps.

Renders elevation/value legends with gradient bars, tick marks, and labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .certificate import _captured_cpu_render


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB values in [0, 1] to display sRGB."""
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


def _rgba_linear_to_display_u8(rgba: np.ndarray) -> np.ndarray:
    """Convert a linear RGBA color to uint8 display space."""
    color = np.clip(np.asarray(rgba, dtype=np.float32), 0.0, 1.0)
    out = np.empty(4, dtype=np.uint8)
    out[:3] = np.round(_linear_to_srgb(color[:3]) * 255.0).astype(np.uint8)
    out[3] = np.uint8(round(float(color[3]) * 255.0))
    return out


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
                color = _rgba_linear_to_display_u8(cmap[idx])
                bar[y, :] = color
        else:
            bar = np.zeros((cfg.bar_width, cfg.bar_height, 4), dtype=np.uint8)
            for x in range(cfg.bar_height):
                t = x / (cfg.bar_height - 1)
                idx = int(t * (n_colors - 1))
                idx = max(0, min(n_colors - 1, idx))
                color = _rgba_linear_to_display_u8(cmap[idx])
                bar[:, x] = color
        return bar

    @_captured_cpu_render("python.legend.render", "legend.cpu", draw_calls=1)
    def render(self, *, certificate: bool | str = False, cache: str | None = None) -> np.ndarray:
        """Render geometry plus every label through packaged native text."""
        _ = cache
        from ._map_scene_render import _draw_text, _text_outline_metrics

        def extent(text: str, size: int) -> tuple[int, int]:
            width, height, _bounds = _text_outline_metrics(text, float(size))
            return width, max(height, max(1, int(np.ceil(size * 1.3))))

        cfg = self.config
        bar = self._render_gradient_bar()
        bar_h, bar_w = bar.shape[:2]
        ticks = self.get_ticks()
        label_extents = [extent(label, cfg.font_size) for _, label in ticks]
        max_label_width = max((value[0] for value in label_extents), default=0)
        label_height = max((value[1] for value in label_extents), default=0)
        title_extent = extent(cfg.title, cfg.title_font_size) if cfg.title else (0, 0)
        title_height = title_extent[1] + cfg.padding if cfg.title else 0
        if cfg.orientation == "vertical":
            total_width = cfg.padding + bar_w + cfg.tick_length + cfg.padding + max_label_width + cfg.padding
            total_height = cfg.padding + title_height + bar_h + cfg.padding
        else:
            total_width = cfg.padding + bar_w + cfg.padding
            total_height = cfg.padding + title_height + bar_h + cfg.tick_length + cfg.padding + label_height + cfg.padding
        image = np.empty((total_height, total_width, 4), dtype=np.uint8)
        image[...] = cfg.background
        span = self.domain[1] - self.domain[0]
        if cfg.orientation == "vertical":
            bar_x = cfg.padding
            bar_y = cfg.padding + title_height
            image[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = bar
            for index, (value, label) in enumerate(ticks):
                t = (value - self.domain[0]) / span if span else 0.5
                y = bar_y + int((1.0 - t) * (bar_h - 1))
                tick_x0 = bar_x + bar_w
                tick_x1 = tick_x0 + cfg.tick_length
                image[y : y + 1, tick_x0 : tick_x1 + 1] = cfg.tick_color
                _lw, lh = label_extents[index]
                label_x = tick_x1 + cfg.padding // 2
                label_y = y - lh // 2
                _draw_text(
                    image,
                    label,
                    (label_x, label_y),
                    color=cfg.label_color,
                    halo=(0, 0, 0, 0),
                    halo_width_px=0.0,
                    font_size=float(cfg.font_size),
                )
        else:
            bar_x = cfg.padding
            bar_y = cfg.padding + title_height
            image[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = bar
            for index, (value, label) in enumerate(ticks):
                t = (value - self.domain[0]) / span if span else 0.5
                x = bar_x + int(t * (bar_w - 1))
                tick_y0 = bar_y + bar_h
                tick_y1 = tick_y0 + cfg.tick_length
                image[tick_y0 : tick_y1 + 1, x : x + 1] = cfg.tick_color
                lw, _lh = label_extents[index]
                label_x = x - lw // 2
                label_y = tick_y1 + 2
                _draw_text(
                    image,
                    label,
                    (label_x, label_y),
                    color=cfg.label_color,
                    halo=(0, 0, 0, 0),
                    halo_width_px=0.0,
                    font_size=float(cfg.font_size),
                )
        if cfg.title:
            _draw_text(
                image,
                cfg.title,
                ((total_width - title_extent[0]) // 2, cfg.padding // 2),
                color=cfg.label_color,
                halo=(0, 0, 0, 0),
                halo_width_px=0.0,
                font_size=float(cfg.title_font_size),
            )
        return image
