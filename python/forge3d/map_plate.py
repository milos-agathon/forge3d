"""P4.1: Map Plate Layout for publication-ready cartographic output.

This module provides the MapPlate class for composing terrain renders
with titles, legends, scale bars, and inset maps into a single image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

PositionAnchor = Literal[
    "top", "bottom", "left", "right",
    "top-left", "top-right", "bottom-left", "bottom-right",
    "center"
]


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in a given CRS."""
    west: float
    south: float
    east: float
    north: float
    crs: str = "EPSG:4326"

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south

    @property
    def center_lat(self) -> float:
        return (self.north + self.south) / 2.0


@dataclass
class MapPlateConfig:
    """Configuration for a map plate."""
    width: int = 1600
    height: int = 1000
    margin: tuple[int, int, int, int] = (60, 200, 80, 40)  # top, right, bottom, left
    background: tuple[int, int, int, int] = (255, 255, 255, 255)
    dpi: int = 150


@dataclass
class PlateRegion:
    """A rectangular region within the plate."""
    x0: int
    y0: int
    x1: int
    y1: int
    content: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TitleElement:
    """Title text element."""
    text: str
    font_size: int = 24
    color: tuple[int, int, int, int] = (0, 0, 0, 255)
    position: PositionAnchor = "top"


@dataclass
class LegendElement:
    """Legend element placeholder (rendered by Legend class)."""
    rendered: np.ndarray
    position: PositionAnchor = "right"


@dataclass
class ScaleBarElement:
    """Scale bar element placeholder (rendered by ScaleBar class)."""
    rendered: np.ndarray
    position: PositionAnchor = "bottom-left"


@dataclass
class InsetElement:
    """Inset map element."""
    image: np.ndarray
    position: PositionAnchor = "bottom-right"
    size: tuple[int, int] | None = None  # width, height; None = use image size
    border_width: int = 2
    border_color: tuple[int, int, int, int] = (0, 0, 0, 255)


@dataclass
class NorthArrowElement:
    """North arrow element placeholder."""
    rendered: np.ndarray
    position: PositionAnchor = "top-right"


class MapPlate:
    """Composes a map plate with title, legend, scale bar, and inset regions."""

    def __init__(self, config: MapPlateConfig | None = None):
        self.config = config or MapPlateConfig()
        self._map_image: np.ndarray | None = None
        self._map_bbox: BBox | None = None
        self._title: TitleElement | None = None
        self._legend: LegendElement | None = None
        self._scale_bar: ScaleBarElement | None = None
        self._north_arrow: NorthArrowElement | None = None
        self._insets: list[InsetElement] = []

    def set_map_region(self, image: np.ndarray, bbox: BBox) -> None:
        """Set the main map image and its geographic bounds."""
        if image.ndim == 2:
            image = np.stack([image] * 3 + [np.full_like(image, 255)], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
            image = np.concatenate([image, alpha], axis=-1)
        self._map_image = image.astype(np.uint8)
        self._map_bbox = bbox

    def add_title(
        self,
        text: str,
        position: PositionAnchor = "top",
        font_size: int = 24,
        color: tuple[int, int, int, int] = (0, 0, 0, 255),
    ) -> None:
        """Add a title to the plate."""
        self._title = TitleElement(text=text, font_size=font_size, color=color, position=position)

    def add_legend(self, legend_image: np.ndarray, position: PositionAnchor = "right") -> None:
        """Add a pre-rendered legend image."""
        self._legend = LegendElement(rendered=legend_image.astype(np.uint8), position=position)

    def add_scale_bar(self, scale_bar_image: np.ndarray, position: PositionAnchor = "bottom-left") -> None:
        """Add a pre-rendered scale bar image."""
        self._scale_bar = ScaleBarElement(rendered=scale_bar_image.astype(np.uint8), position=position)

    def add_north_arrow(self, north_arrow_image: np.ndarray, position: PositionAnchor = "top-right") -> None:
        """Add a pre-rendered north arrow image."""
        self._north_arrow = NorthArrowElement(rendered=north_arrow_image.astype(np.uint8), position=position)

    def add_inset(
        self,
        image: np.ndarray,
        position: PositionAnchor = "bottom-right",
        size: tuple[int, int] | None = None,
        border_width: int = 2,
    ) -> None:
        """Add an inset map image."""
        self._insets.append(InsetElement(
            image=image.astype(np.uint8),
            position=position,
            size=size,
            border_width=border_width,
        ))

    def _get_map_region(self) -> PlateRegion:
        """Compute the region for the main map."""
        cfg = self.config
        top, right, bottom, left = cfg.margin
        return PlateRegion(
            x0=left,
            y0=top,
            x1=cfg.width - right,
            y1=cfg.height - bottom,
        )

    def _blit(self, canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
        """Alpha-composite image onto canvas at (x, y)."""
        h, w = image.shape[:2]
        ch, cw = canvas.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(cw, x + w), min(ch, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sx0, sy0 = x0 - x, y0 - y
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        src = image[sy0:sy1, sx0:sx1]
        dst = canvas[y0:y1, x0:x1]
        if src.shape[2] == 4:
            alpha = src[..., 3:4].astype(np.float32) / 255.0
            dst_rgb = dst[..., :3].astype(np.float32)
            src_rgb = src[..., :3].astype(np.float32)
            blended = src_rgb * alpha + dst_rgb * (1.0 - alpha)
            dst[..., :3] = blended.astype(np.uint8)
            dst[..., 3] = np.maximum(dst[..., 3], src[..., 3])
        else:
            dst[..., :3] = src[..., :3]

    def _render_title(self) -> np.ndarray | None:
        """Render title text to an RGBA image using PIL."""
        if self._title is None:
            return None
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            return None
        font_size = self._title.font_size
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("Arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
        dummy = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), self._title.text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding = 10
        img = Image.new("RGBA", (tw + padding * 2, th + padding * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((padding, padding), self._title.text, font=font, fill=self._title.color)
        return np.array(img, dtype=np.uint8)

    def _position_element(
        self,
        element_size: tuple[int, int],
        position: PositionAnchor,
        region: PlateRegion,
        margin: int = 10,
    ) -> tuple[int, int]:
        """Compute (x, y) for placing an element in a region."""
        ew, eh = element_size
        rx0, ry0, rx1, ry1 = region.rect
        rw, rh = rx1 - rx0, ry1 - ry0
        if position == "top":
            return rx0 + (rw - ew) // 2, ry0 + margin
        elif position == "bottom":
            return rx0 + (rw - ew) // 2, ry1 - eh - margin
        elif position == "left":
            return rx0 + margin, ry0 + (rh - eh) // 2
        elif position == "right":
            return rx1 - ew - margin, ry0 + (rh - eh) // 2
        elif position == "top-left":
            return rx0 + margin, ry0 + margin
        elif position == "top-right":
            return rx1 - ew - margin, ry0 + margin
        elif position == "bottom-left":
            return rx0 + margin, ry1 - eh - margin
        elif position == "bottom-right":
            return rx1 - ew - margin, ry1 - eh - margin
        elif position == "center":
            return rx0 + (rw - ew) // 2, ry0 + (rh - eh) // 2
        return rx0, ry0

    def compose(self) -> np.ndarray:
        """Compose all elements into a final RGBA image."""
        cfg = self.config
        canvas = np.zeros((cfg.height, cfg.width, 4), dtype=np.uint8)
        canvas[..., :] = cfg.background
        map_region = self._get_map_region()
        if self._map_image is not None:
            mh, mw = self._map_image.shape[:2]
            rw, rh = map_region.width, map_region.height
            scale = min(rw / mw, rh / mh)
            new_w, new_h = int(mw * scale), int(mh * scale)
            try:
                from PIL import Image
                pil_img = Image.fromarray(self._map_image)
                pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                resized = np.array(pil_img)
            except ImportError:
                resized = self._map_image
                new_w, new_h = mw, mh
            x = map_region.x0 + (rw - new_w) // 2
            y = map_region.y0 + (rh - new_h) // 2
            self._blit(canvas, resized, x, y)
        if self._title is not None:
            title_img = self._render_title()
            if title_img is not None:
                full_region = PlateRegion(0, 0, cfg.width, cfg.height)
                tx, ty = self._position_element(
                    (title_img.shape[1], title_img.shape[0]),
                    self._title.position,
                    full_region,
                    margin=15,
                )
                self._blit(canvas, title_img, tx, ty)
        if self._legend is not None:
            leg_img = self._legend.rendered
            leg_h, leg_w = leg_img.shape[:2]
            legend_region = PlateRegion(
                cfg.width - cfg.margin[1], cfg.margin[0],
                cfg.width - 10, cfg.height - cfg.margin[2]
            )
            lx, ly = self._position_element(
                (leg_w, leg_h), "center", legend_region, margin=5
            )
            self._blit(canvas, leg_img, lx, ly)
        if self._scale_bar is not None:
            sb_img = self._scale_bar.rendered
            sb_h, sb_w = sb_img.shape[:2]
            sx, sy = self._position_element(
                (sb_w, sb_h), self._scale_bar.position, map_region, margin=20
            )
            self._blit(canvas, sb_img, sx, sy)
        if self._north_arrow is not None:
            na_img = self._north_arrow.rendered
            na_h, na_w = na_img.shape[:2]
            nx, ny = self._position_element(
                (na_w, na_h), self._north_arrow.position, map_region, margin=15
            )
            self._blit(canvas, na_img, nx, ny)
        for inset in self._insets:
            inset_img = inset.image
            if inset.size is not None:
                try:
                    from PIL import Image
                    pil_img = Image.fromarray(inset_img)
                    pil_img = pil_img.resize(inset.size, Image.LANCZOS)
                    inset_img = np.array(pil_img)
                except ImportError:
                    pass
            ih, iw = inset_img.shape[:2]
            ix, iy = self._position_element(
                (iw, ih), inset.position, map_region, margin=15
            )
            if inset.border_width > 0:
                bw = inset.border_width
                bc = inset.border_color
                canvas[iy - bw:iy + ih + bw, ix - bw:ix + iw + bw, :3] = bc[:3]
                canvas[iy - bw:iy + ih + bw, ix - bw:ix + iw + bw, 3] = bc[3]
            self._blit(canvas, inset_img, ix, iy)
        return canvas

    def export_png(self, path: Path | str) -> None:
        """Compose and export to PNG file."""
        from PIL import Image
        composed = self.compose()
        Image.fromarray(composed).save(str(path))

    def export_jpeg(self, path: Path | str, quality: int = 90) -> None:
        """Compose and export to JPEG file."""
        from PIL import Image
        composed = self.compose()
        rgb = composed[..., :3]
        Image.fromarray(rgb).save(str(path), quality=quality)
