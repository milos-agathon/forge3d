"""Raster registration at the normalized corner-affine boundary.

GeoTIFF Point anchors are sample positions. GDAL exposes both kinds of
GeoTIFF with a corner affine, so a sample is always at (col + .5, row + .5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .preparation import PreparationGrid

PixelRegistration = Literal["area", "point"]
RegistrationConversion = Literal["preserve_sample_locations"]


def pixel_registration(value: object) -> PixelRegistration:
    if value == "area":
        return "area"
    if value == "point":
        return "point"
    raise ValueError(f"invalid raster registration {value!r}; expected 'area' or 'point'")


def geotiff_registration(dataset) -> PixelRegistration:
    # RasterPixelIsArea is the GeoTIFF default when GTRasterTypeGeoKey is absent.
    return pixel_registration(dataset.tags().get("AREA_OR_POINT", "Area").lower())


def corner_to_node_uv(column: float, row: float, width: int, height: int) -> tuple[float, float]:
    """Convert corner-normalized raster coordinates to terrain node UV."""
    u = (column - 0.5) / (width - 1) if width > 1 else column
    v = (row - 0.5) / (height - 1) if height > 1 else row
    return u, v


def sample_extent_output_grid(
    source: PreparationGrid, width: int, height: int,
) -> PreparationGrid:
    """Area pixels filling the viewport between the outer terrain samples."""
    from .preparation import PreparationGrid

    if width <= 0 or height <= 0:
        raise ValueError("output grid dimensions must be positive")
    dx = (source.width - 1) / width if source.width > 1 else 1 / width
    dy = (source.height - 1) / height if source.height > 1 else 1 / height
    # For a singleton axis there is one texture cell and no distance between
    # nodes. Its complete cell provides the continuous viewport.
    transform = source.transform
    x = 0.5 if source.width > 1 else 0.0
    y = 0.5 if source.height > 1 else 0.0
    output = transform * transform.translation(x, y) * transform.scale(dx, dy)
    return PreparationGrid(source.crs, width, height, output.to_gdal(), registration="area")


def require_compatible_registration(
    source: PixelRegistration,
    target: PixelRegistration,
    conversion: RegistrationConversion | None = None,
) -> float:
    """Return the change of native affine origin, in pixel coordinates.

    Area -> Point adds half of each pixel axis to the native origin. Point ->
    Area subtracts the same. A corner-normalized affine already accounts for
    this, so a subsequent warp uses its source and target affines unchanged.
    """
    if conversion not in (None, "preserve_sample_locations"):
        raise ValueError(f"invalid raster registration conversion {conversion!r}")
    if source == target:
        return 0.0
    if conversion is None:
        raise ValueError(
            f"raster registration mismatch ({source} to {target}); "
            "declare registration_conversion='preserve_sample_locations'"
        )
    return 0.5 if target == "point" else -0.5
