"""Measured WGS 84 terrain transects and linked stills."""

from __future__ import annotations

import csv
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ._canonical_json import canonical_json_bytes
from .bundle import _compute_sha256
from .crs import geodesic_direct, geodesic_inverse

__all__ = ["Transect", "TerrainSample", "TerrainGrid", "MeasuredSection", "sample_transect", "render_section"]

LonLat = tuple[float, float]
SampleStatus = Literal["valid", "nodata", "out_of_bounds"]
_PIXEL_CONVENTION = (
    "Zero-based row/column, GDAL affine at pixel edges. "
    "Cells own their west and north edges and exclude their east and south edges."
)


@dataclass(frozen=True)
class Transect:
    """Two (longitude, latitude) endpoints in EPSG:4326 and maximum spacing in metres."""

    start: LonLat
    end: LonLat
    spacing_m: float


@dataclass(frozen=True)
class TerrainSample:
    index: int
    longitude: float
    latitude: float
    distance_m: float
    elevation_m: float | None
    status: SampleStatus
    row: int | None
    column: int | None


@dataclass(frozen=True)
class TerrainGrid:
    source_id: str
    sha256: str
    raster_crs: str
    horizontal_crs: str
    vertical_crs: str
    vertical_reference: str
    elevation_unit: str
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    nodata: float | Literal["NaN", "+Infinity", "-Infinity"] | None
    pixel_registration: str | None


@dataclass(frozen=True)
class MeasuredSection:
    transect: Transect
    grid: TerrainGrid
    samples: tuple[TerrainSample, ...]

    def to_json(self) -> str:
        """Serialize metadata and all sample rows, including null elevations."""
        return canonical_json_bytes({
            "schema_version": 1, "source": asdict(self.grid), "transect": asdict(self.transect),
            "distance_unit": "m", "coordinate_unit": "degree",
            "distance_model": "WGS 84 ellipsoid, shortest geodesic",
            "interpolation": "nearest (containing cell)",
            "spacing_rule": "Equal intervals, at most spacing_m, with both endpoints included.",
            "interval_m": self.samples[-1].distance_m / (len(self.samples) - 1),
            "pixel_convention": _PIXEL_CONVENTION,
            "samples": [asdict(sample) for sample in self.samples],
        }, error_context="Measured section").decode("utf-8")

    def to_csv(self) -> str:
        """Serialize the sample table. Use the companion JSON for the grid and convention."""
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=TerrainSample.__dataclass_fields__,
                                lineterminator="\n")
        writer.writeheader()
        for sample in self.samples:
            writer.writerow(asdict(sample))
        return output.getvalue()


def _read_grid(path: Path, source_id: str, vertical_crs: str) -> tuple[TerrainGrid, np.ma.MaskedArray]:
    import rasterio
    from pyproj import CRS
    from pyproj.exceptions import CRSError

    if not source_id.strip():
        raise ValueError("Declare a nonempty terrain source ID")
    try:
        vertical = CRS.from_user_input(vertical_crs)
    except CRSError as exc:
        raise ValueError("Declare a supported vertical CRS explicitly") from exc
    if (not vertical.is_vertical or vertical.to_epsg() is None
            or vertical.axis_info[0].direction != "up"
            or vertical.axis_info[0].unit_conversion_factor != 1.0):
        raise ValueError("Vertical CRS must be an EPSG vertical height CRS in metres, positive up")
    horizontal = CRS.from_epsg(4326)
    with rasterio.open(path) as dataset:
        if dataset.driver != "GTiff" or len(dataset.files) != 1 or dataset.count != 1:
            raise ValueError("Use a self-contained, single-band terrain GeoTIFF")
        if dataset.crs is None:
            raise ValueError("Raster must declare a horizontal CRS")
        raster_crs = CRS.from_wkt(dataset.crs.to_wkt())
        horizontal_part = raster_crs
        if raster_crs.is_compound:
            horizontal_part, vertical_part = raster_crs.sub_crs_list
            if not vertical.equals(vertical_part):
                raise ValueError("Declared vertical CRS differs from the GeoTIFF vertical CRS")
        if not horizontal_part.equals(horizontal, ignore_axis_order=True):
            raise ValueError("Raster horizontal CRS must be EPSG:4326 (WGS 84 longitude/latitude)")
        if (dataset.units[0] not in (None, "", "m", "metre", "meter")
                or dataset.scales[0] != 1 or dataset.offsets[0] != 0):
            raise ValueError("Terrain values must be stored directly in metres, without scale or offset")
        a, b, c, d, e, f = tuple(dataset.transform)[:6]
        if not (all(map(math.isfinite, (a, b, c, d, e, f))) and b == d == 0 and a > 0 and e < 0
                and -180 <= c < c + a * dataset.width <= 180
                and -90 <= f + e * dataset.height < f <= 90):
            raise ValueError("Use a north-up WGS 84 raster within longitude/latitude bounds")
        values = dataset.read(1, masked=True)
        if values.dtype.kind not in "iuf":
            raise ValueError("Terrain elevations must be real numbers")
        nodata = dataset.nodata
        mask = np.ma.getmaskarray(values) | ~np.isfinite(values.data)
        if nodata is not None and math.isfinite(nodata):
            mask |= values.data == nodata
        values.mask = mask
        if nodata is not None and not math.isfinite(nodata):
            nodata = "NaN" if math.isnan(nodata) else ("+Infinity" if nodata > 0 else "-Infinity")
        grid = TerrainGrid(
            source_id, _compute_sha256(path), raster_crs.to_string(), "EPSG:4326",
            vertical.to_string(), vertical.name, "m", (a, b, c, d, e, f),
            dataset.width, dataset.height, nodata, dataset.tags().get("AREA_OR_POINT"),
        )
    return grid, values


def _positions(transect: Transect) -> tuple[Transect, list[tuple[LonLat, float]]]:
    start, end = tuple(map(float, transect.start)), tuple(map(float, transect.end))
    spacing = float(transect.spacing_m)
    for lon, lat in (start, end):
        if not (-180 <= lon <= 180 and -90 < lat < 90):
            raise ValueError("Transect endpoints must be finite WGS 84 lon/lat, excluding the poles")
    if abs(start[0] - end[0]) >= 180:
        raise ValueError("Antimeridian-crossing and ambiguous opposite-meridian transects are unsupported")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Transect spacing_m must be finite and positive")

    # Construct the same positions in either direction, including on cell edges.
    first, last = sorted((start, end))
    inverse = geodesic_inverse(first[1], first[0], last[1], last[0])
    length = inverse["s12"]
    if length <= 0:
        raise ValueError("Transect endpoints must differ")
    intervals = max(1, math.ceil(length / spacing))
    positions = [(first, 0.0)]
    for index in range(1, intervals):
        distance = length * (index / intervals)
        point = geodesic_direct(first[1], first[0], inverse["azi1"], distance)
        positions.append(((point["lon2"], point["lat2"]), distance))
    positions.append((last, length))
    if start != first:
        positions = [(point, length - distance) for point, distance in reversed(positions)]
    return Transect(start, end, spacing), positions


def sample_transect(
    path: str | Path, transect: Transect, *, source_id: str, vertical_crs: str,
) -> MeasuredSection:
    """Measure a geodesic transect on a north-up WGS 84 GeoTIFF.

    Supply the source's documented vertical CRS even when the file has only
    horizontal metadata. Heights are preserved in that reference.
    """
    grid, values = _read_grid(Path(path), source_id, vertical_crs)
    declared, positions = _positions(transect)
    a, _, west, _, e, north = grid.transform
    x_edges = west + a * np.arange(grid.width + 1)
    y_edges = -north - e * np.arange(grid.height + 1)
    samples = []
    for index, ((lon, lat), distance) in enumerate(positions):
        col = int(np.searchsorted(x_edges, lon, side="right")) - 1
        row = int(np.searchsorted(y_edges, -lat, side="right")) - 1
        if not (0 <= row < grid.height and 0 <= col < grid.width):
            samples.append(TerrainSample(index, lon, lat, distance, None, "out_of_bounds", None, None))
        elif values.mask[row, col]:
            samples.append(TerrainSample(index, lon, lat, distance, None, "nodata", row, col))
        else:
            elevation = float(values.data[row, col])
            samples.append(TerrainSample(index, lon, lat, distance, elevation or 0.0, "valid", row, col))
    return MeasuredSection(declared, grid, tuple(samples))


def render_section(
    section: MeasuredSection, path: str | Path, directory: str | Path, *, title: str,
    certificate: bool | str | Path = False, cache: str | Path | None = None,
) -> dict:
    """Render a linked map/profile still and write its sample table and receipts.

    The output directory must be new. This uses the C06 MapPlate Pro feature.
    Certificate and validated base-map cache options pass through to MapScene.
    """
    from ._measured_section_plate import render_section_plate

    return render_section_plate(section, Path(path), Path(directory), title, certificate, cache=cache)
