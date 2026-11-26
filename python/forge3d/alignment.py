"""CRS and geotransform alignment helpers for MapScene recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._map_scene_common import _layer_id, _metadata, _same_crs, _stable_hash
from .crs import transform_coords
from ._raster_sampling import output_nodata, read_band_validity
from .raster_registration import (
    PixelRegistration, RegistrationConversion, geotiff_registration,
    pixel_registration, require_compatible_registration,
)


def _is_point(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2


def _transform_points(points: Sequence[Any], from_crs: str, to_crs: str) -> list[list[float]]:
    xy = np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64)
    transformed = transform_coords(xy, from_crs, to_crs)
    result: list[list[float]] = []
    for source, target in zip(points, transformed):
        item = [float(target[0]), float(target[1])]
        if len(source) > 2:
            item.extend(float(value) for value in source[2:])
        result.append(item)
    return result


def transform_geometry(geometry: Mapping[str, Any], from_crs: str, to_crs: str) -> dict[str, Any]:
    """Transform a GeoJSON-like Point, LineString, or Polygon geometry."""

    geometry_type = str(geometry.get("type", ""))
    if geometry_type == "GeometryCollection":
        return {**geometry, "geometries": [transform_geometry(g, from_crs, to_crs)
                                           for g in geometry.get("geometries", [])]}
    depths = {"Point": 0, "MultiPoint": 1, "LineString": 1,
              "MultiLineString": 2, "Polygon": 2, "MultiPolygon": 3}
    if geometry_type not in depths:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")

    def transform(value, depth):
        if depth == 0:
            if not _is_point(value):
                raise ValueError("Invalid geometry position")
            return _transform_points([value], from_crs, to_crs)[0]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError("Invalid geometry coordinates")
        return [transform(child, depth - 1) for child in value]

    return {**geometry, "coordinates": transform(geometry.get("coordinates"), depths[geometry_type])}


def transform_features(
    features: Sequence[Mapping[str, Any]],
    from_crs: str,
    to_crs: str,
) -> list[dict[str, Any]]:
    """Transform GeoJSON-like feature geometries into ``to_crs``."""

    transformed_features = []
    for feature in features or ():
        item = dict(feature)
        geometry = item.get("geometry")
        if isinstance(geometry, Mapping):
            item["geometry"] = transform_geometry(geometry, from_crs, to_crs)
        transformed_features.append(item)
    return transformed_features


def _require_rasterio() -> Any:
    try:
        import rasterio
    except Exception as exc:  # pragma: no cover - exercised only without optional dep
        raise ImportError(
            "rasterio is required for raster alignment. "
            "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
        ) from exc
    if getattr(rasterio, "__forge3d_stub__", False):
        raise ImportError(
            "rasterio is required for raster alignment. "
            "Install with: pip install rasterio (or pip install 'forge3d[raster]')."
        )
    return rasterio


def _resampling_method(name: str) -> Any:
    _require_rasterio()
    from rasterio.enums import Resampling

    key = str(name or "nearest").lower()
    aliases = {"bilinear": "bilinear", "nearest": "nearest", "mode": "mode", "cubic": "cubic"}
    if key not in aliases:
        raise ValueError("resampling must be one of: nearest, bilinear, cubic, mode")
    return getattr(Resampling, aliases[key])


def _affine_to_metadata(transform: Any) -> list[float]:
    return [float(value) for value in transform.to_gdal()]


def _affine_from_metadata(value: Any) -> Any:
    _require_rasterio()
    from affine import Affine

    if isinstance(value, Affine):
        return value
    if isinstance(value, Mapping):
        if "transform" in value:
            return _affine_from_metadata(value["transform"])
        if "geotransform" in value:
            return _affine_from_metadata(value["geotransform"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
        if len(values) == 6:
            return Affine.from_gdal(*values)
        if len(values) >= 9:
            return Affine(values[0], values[1], values[2], values[3], values[4], values[5])
    raise ValueError("target grid requires a rasterio Affine transform or a 6-value GDAL geotransform")


def _grid_shape(grid: Mapping[str, Any]) -> tuple[int, int]:
    if "height" in grid and "width" in grid:
        return int(grid["height"]), int(grid["width"])
    shape = grid.get("shape")
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    raise ValueError("target grid requires height/width or shape")


def _grid_metadata(crs: Any, transform: Any, width: int, height: int, nodata: Any = None,
                   *, registration: PixelRegistration = "area") -> dict[str, Any]:
    return {
        "crs": str(crs) if crs is not None else None,
        "width": int(width),
        "height": int(height),
        "geotransform": _affine_to_metadata(transform),
        "resolution": [abs(float(transform.a)), abs(float(transform.e))],
        "nodata": nodata,
        "registration": registration,
    }


def _warp_band(values, valid, destination, *, src_transform, src_crs, dst_transform,
               dst_crs, method, nodata):
    """Warp continuous values and support using identical kernel coordinates."""
    from rasterio.warp import reproject
    kwargs = dict(src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=dst_crs,
                  resampling=_resampling_method(method))
    if valid.all() or str(method).lower() in ("nearest", "mode"):
        scratch = values if valid.all() else np.where(valid, values, np.nan)
        # Coverage determines padding. Passing a finite destination marker here
        # lets GDAL perturb a real value that rounds onto that marker.
        reproject(scratch, destination, src_nodata=None if valid.all() else np.nan,
                  init_dest_nodata=False, **kwargs)
        # Distinguish real zero values (including integer imagery) from padding.
        weight = np.zeros(destination.shape, np.float64)
        reproject(valid.astype(np.float64), weight, dst_nodata=0, **kwargs)
        covered = weight >= 0.5
        destination[~covered] = nodata
        return covered
    weight = np.zeros(destination.shape, np.float64)
    total = np.zeros_like(weight)
    reproject(valid.astype(np.float64), weight, dst_nodata=0, **kwargs)
    reproject(np.where(valid, values, 0).astype(np.float64), total, dst_nodata=0, **kwargs)
    covered = weight >= 0.5
    result = total[covered] / weight[covered]
    if np.issubdtype(destination.dtype, np.integer):
        limits = np.iinfo(destination.dtype)
        result = np.clip(np.rint(result), limits.min, limits.max)
    destination.fill(nodata)
    destination[covered] = result
    return covered


def _verify_written_validity(rasterio, path, expected):
    with rasterio.open(path) as written:
        for _, window in written.block_windows(1):
            actual = np.ones((int(window.height), int(window.width)), dtype=bool)
            for band in written.indexes:
                _, valid = read_band_validity(written, band, window=window)
                actual &= valid
            if not np.array_equal(actual, expected[window.toslices()]):
                raise ValueError(f"GeoTIFF validity changed after writing {path} at {window}")


def reproject_dem_to_target(
    source: str | Path,
    target_crs: str,
    *,
    output_path: str | Path | None = None,
    resampling: str = "bilinear",
    dst_nodata: float | None = None,
) -> dict[str, Any]:
    """Reproject a single-band DEM GeoTIFF into ``target_crs``.

    Returns the reprojected array plus exact target grid metadata. If
    ``output_path`` is provided, the same data is written as a GeoTIFF.
    """

    rasterio = _require_rasterio()
    from rasterio.warp import calculate_default_transform

    src_path = Path(source)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("source DEM is missing CRS metadata")
        registration = geotiff_registration(src)
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
        )
        nodata = dst_nodata if dst_nodata is not None else src.nodata
        if nodata is None:
            nodata = -9999.0
        destination = np.full((height, width), np.nan, dtype=np.float32)
        raw, valid = read_band_validity(src, 1)
        validity = _warp_band(raw, valid, destination, src_transform=src.transform,
                              src_crs=src.crs, dst_transform=transform, dst_crs=target_crs,
                              method=resampling, nodata=np.nan)
        if np.any(validity & ~np.isfinite(destination)):
            raise ValueError("valid warped elevation cannot be represented as Float32")
        nodata = output_nodata(destination, validity, nodata)
        destination[~validity] = nodata
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
        )
        written_path: str | None = None
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(destination, 1)
                dst.write_mask(validity)
                dst.update_tags(AREA_OR_POINT=registration.title())
                dst.scales = src.scales[:1]
                dst.offsets = src.offsets[:1]
            _verify_written_validity(rasterio, out_path, validity)
            written_path = str(out_path)
        metadata = _grid_metadata(target_crs, transform, width, height, nodata, registration=registration)
        metadata["scales"] = list(src.scales[:1])
        metadata["offsets"] = list(src.offsets[:1])
        metadata.update(
            {
                "source_crs": str(src.crs),
                "target_crs": str(target_crs),
                "source_geotransform": _affine_to_metadata(src.transform),
                "resampling": str(resampling),
            }
        )
    return {"array": destination, "validity_mask": validity, "metadata": metadata, "profile": profile, "path": written_path}


def resample_raster_to_grid(
    source: str | Path,
    target_grid: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    resampling: str = "nearest",
    dst_nodata: float | int | None = None,
    registration_conversion: RegistrationConversion | None = None,
) -> dict[str, Any]:
    """Resample a raster onto an explicit target grid."""

    rasterio = _require_rasterio()

    target_crs = target_grid.get("crs") or target_grid.get("target_crs")
    if target_crs is None:
        raise ValueError("target grid requires crs")
    transform = _affine_from_metadata(target_grid.get("transform", target_grid.get("geotransform")))
    height, width = _grid_shape(target_grid)
    src_path = Path(source)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("source raster is missing CRS metadata")
        source_registration = geotiff_registration(src)
        registration = pixel_registration(target_grid.get("registration", "area"))
        native_origin_shift = require_compatible_registration(
            source_registration, registration, registration_conversion,
        )
        nodata = dst_nodata if dst_nodata is not None else target_grid.get("nodata", src.nodata)
        fill = 0 if nodata is None else nodata
        destination = np.full((src.count, height, width), fill, dtype=src.dtypes[0])
        validity = np.ones((height, width), bool)
        for band in range(1, src.count + 1):
            raw, valid = read_band_validity(src, band)
            validity &= _warp_band(raw, valid, destination[band-1], src_transform=src.transform,
                                   src_crs=src.crs, dst_transform=transform, dst_crs=target_crs,
                                   method=resampling, nodata=fill)
        nodata = output_nodata(destination, validity, nodata)
        if nodata is not None and np.isnan(nodata):
            destination[:, ~validity] = nodata
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            nodata=nodata,
        )
        written_path: str | None = None
        if output_path is not None:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(destination)
                dst.write_mask(validity)
                dst.update_tags(AREA_OR_POINT=registration.title())
                dst.scales = src.scales
                dst.offsets = src.offsets
            _verify_written_validity(rasterio, out_path, validity)
            written_path = str(out_path)
        metadata = _grid_metadata(target_crs, transform, width, height, nodata, registration=registration)
        if native_origin_shift:
            metadata["registration_conversion"] = {
                "method": registration_conversion, "from": source_registration, "to": registration,
                "native_origin_shift_pixels": [native_origin_shift, native_origin_shift],
            }
        metadata["scales"] = list(src.scales)
        metadata["offsets"] = list(src.offsets)
        metadata.update(
            {
                "source_crs": str(src.crs),
                "target_crs": str(target_crs),
                "source_geotransform": _affine_to_metadata(src.transform),
                "resampling": str(resampling),
            }
        )
    array = destination[0] if destination.shape[0] == 1 else destination
    return {"array": array, "validity_mask": validity > 0, "metadata": metadata, "profile": profile, "path": written_path}


def alignment_residual(
    points: Sequence[Sequence[float]],
    from_crs: str,
    to_crs: str,
    *,
    expected: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Measure CRS transform residual for control points.

    Without ``expected`` this performs a forward/backward round trip and reports
    residuals in the source CRS coordinate units. With ``expected`` it compares
    forward-transformed points to the supplied target-CRS control points.
    """

    source = np.asarray([[float(point[0]), float(point[1])] for point in points], dtype=np.float64)
    if source.ndim != 2 or source.shape[0] == 0:
        raise ValueError("at least one control point is required")
    forward = transform_coords(source, from_crs, to_crs)
    if expected is None:
        comparison = transform_coords(forward, to_crs, from_crs)
        reference = source
        units = str(from_crs)
        mode = "roundtrip"
    else:
        reference = np.asarray([[float(point[0]), float(point[1])] for point in expected], dtype=np.float64)
        if reference.shape != forward.shape:
            raise ValueError("expected control points must match source point count")
        comparison = forward
        units = str(to_crs)
        mode = "expected"
    distances = np.linalg.norm(comparison[:, :2] - reference[:, :2], axis=1)
    return {
        "mode": mode,
        "count": int(distances.size),
        "mean": float(distances.mean()),
        "max": float(distances.max()),
        "rms": float(np.sqrt(np.mean(distances * distances))),
        "units": units,
        "source_crs": str(from_crs),
        "target_crs": str(to_crs),
    }


def alignment_report(scene_or_recipe: Any) -> dict[str, Any]:
    """Return deterministic CRS/geotransform alignment metadata for a scene."""

    recipe = getattr(scene_or_recipe, "recipe", scene_or_recipe)
    payload = recipe.to_dict() if hasattr(recipe, "to_dict") else dict(recipe)
    terrain = dict(payload.get("terrain") or {})
    target_crs = payload.get("target_crs") or terrain.get("crs")
    layers = []
    for index, layer in enumerate(payload.get("layers") or ()):
        if not isinstance(layer, Mapping):
            continue
        metadata = layer.get("metadata") if isinstance(layer.get("metadata"), Mapping) else {}
        source_crs = metadata.get("source_crs") or layer.get("crs")
        layer_target = layer.get("crs") or target_crs
        layers.append(
            {
                "layer_id": str(layer.get("layer_id") or _layer_id(layer, f"layer_{index}")),
                "kind": str(layer.get("kind", "layer")),
                "source_crs": source_crs,
                "target_crs": layer_target,
                "aligned": bool(target_crs and layer_target and _same_crs(str(layer_target), str(target_crs))),
                "transform_applied": bool(metadata.get("alignment_transform_applied")),
                "geotransform": metadata.get("geotransform") or metadata.get("transform"),
            }
        )
    return {
        "kind": "mapscene_alignment_report",
        "schema": "forge3d.mapscene.alignment.v1",
        "target_crs": target_crs,
        "terrain_crs": terrain.get("crs"),
        "terrain_geotransform": (terrain.get("metadata") or {}).get("geotransform") if isinstance(terrain.get("metadata"), Mapping) else None,
        "layers": layers,
        "hash": _stable_hash({"target_crs": target_crs, "terrain": terrain, "layers": layers}),
    }


__all__ = [
    "alignment_report",
    "alignment_residual",
    "reproject_dem_to_target",
    "resample_raster_to_grid",
    "transform_features",
    "transform_geometry",
]
