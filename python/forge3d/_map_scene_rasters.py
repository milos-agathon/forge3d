"""Raster source identity and terrain-UV preparation for MapScene."""
from __future__ import annotations
import hashlib
import io
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path


def raster_content_hashes(recipe):
    """Identify source bytes before cache lookup, including in-place edits."""
    from .map_scene import MapSceneNativeUnavailable, RasterOverlay
    from ._map_scene_validation import diagnostic_block
    identities = []
    for layer in recipe.layers:
        if not isinstance(layer, RasterOverlay):
            continue
        try:
            path = Path(layer.path or "")
            if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                raise ValueError("supported raster sources are PNG and GeoTIFF")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                    digest.update(chunk)
            identities.append({"layer_id": layer.layer_id, "sha256": digest.hexdigest()})
        except (OSError, ValueError) as exc:
            raise MapSceneNativeUnavailable([diagnostic_block(
                layer=str(layer.layer_id), reason=f"RasterOverlay source is unavailable: {exc}",
                required_native="readable RasterOverlay source",
            )]) from exc
    return identities


def _overlay_in_terrain_uv(layer, recipe, shape):
    import numpy as np
    from .map_scene import (_load_native_raster_overlay, _terrain_alignment_grid,
                            _terrain_scene_diagonal, _raster_to_rgba8)
    from ._map_scene_common import _metadata_dict, _same_crs
    path = Path(layer.path or '')
    metadata = _metadata_dict(layer.metadata)
    grid = _terrain_alignment_grid(recipe.terrain, target_crs=recipe.target_crs or recipe.terrain.crs,
                                   fallback_shape=shape)
    if path.suffix.lower() in {'.tif', '.tiff'}:
        if grid is None:
            raise ValueError('GeoTIFF draping requires a georeferenced terrain grid')
        from .alignment import _require_rasterio, _affine_from_metadata, _resampling_method
        rasterio = _require_rasterio()
        from rasterio.warp import reproject
        with rasterio.open(path) as source:
            if source.crs is None:
                raise ValueError('GeoTIFF draping requires source CRS metadata')
            raw = source.read()
            if not np.isfinite(raw).all():
                raise ValueError('raster source contains non-finite samples')
            rgba = _raster_to_rgba8(np.moveaxis(raw, 0, -1))
            # Preserve both nodata and source footprint even for opaque RGB input.
            rgba[..., 3] = np.minimum(rgba[..., 3], source.dataset_mask())
            output = np.zeros((*shape, 4), dtype=np.uint8)
            for band in range(4):
                reproject(rgba[..., band], output[..., band], src_transform=source.transform,
                          src_crs=source.crs, dst_transform=_affine_from_metadata(grid['transform']),
                          dst_crs=grid['crs'], src_nodata=None, dst_nodata=0,
                          resampling=_resampling_method(str(metadata.get('alignment_resampling') or metadata.get('resampling') or 'nearest')))
            return output
    if not _same_crs(layer.crs, recipe.target_crs or recipe.terrain.crs):
        raise ValueError('PNG draping requires the terrain CRS; use a georeferenced GeoTIFF for reprojection')
    rgba = _load_native_raster_overlay(layer)
    if rgba is None:
        raise ValueError('raster source is missing, unreadable, or unsupported')
    bounds = metadata.get('bounds')
    if bounds is None:
        if metadata.get('transform') is not None or metadata.get('geotransform') is not None:
            raise ValueError('PNG transform requires explicit bounds; use GeoTIFF for affine reprojection')
        from .map_scene import _resize_nearest_rgba
        return _resize_nearest_rgba(rgba, shape)
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
        raise ValueError('raster bounds must be finite [west, south, east, north] with positive extent')
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    if grid is not None:
        from .alignment import _affine_from_metadata
        transform = _affine_from_metadata(grid['transform'])
        east, north = transform * (xx + 0.5, yy + 0.5)
    else:
        span = max(1.0, _terrain_scene_diagonal(recipe.terrain))
        east = (xx / max(shape[1] - 1, 1) - 0.5) * span
        north = (0.5 - yy / max(shape[0] - 1, 1)) * span
    u = (east - bounds[0]) / (bounds[2] - bounds[0])
    v = (bounds[3] - north) / (bounds[3] - bounds[1])
    inside = (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)
    col = np.clip(np.floor(u * rgba.shape[1]).astype(np.int64), 0, rgba.shape[1] - 1)
    row = np.clip(np.floor(v * rgba.shape[0]).astype(np.int64), 0, rgba.shape[0] - 1)
    output = rgba[row, col].copy()
    output[~inside] = 0
    return output


@contextmanager
def terrain_raster_drape(recipe, heightmap):
    """Keep one ordered straight-alpha UV albedo map alive through native draw."""
    import numpy as np
    from .map_scene import MapSceneNativeUnavailable, RasterOverlay
    from ._map_scene_validation import diagnostic_block
    from ._native import get_native_module
    from .helpers.offscreen import save_png_deterministic
    layers = [layer for layer in recipe.layers if isinstance(layer, RasterOverlay)]
    if not layers:
        yield None, 0
        return
    raster_content_hashes(recipe)
    shape = heightmap.shape
    native = get_native_module()
    reserve = getattr(native, '_reserve_label_depth_host_allocation', None)
    if not callable(reserve):
        raise MapSceneNativeUnavailable([diagnostic_block(
            layer=str(layers[0].layer_id), reason='native host allocation tracking is unavailable',
            required_native='terrain-UV raster drape host allocation tracking')])
    # Reserve the simultaneous RGBA blend buffers and the affine/bounds grid
    # temporaries before allocating them. This uses the existing native host
    # budget, and derives bytes from array shapes/dtypes rather than a new cap.
    grid_bytes = int(np.prod(shape)) * (
        6 * 4 * np.dtype(np.float32).itemsize +  # source-over/rounding temporaries
        8 * np.dtype(np.float64).itemsize +    # XY, UV, indices, affine temporaries
        2 * 4 * np.dtype(np.uint8).itemsize + np.dtype(np.bool_).itemsize)
    reservation = reserve(grid_bytes, 'mapscene.terrain_uv_raster')
    try:
        accum = np.zeros((*shape, 4), dtype=np.float32)
        for layer in layers:
            try:
                # Read dimensions without decoding; account for the source
                # pixels as well as the destination grid before a large decode.
                path = Path(layer.path)
                if path.suffix.lower() == '.png':
                    # PNG's signature and IHDR dimensions occupy 24 bytes;
                    # use the format header without adding a Pillow dependency.
                    with path.open('rb') as header_file:
                        header = header_file.read(24)
                    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
                        raise ValueError('invalid PNG header')
                    width, height = struct.unpack('>II', header[16:24])
                    if not width or not height:
                        raise ValueError('PNG dimensions must be positive')
                    source_bytes = width * height * 4 * (
                        2 * np.dtype(np.uint16).itemsize + 2 * np.dtype(np.uint8).itemsize) + height
                else:
                    from .alignment import _require_rasterio
                    with _require_rasterio().open(path) as header:
                        source_bytes = header.width * header.height * (
                            sum(np.dtype(dtype).itemsize for dtype in header.dtypes) +
                            4 * header.count * np.dtype(np.float32).itemsize + 2 * 4)
                source_reservation = reserve(int(source_bytes), 'mapscene.terrain_uv_raster.source')
                try:
                    source = _overlay_in_terrain_uv(layer, recipe, shape).astype(np.float32) / 255.0
                finally:
                    source_reservation.close()
                opacity = float(layer.opacity)
                if not np.isfinite(opacity):
                    raise ValueError('raster opacity must be finite')
                alpha = source[..., 3:4] * np.clip(opacity, 0.0, 1.0)
                accum[..., :3] = source[..., :3] * alpha + accum[..., :3] * (1.0 - alpha)
                accum[..., 3:4] = alpha + accum[..., 3:4] * (1.0 - alpha)
            except Exception as exc:
                raise MapSceneNativeUnavailable([diagnostic_block(
                    layer=str(layer.layer_id), reason=f'RasterOverlay cannot be aligned to terrain UV: {exc}',
                    required_native='readable, aligned RasterOverlay source')]) from exc
        np.divide(accum[..., :3], accum[..., 3:4], out=accum[..., :3], where=accum[..., 3:4] > 0.0)
        rgba = np.clip(np.rint(accum * 255.0), 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory(prefix='forge3d-terrain-drape-') as directory:
            path = Path(directory) / 'albedo.png'
            save_png_deterministic(path, rgba)
            yield str(path), len(layers)
    finally:
        reservation.close()
