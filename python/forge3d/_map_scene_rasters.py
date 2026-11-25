"""Raster source identity and terrain-UV preparation for MapScene."""
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path


def raster_content_hashes(recipe):
    """Identify source bytes before cache lookup, including in-place edits."""
    from .map_scene import MapSceneNativeUnavailable, RasterOverlay
    from ._map_scene_validation import diagnostic_block
    from .raster_style import BivariateRasterStyle, raster_values_to_dict
    from .terrain_materials import TerrainMaterialStyle
    from .geospatial_materials import MaterialRasterStyle
    identities = []
    for layer in (recipe.terrain, *recipe.layers):
        if not isinstance(layer, RasterOverlay) and not isinstance(getattr(layer, "style", None), TerrainMaterialStyle):
            continue
        sources = [("primary", layer.path, layer.data)]
        if isinstance(layer.style, (TerrainMaterialStyle, MaterialRasterStyle)):
            sources.extend((f"texture:{material.name}:{texture.channel}", texture.resolved_path, None)
                           for material in layer.style.recipe.layers for texture in material.textures())
        if isinstance(layer.style, BivariateRasterStyle):
            sources.append(("secondary", layer.secondary_path, layer.secondary_data))
        for role, source_path, data in sources:
            try:
                digest = hashlib.sha256()
                if data is not None:
                    payload = raster_values_to_dict(data)
                    digest.update(json.dumps(payload, sort_keys=True, allow_nan=False,
                                             separators=(",", ":")).encode("utf-8"))
                else:
                    path = Path(source_path or "")
                    if layer.style is None and path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                        raise ValueError("supported unstyled raster sources are PNG and GeoTIFF")
                    with path.open("rb") as source:
                        while chunk := source.read(io.DEFAULT_BUFFER_SIZE):
                            digest.update(chunk)
                identity = {"layer_id": getattr(layer, "layer_id", "terrain-material"), "sha256": digest.hexdigest()}
                expected = (layer.metadata or {}).get("observation_snapshot_sha256")
                if expected is not None and identity["sha256"] != expected:
                    raise ValueError("compiled observation raster snapshot was modified")
                if layer.style is not None:
                    identity["source"] = role
                if role == "primary" and isinstance(layer.style, MaterialRasterStyle) and layer.style.recipe.has_surface_channels:
                    # Cached frames made before ordered channel composition may
                    # have omitted the raster maps, even for identical inputs.
                    identity["channel_composition"] = 1
                identities.append(identity)
            except (OSError, ValueError, TypeError) as exc:
                source_name = f" {role}" if layer.style is not None else ""
                raise MapSceneNativeUnavailable([diagnostic_block(
                    layer=str(getattr(layer, "layer_id", "terrain-material")), reason=f"RasterOverlay{source_name} source is unavailable: {exc}",
                    required_native="readable RasterOverlay source",
                )]) from exc
    return identities


def raster_route_blocks(recipe):
    """Reject raster stacks whose order the single thematic pass cannot preserve."""
    from .map_scene import RasterOverlay
    from ._map_scene_validation import diagnostic_block
    layers = [layer for layer in recipe.layers if isinstance(layer, RasterOverlay)]
    styled = [layer for layer in layers if layer.style is not None]
    from .terrain_materials import TerrainMaterialStyle
    from .geospatial_materials import MaterialRasterStyle
    from .raster_style import CategoricalRasterStyle
    if (styled and len(styled) == len(layers) and
        any(isinstance(layer.style, MaterialRasterStyle) for layer in styled) and
        all(isinstance(layer.style, (TerrainMaterialStyle, MaterialRasterStyle, CategoricalRasterStyle)) for layer in styled)):
        return []  # Explicit C01 aligned, ordered colour stack.
    if len(styled) > 1 or (styled and len(styled) != len(layers)):
        return [diagnostic_block(
            layer=str(layer.layer_id),
            reason="native terrain supports one styled RasterOverlay or an ordered unstyled raster stack; "
                   "mixed styled/unstyled overlays cannot preserve recipe order",
            required_native="separate thematic and terrain-UV raster passes",
        ) for layer in layers]
    return []


def _overlay_in_terrain_uv(layer, recipe, shape):
    import numpy as np
    from .map_scene import (_load_native_raster_overlay, _terrain_alignment_grid,
                            _terrain_scene_diagonal, _raster_to_rgba8)
    from ._map_scene_common import _metadata_dict, _same_crs
    from .raster_registration import (
        geotiff_registration, pixel_registration, require_compatible_registration,
    )
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
        expected = metadata.get("observation_snapshot_sha256")
        with contextlib.ExitStack() as stack:
            if expected is None:
                source = stack.enter_context(rasterio.open(path))
            else:
                # The compiler's single-file snapshot excludes external masks
                # and PAM overrides. Verify the exact bytes being decoded.
                payload = path.read_bytes()
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError("compiled observation raster snapshot was modified")
                memory = stack.enter_context(rasterio.MemoryFile(payload))
                source = stack.enter_context(memory.open())
            if source.crs is None:
                raise ValueError('GeoTIFF draping requires source CRS metadata')
            source_registration = geotiff_registration(source)
            if "registration" in metadata and pixel_registration(metadata["registration"]) != source_registration:
                raise ValueError("GeoTIFF drape registration differs from its declaration")
            require_compatible_registration(
                source_registration, grid["registration"], metadata.get("registration_conversion"),
            )
            raw = source.read()
            valid = np.min(source.read_masks(), axis=0) > 0
            if not np.isfinite(raw[:, valid]).all():
                raise ValueError('raster source contains non-finite valid samples')
            rgba = _raster_to_rgba8(np.moveaxis(raw, 0, -1), valid)
            # Preserve both nodata and source footprint even for opaque RGB input.
            rgba[..., 3] = np.minimum(rgba[..., 3], source.dataset_mask())
            output = np.zeros((*shape, 4), dtype=np.uint8)
            # GDAL can move a valid zero byte to 1 when it coincides with the
            # destination nodata value. Warp into a floating-point band so its
            # nodata adjustment stays below one byte, then quantize explicitly.
            band_output = np.zeros(shape, dtype=np.float32)
            for band in range(4):
                band_output.fill(0)
                reproject(rgba[..., band], band_output, src_transform=source.transform,
                          src_crs=source.crs, dst_transform=_affine_from_metadata(grid['transform']),
                          dst_crs=grid['crs'], src_nodata=None, dst_nodata=0,
                          resampling=_resampling_method(str(metadata.get('alignment_resampling') or metadata.get('resampling') or 'nearest')))
                output[..., band] = np.clip(np.rint(band_output), 0, 255).astype(np.uint8)
            return output
    if not _same_crs(layer.crs, recipe.target_crs or recipe.terrain.crs):
        raise ValueError('PNG draping requires the terrain CRS; use a georeferenced GeoTIFF for reprojection')
    if grid is not None:
        require_compatible_registration(
            pixel_registration(metadata.get("registration", "area")),
            grid["registration"], metadata.get("registration_conversion"),
        )
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
    blocks = raster_route_blocks(recipe)
    if blocks:
        raise MapSceneNativeUnavailable(blocks)
    layers = [layer for layer in recipe.layers if isinstance(layer, RasterOverlay) and layer.style is None]
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
        np.dtype(np.float32).itemsize +        # GeoTIFF reprojection band
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


@contextmanager
def material_surface_maps(recipe):
    """Keep recipe-composed data images alive through the existing native draw."""
    from dataclasses import replace
    from collections.abc import Mapping
    import numpy as np
    from .raster_style import _single
    from .terrain_materials import TerrainMaterialStyle, _channel_rgba
    from .geospatial_materials import MaterialRasterStyle
    from .map_scene import RasterOverlay, _load_raw_heightmap, _thematic_source
    from .helpers.offscreen import save_png_deterministic
    ground = recipe.terrain.style
    ground = ground if isinstance(ground, TerrainMaterialStyle) else None
    rasters = [layer for layer in recipe.layers if isinstance(layer, RasterOverlay)
               and isinstance(layer.style, MaterialRasterStyle)]
    if ground is None and not rasters:
        yield recipe
        return
    channel_rasters = [layer for layer in rasters if layer.style.recipe.has_surface_channels]
    metadata = dict(recipe.terrain.metadata or {})
    legacy = metadata.get("material_maps") or metadata.get("materials")
    ground_has_channels = ground is not None and ground.recipe.has_surface_channels
    if legacy and (ground_has_channels or channel_rasters):
        raise ValueError("recipe surface channels and legacy material_maps are competing sources")

    channels = {}
    if ground_has_channels or channel_rasters:
        heights = _load_raw_heightmap(recipe.terrain)
    if ground_has_channels:
        prepared = ground.prepare(heights)
        # The base terrain map already supplies the native default over its grid.
        for name, data in (("normal", prepared.normals), ("roughness", prepared.roughness)):
            if data is not None:
                channels[name] = data, np.ones(data.shape[:2], dtype=bool)
    if channel_rasters:
        grid = ground.grid if ground is not None else rasters[0].style.grid
        for layer in channel_rasters:
            style = layer.style
            values = _single(_thematic_source(layer.data, layer.path))
            if style.grid != grid or values.shape != heights.shape:
                raise ValueError(f"raster {layer.layer_id!r} surface channels must match the terrain grid")
            valid = np.isfinite(values)
            if style.elevations is not None:
                valid &= np.isfinite(style.elevations)
            if not valid.any():
                continue
            normals, roughness = style.channels(values)
            for name, data in (("normal", normals), ("roughness", roughness)):
                if data is None:
                    continue
                if name in channels:
                    under, under_valid = channels[name]
                    selected = valid[..., None] if data.ndim == 3 else valid
                    data = np.where(selected, data, under)
                    channels[name] = data, valid | under_valid
                else:
                    channels[name] = data, valid

    masks = [valid for _, valid in channels.values()]
    mask = masks[0] if masks else None
    if mask is not None and any(not np.array_equal(mask, other) for other in masks[1:]):
        raise ValueError("normal and roughness channels have different coverage; the shared native material mask requires equal coverage")

    with tempfile.TemporaryDirectory(prefix="forge3d-material-channels-") as directory:
        maps = dict(legacy) if isinstance(legacy, Mapping) else {}
        maps.setdefault("physical_normals", True)
        for name, (data, _) in channels.items():
            if name == "normal":
                # Authoring/export: +X east, OpenGL +G north, +Z outward.
                # build_tbn in terrain_pbr_pom uses a south-facing bitangent.
                # Convert only at this native boundary, leaving exchange intact.
                data = data.copy()
                data[..., 1] *= -1
            rgba = _channel_rgba(name, data)
            path = Path(directory) / (name + ".png")
            save_png_deterministic(path, rgba)
            maps[name + "_path"] = str(path)
        if mask is not None and not mask.all():
            path = Path(directory) / "mask.png"
            save_png_deterministic(path, _channel_rgba("roughness", mask))
            maps["mask_path"] = str(path)
        metadata["material_maps"] = maps
        yield replace(recipe, terrain=replace(recipe.terrain, metadata=metadata))
