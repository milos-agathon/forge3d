"""C01 colour authoring in the existing terrain albedo path."""
import numpy as np


def physical_vector(layer):
    return (getattr(layer, "metadata", None) or {}).get("material_presentation") == "physical terrain albedo"


def blend_surface(base, rgba, valid, opacity=1):
    """Straight alpha source-over in linear sRGB, preserving source heights."""
    from .terrain_materials import _linear, _display
    from .raster_style import RasterStyleResult
    source = np.asarray(rgba, np.uint8)
    if base is None:
        destination = np.zeros_like(source)
        heights = None
    else:
        destination, heights = base.rgba, base.heights
    a = source[..., 3:4].astype(np.float32) / 255 * np.asarray(valid)[..., None] * opacity
    b = destination[..., 3:4].astype(np.float32) / 255
    alpha = a + b * (1-a)
    rgb = _linear(source[..., :3].astype(np.float32)/255)*a + _linear(destination[..., :3].astype(np.float32)/255)*b*(1-a)
    np.divide(rgb, alpha, out=rgb, where=alpha > 0)
    output = np.concatenate((_display(rgb), np.rint(alpha*255).astype(np.uint8)), axis=-1)
    return RasterStyleResult(output, alpha[..., 0] > 0, {} if base is None else base.legend, heights=heights)


def material_surface_stack(recipe, shape):
    from . import map_scene as ms
    from .terrain_materials import TerrainMaterialStyle
    base = (recipe.terrain.style.apply(ms._load_raw_heightmap(recipe.terrain))
            if isinstance(recipe.terrain.style, TerrainMaterialStyle) else None)
    for layer in recipe.layers:
        if isinstance(layer, ms.RasterOverlay):
            result = ms._styled_raster_result(layer)
            if result.rgba.shape[:2] != tuple(shape):
                raise ValueError("material raster must match the terrain grid")
            base = blend_surface(base, result.rgba, result.valid_mask, layer.opacity)
    return base


def drape_physical_vectors(recipe, shape, base):
    from . import map_scene as ms
    from ._map_scene_common import _same_crs
    from ._map_scene_render import _paint, _feature_number
    from .alignment import _require_rasterio, _affine_from_metadata
    from shapely.geometry import shape as geometry_shape
    layers = [layer for layer in recipe.layers if isinstance(layer, ms.VectorOverlay) and physical_vector(layer)]
    if not layers:
        return base
    _require_rasterio()
    from rasterio.features import rasterize
    grid = ms._terrain_alignment_grid(recipe.terrain, fallback_shape=shape)
    if grid is None:
        from ._map_scene_projection import material_style
        style = material_style(recipe)
        if style is not None:
            from rasterio.transform import Affine
            authored = style.grid
            sx, sy = authored.spacing
            transform = Affine(sx, 0, authored.origin[0] - sx/2,
                               0, -sy, authored.origin[1] + sy/2)
            grid = {'crs': authored.crs, 'transform': transform}
    if grid is None:
        raise ValueError("physical vector paint requires the terrain's projected grid")
    transform = _affine_from_metadata(grid['transform'])
    cell_metres = max(abs(transform.a), abs(transform.e))
    for layer in layers:
        if not _same_crs(layer.crs, grid['crs']):
            raise ValueError("physical vector paint requires the terrain CRS")
        for feature in layer.features or ():
            properties = feature.get('properties') or {}
            color = properties['forge3d:material']['rgba']
            geometry = geometry_shape(feature['geometry'])
            kind = 'fill' if 'Polygon' in geometry.geom_type else 'line' if 'Line' in geometry.geom_type else 'circle'
            paint = _paint(layer, kind)
            opacity = _feature_number(paint.get(kind+'-opacity', 1), properties, 1)
            if not np.isfinite(opacity) or not 0 <= opacity <= 1:
                raise ValueError("physical vector opacity must be finite in [0, 1]")
            if kind != 'fill':
                width = layer.width_world
                if width is None:
                    key = 'line-width' if kind == 'line' else 'circle-radius'
                    width = _feature_number(paint.get(key, layer.width_px or 1), properties, layer.width_px or 1) * cell_metres
                    if kind == 'circle':
                        width *= 2
                if not np.isfinite(width) or width <= 0:
                    raise ValueError("physical vector width must be positive ground metres")
                geometry = geometry.buffer(width/2)
            coverage = rasterize([(geometry, 1)], out_shape=tuple(shape), transform=transform,
                                 fill=0, all_touched=False, dtype='uint8').astype(bool)
            rgba = np.empty((*shape, 4), np.uint8)
            rgba[:] = color
            base = blend_surface(base, rgba, coverage, opacity)
    return base
