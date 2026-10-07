"""Compile-time MapScene projection using the terrain renderer's native camera."""
from __future__ import annotations
import copy
from collections.abc import Mapping
import math
import numpy as np


class TerrainPixel(tuple):
    """A projected pixel coordinate; bypass the legacy screen-coordinate heuristic."""
    terrain_projected = True


class TerrainProjector:
    projection_authority = 'deterministic'

    def __init__(self, recipe, *, layer_id='terrain', crs=None):
        from . import map_scene as ms
        from ._map_scene_common import _metadata_dict, _same_crs
        if recipe.output is None:
            raise ValueError('world projection requires an output viewport')
        self.recipe = recipe
        self.layer_id = str(layer_id)
        target_crs = recipe.target_crs or recipe.terrain.crs
        if crs is not None and not _same_crs(crs, target_crs):
            raise ValueError('world overlay CRS must match the compiled terrain CRS')
        self.heightmap = ms._load_native_heightmap(recipe.terrain)
        if self.heightmap is None:
            raise ValueError('world projection requires a terrain heightmap')
        self.heightmap, self.nodata_height_below = ms._mapscene_nodata_heightmap(self.heightmap)
        self.heightmap = np.asarray(self.heightmap, dtype=np.float32)
        self.params = ms._build_mapscene_terrain_params(
            recipe, self.heightmap, (max(64, recipe.output.width), max(64, recipe.output.height)))
        if self.params is None or not callable(getattr(self.params, 'project_terrain_points', None)):
            raise ValueError('native terrain camera projection is unavailable; rebuild the bindings')
        self.span = max(1.0, ms._terrain_scene_diagonal(recipe.terrain))
        self.grid = ms._terrain_alignment_grid(recipe.terrain, target_crs=target_crs,
                                              fallback_shape=self.heightmap.shape)
        self.bounds = _metadata_dict(recipe.terrain.metadata).get('bounds')
        self._inverse = None
        if self.grid is not None:
            from .alignment import _affine_from_metadata
            self._inverse = ~_affine_from_metadata(self.grid['transform'])

    def uv(self, point):
        if isinstance(point, (str, bytes)) or not hasattr(point, '__len__') or len(point) < 2:
            raise ValueError('world overlay anchors require east and north coordinates')
        east, north = float(point[0]), float(point[1])
        if self._inverse is not None:
            col, row = self._inverse * (east, north)
            u = (col - 0.5) / max(self.heightmap.shape[1] - 1, 1)
            v = (row - 0.5) / max(self.heightmap.shape[0] - 1, 1)
        elif self.bounds is not None:
            west, south, east_edge, north_edge = map(float, self.bounds)
            if not east_edge > west or not north_edge > south:
                raise ValueError('terrain bounds must have positive extent')
            u, v = (east - west) / (east_edge - west), (north_edge - north) / (north_edge - south)
        else:
            u, v = east / self.span + 0.5, 0.5 - north / self.span
        if not all(math.isfinite(value) for value in (u, v)):
            raise ValueError('world overlay coordinates must be finite')
        if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
            raise ValueError('world overlay anchor is outside the terrain footprint')
        return u, v

    def height(self, uv):
        x, y = uv[0] * (self.heightmap.shape[1] - 1), uv[1] * (self.heightmap.shape[0] - 1)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.heightmap.shape[1] - 1), min(y0 + 1, self.heightmap.shape[0] - 1)
        tx, ty = x - x0, y - y0
        top = float(self.heightmap[y0, x0]) * (1.0 - tx) + float(self.heightmap[y0, x1]) * tx
        bottom = float(self.heightmap[y1, x0]) * (1.0 - tx) + float(self.heightmap[y1, x1]) * tx
        return top * (1.0 - ty) + bottom * ty

    def project(self, point, *, viewport=None):
        uv = self.uv(point)
        height = self.height(uv)
        offset = float(point[2]) - height if len(point) > 2 else 0.0
        if not math.isfinite(offset):
            raise ValueError('world overlay elevation must be finite')
        value = self.params.project_terrain_points([[uv[0], uv[1], height, offset]])[0]
        if value is None:
            raise ValueError('world overlay anchor is behind the terrain camera')
        if not 0.0 <= value[2] <= 1.0:
            raise ValueError('world overlay anchor is outside the terrain camera clip planes')
        # Native renders at least 64px; map back to the requested output viewport.
        sx = self.recipe.output.width / self.params.size_px[0]
        sy = self.recipe.output.height / self.params.size_px[1]
        return TerrainPixel((value[0] * sx, value[1] * sy, value[2]))

    def world_path(self, points):
        result = []
        for start, end in zip(points, points[1:]):
            uv_a, uv_b = self.uv(start), self.uv(end)
            cells = max(abs(uv_b[0]-uv_a[0])*(self.heightmap.shape[1]-1),
                        abs(uv_b[1]-uv_a[1])*(self.heightmap.shape[0]-1))
            segments = max(1, math.ceil(cells))
            dims = 3 if len(start)>2 or len(end)>2 else 2
            a, b = list(start[:dims]), list(end[:dims])
            if dims == 3:
                if len(a) == 2:
                    a.append(self.height(uv_a))
                if len(b) == 2:
                    b.append(self.height(uv_b))
            for index in range(segments):
                t = index / segments
                result.append([float(a[i])*(1-t)+float(b[i])*t for i in range(dims)])
        if points:
            result.append(points[-1])
        return result

    def path(self, points):
        return [self.project(point) for point in self.world_path(points)]

    @staticmethod
    def first_point(geometry):
        coords = geometry['coordinates']
        while coords and hasattr(coords[0], '__len__'):
            coords = coords[0]
        return coords

    def stroke_width(self, start, end, width):
        if end is None:
            dx, dy = 0.0, 1.0
            center = list(start)
        else:
            dx, dy = float(end[0])-float(start[0]), float(end[1])-float(start[1])
            center = [(float(a)+float(b))*0.5 for a,b in zip(start,end)]
        length = math.hypot(dx,dy)
        if length == 0.0:
            dx,dy,length = 0.0,1.0,1.0
        normal = (-dy/length*float(width)*0.5, dx/length*float(width)*0.5)
        # Stroke width is horizontal world distance, at the centre's elevation.
        ground = float(center[2]) if len(center)>2 else self.height(self.uv(center))
        left = self.project([center[0]+normal[0],center[1]+normal[1],ground])
        right = self.project([center[0]-normal[0],center[1]-normal[1],ground])
        return math.hypot(right[0]-left[0],right[1]-left[1])

    def depth_image(self):
        """Deterministic DEM-grid triangle depth proxy, compiled without a GPU frame."""
        from ._native import get_native_module
        native = get_native_module()
        reserve = getattr(native, '_reserve_label_depth_host_allocation', None)
        rasterize = getattr(self.params, 'project_terrain_depth', None)
        if not callable(reserve) or not callable(rasterize):
            raise RuntimeError('native terrain depth projection is unavailable; rebuild the bindings')
        width, height = self.recipe.output.width, self.recipe.output.height
        # Native working memory: two rows of Option<[f64;3]> and the depth image.
        reservation = reserve(int(width * height * 4 + self.heightmap.shape[1] * 2 * 32),
                              'mapscene.camera_depth_proxy')
        try:
            return rasterize(self.heightmap, (width, height), self.nodata_height_below)
        finally:
            reservation.close()



def project_vector_recipe(recipe, *, projector=None):
    from . import map_scene as ms
    if not ms._is_3d_camera_mode(ms._mapscene_effective_camera_mode(recipe)):
        return recipe
    projected = copy.copy(recipe)
    layers = []
    for layer in recipe.layers:
        if not isinstance(layer, ms.VectorOverlay):
            layers.append(layer)
            continue
        try:
            if not layer.features:
                raise ValueError('world vector projection requires inline GeoJSON features')
            from ._map_scene_common import _same_crs
            if layer.crs is not None and not _same_crs(layer.crs, recipe.target_crs or recipe.terrain.crs):
                raise ValueError('world overlay CRS must match the compiled terrain CRS')
            if layer.width_world is not None and (not math.isfinite(float(layer.width_world)) or layer.width_world <= 0):
                raise ValueError('world stroke width must be finite and positive')
            projector = projector or TerrainProjector(recipe, layer_id=layer.layer_id, crs=layer.crs)
            clone = copy.copy(layer)
            clone.features = copy.deepcopy(layer.features)
            for feature in clone.features or ():
                if not isinstance(feature, Mapping) or not isinstance(feature.get('geometry'), Mapping):
                    raise ValueError('world vectors require GeoJSON feature geometries')
                geometry = feature['geometry']
                kind, coords = geometry.get('type', ''), geometry.get('coordinates', [])
                paths = ([coords] if kind == 'LineString' else coords if kind in ('MultiLineString','Polygon')
                         else [ring for polygon in coords for ring in polygon] if kind == 'MultiPolygon' else [])
                minimum = 4 if 'Polygon' in kind else 2
                if kind != 'Point' and not coords:
                    raise ValueError('world vector geometry must not be empty')
                if any(len(path)<minimum for path in paths):
                    raise ValueError('world vector geometry has too few coordinates')
                if kind == 'Polygon':
                    feature['_terrain_rings_uv'] = [[projector.uv(point) for point in ring] for ring in coords]
                elif kind == 'MultiPolygon':
                    feature['_terrain_polygons_uv'] = [
                        [[projector.uv(point) for point in ring] for ring in polygon] for polygon in coords]
                if kind == 'Point':
                    geometry['coordinates'] = projector.project(coords)
                elif kind == 'MultiPoint':
                    geometry['coordinates'] = [projector.project(point) for point in coords]
                elif kind == 'LineString':
                    geometry['coordinates'] = projector.path(coords)
                elif kind in ('MultiLineString', 'Polygon'):
                    geometry['coordinates'] = [projector.path(path) for path in coords]
                elif kind == 'MultiPolygon':
                    geometry['coordinates'] = [[projector.path(path) for path in polygon] for polygon in coords]
                else:
                    raise ValueError(f'unsupported world vector geometry {kind!r}')
            expanded = []
            for feature in clone.features or ():
                geometry = feature.get('geometry', {})
                kind, coords = geometry.get('type'), geometry.get('coordinates', [])
                if kind in ('MultiPoint', 'MultiLineString'):
                    for index, primitive in enumerate(coords):
                        item = dict(feature)
                        item['id'] = str(feature.get('id', layer.layer_id)) + ':' + str(index)
                        item['geometry'] = dict(geometry, type='Point' if kind == 'MultiPoint' else 'LineString', coordinates=primitive)
                        expanded.append(item)
                else:
                    expanded.append(feature)
            if layer.width_world is not None and layer.width_px is None:
                widened = []
                def append_stroke(feature, source, projected_path):
                    world_path = projector.world_path(source)
                    for index,(a,b) in enumerate(zip(projected_path,projected_path[1:])):
                        item = copy.copy(feature)
                        item['geometry'] = dict(type='LineString',coordinates=[a,b])
                        item['_projected_width_px'] = projector.stroke_width(
                            world_path[index],world_path[index+1],layer.width_world)
                        widened.append(item)
                for original, feature in zip(layer.features, clone.features):
                    geometry = feature['geometry']
                    kind,coords = geometry['type'],geometry['coordinates']
                    source = original['geometry']['coordinates']
                    if kind == 'LineString':
                        append_stroke(feature,source,coords)
                    elif kind == 'MultiLineString':
                        for world_path,screen_path in zip(source,coords):
                            append_stroke(feature,world_path,screen_path)
                    elif kind in ('Polygon','MultiPolygon'):
                        feature['_terrain_fill_only'] = True
                        widened.append(feature)
                        world_polygons = [source] if kind == 'Polygon' else source
                        screen_polygons = [coords] if kind == 'Polygon' else coords
                        for world_polygon,screen_polygon in zip(world_polygons,screen_polygons):
                            for world_ring,screen_ring in zip(world_polygon,screen_polygon):
                                outline = copy.copy(feature)
                                outline.pop('_terrain_fill_only',None)
                                append_stroke(outline,world_ring,screen_ring)
                    elif kind in ('Point','MultiPoint'):
                        world_points = [source] if kind == 'Point' else source
                        screen_points = [coords] if kind == 'Point' else coords
                        for world_point,screen_point in zip(world_points,screen_points):
                            item = copy.copy(feature)
                            item['geometry'] = dict(type='Point',coordinates=screen_point)
                            item['_projected_width_px'] = projector.stroke_width(world_point,None,layer.width_world)
                            widened.append(item)
                clone.features = widened
                clone.width_world = None
            else:
                clone.features = expanded
            layers.append(clone)
        except (ValueError, TypeError, RuntimeError) as exc:
            from ._map_scene_validation import diagnostic_block
            raise ms.MapSceneNativeUnavailable([diagnostic_block(
                layer=str(layer.layer_id), reason=str(exc),
                required_native='vector projection through the terrain camera')]) from exc
    projected.layers = tuple(layers)
    return projected


def composite_projected_vectors(base, projected_recipe, recipe):
    """Depth-test both vector compositors before any changed pixel is retained.

    Strokes interpolate their projected device depth. Polygon fills use the
    frontmost terrain UV, so a hidden footprint cannot paint an occluding ridge.
    """
    from . import map_scene as ms
    from ._map_scene_render import _ring_contains
    from ._native import get_native_module
    layers = [layer for layer in projected_recipe.layers if isinstance(layer, ms.VectorOverlay)]
    if not layers:
        return base, False
    projector = TerrainProjector(recipe)
    height, width = base.shape[:2]
    # Depth and UV remain covered by the ledger for the compositor's lifetime.
    reservation = get_native_module()._reserve_label_depth_host_allocation(
        width * height * (4 + 8), 'mapscene.vector_visibility')
    try:
        depth = projector.depth_image()
        surface_uv = None
        composited = False
        for layer in layers:
            for feature in layer.features or ():
                one_layer = copy.copy(layer)
                one_layer.features = (feature,)
                one_recipe = copy.copy(projected_recipe)
                one_recipe.layers = (one_layer,)
                painted, drawn = ms._composite_native_vector_layers(base, one_recipe)
                if not drawn:
                    return base, False
                composited = True
                yy, xx = np.nonzero(np.any(painted != base, axis=2))
                if not len(xx):
                    continue
                geometry = feature['geometry']
                kind, coords = geometry['type'], geometry['coordinates']
                visible = np.zeros(len(xx), dtype=bool)
                if kind in ('Polygon', 'MultiPolygon'):
                    if surface_uv is None:
                        surface_uv = projector.params.unproject_terrain_depth(depth)
                    uv = surface_uv[yy, xx]
                    polygons = (feature.get('_terrain_polygons_uv') if kind == 'MultiPolygon'
                                else [feature['_terrain_rings_uv']])
                    for rings in polygons:
                        inside = _ring_contains(rings[0], uv[:, 0], uv[:, 1])
                        for ring in rings[1:]:
                            inside &= ~_ring_contains(ring, uv[:, 0], uv[:, 1])
                        visible |= inside
                    paths = [ring for polygon in ms._render_geometry_polygon_rings(geometry) for ring in polygon]
                else:
                    paths = [[coords]] if kind == 'Point' else [coords]
                # For pixels on a stroke/cap/join choose the nearest projected
                # segment's depth. Keeping NDC depth until here closes the old
                # _pixel_to_ndc information loss for both drawing backends.
                distance = np.full(len(xx), np.inf)
                stroke_depth = np.ones(len(xx))
                px, py = xx + 0.5, yy + 0.5
                for path in paths:
                    segments = zip(path, path[1:]) if len(path) > 1 else [(path[0], path[0])]
                    for a, b in segments:
                        dx, dy = b[0]-a[0], b[1]-a[1]
                        length2 = dx*dx + dy*dy
                        t = np.clip(((px-a[0])*dx+(py-a[1])*dy)/length2, 0, 1) if length2 else np.zeros(len(xx))
                        d = (px-a[0]-t*dx)**2 + (py-a[1]-t*dy)**2
                        nearest = d < distance
                        distance[nearest] = d[nearest]
                        stroke_depth[nearest] = a[2]+t[nearest]*(b[2]-a[2])
                stroke_visible = stroke_depth <= np.nextafter(depth[yy, xx], np.float32(np.inf))
                if kind in ('Polygon', 'MultiPolygon'):
                    # Fills are grounded by frontmost UV; outlines may extend
                    # outside that footprint by their declared pixel width.
                    line_width = feature.get('_projected_width_px', ms._render_resolve_line_width_px(
                        layer, ms._render_paint(layer, 'line'), recipe, width, height))
                    if not feature.get('_terrain_fill_only'):
                        visible |= stroke_visible & (distance <= (line_width * 0.5)**2)
                else:
                    visible = stroke_visible
                painted[yy[~visible], xx[~visible]] = base[yy[~visible], xx[~visible]]
                base = painted
        return base, composited
    finally:
        reservation.close()
