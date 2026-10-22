"""Immutable cartographic inputs for the fixed native frame.

The CPU authors coverage and projected device depth from recipe geometry, as
it authors glyph atlases. It never consumes a rendered image. Occlusion and
blending against the live shared depth/color happen in OverlayRenderer on GPU.
"""
from __future__ import annotations

import copy
import numpy as np


def vector_batches(recipe, projected):
    from . import map_scene as ms
    from ._map_scene_projection import TerrainProjector
    from ._map_scene_render import _ring_contains

    width, height = recipe.output.width, recipe.output.height
    layers = [layer for layer in projected.layers if isinstance(layer, ms.VectorOverlay)]
    if not layers:
        return []
    projector = TerrainProjector(recipe)
    terrain_depth = None
    surface_uv = None
    terrain_dx = terrain_dy = None
    batches = []
    for layer in layers:
        style_payload = copy.deepcopy(layer.to_dict())
        for value in style_payload.get('features') or ():
            value.pop('_terrain_draped', None)
        fallback_rgb = ms._render_rgb(style_payload, salt='vector')
        parts = []
        for feature in layer.features or ():
            if (feature['geometry']['type'] in ('Polygon', 'MultiPolygon')
                    and not feature.get('_terrain_fill_only') and not feature.get('_terrain_stroke_only')):
                # Fill and outline can have different visible depths. A hidden
                # outline must reveal its fill, not erase the whole pixel.
                parts.extend((dict(feature, _terrain_fill_only=True),
                              dict(feature, _terrain_stroke_only=True)))
            else:
                parts.append(feature)
        for feature in parts:
            one_layer = copy.copy(layer)
            one_layer.features = (feature,)
            one_recipe = copy.copy(projected)
            one_recipe.layers = (one_layer,)
            rgba, drawn = ms._composite_python_precise_vector_layers(
                np.zeros((height, width, 4), dtype=np.uint8), one_recipe, [one_layer],
                clip_to_geometry=True, fallback_rgb=fallback_rgb)
            if not drawn:
                raise ValueError(f'vector {layer.layer_id} has no authored coverage')
            if not rgba[..., 3].any():
                continue
            if projected is not recipe and terrain_depth is None:
                terrain_depth = projector.depth_image(include_buildings=False)
                dx = np.abs(np.diff(terrain_depth, axis=1))
                dy = np.abs(np.diff(terrain_depth, axis=0))
                terrain_dx = np.maximum(np.pad(dx, ((0,0),(0,1))), np.pad(dx, ((0,0),(1,0))))
                terrain_dy = np.maximum(np.pad(dy, ((0,1),(0,0))), np.pad(dy, ((1,0),(0,0))))
            depth = (terrain_depth.copy() if projected is not recipe
                     else np.ones((height, width), dtype=np.float32))
            yy, xx = np.nonzero(rgba[..., 3])
            if len(xx) and projected is not recipe:
                geometry = feature['geometry']
                kind, coords = geometry['type'], geometry['coordinates']
                distance = np.full(len(xx), np.inf)
                offset_x = np.zeros(len(xx))
                offset_y = np.zeros(len(xx))
                stroke_depth = np.ones(len(xx))
                if kind in ('Polygon', 'MultiPolygon'):
                    if surface_uv is None:
                        surface_uv = projector.params.unproject_terrain_depth(
                            terrain_depth, shared_frame=projector.shared_frame)
                    uv = surface_uv[yy, xx]
                    polygons = (feature.get('_terrain_polygons_uv') if kind == 'MultiPolygon'
                                else [feature['_terrain_rings_uv']])
                    inside = np.zeros(len(xx), dtype=bool)
                    for rings in polygons:
                        mask = _ring_contains(rings[0], uv[:, 0], uv[:, 1])
                        for ring in rings[1:]:
                            mask &= ~_ring_contains(ring, uv[:, 0], uv[:, 1])
                        inside |= mask
                    if np.ptp(projector.heightmap) == 0.0:
                        # A plane maps polygon rings by an exact homography.
                        # Keep the already authored coverage (including holes);
                        # inverse projection through f32 depth otherwise drops
                        # edge pixels solely because of depth quantization.
                        inside[:] = True
                    depth[yy, xx] = terrain_depth[yy, xx]
                    paths = [ring for polygon in ms._render_geometry_polygon_rings(geometry) for ring in polygon]
                else:
                    inside = np.ones(len(xx), dtype=bool)
                    paths = [[coords]] if kind == 'Point' else [coords]
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
                        offset_x[nearest] = (px-a[0]-t*dx)[nearest]
                        offset_y[nearest] = (py-a[1]-t*dy)[nearest]
                        stroke_depth[nearest] = a[2]+t[nearest]*(b[2]-a[2])
                if kind in ('Polygon', 'MultiPolygon'):
                    line_width = feature.get('_projected_width_px', ms._render_resolve_line_width_px(
                        layer, ms._render_paint(layer, 'line'), recipe, width, height))
                    stroke = (distance <= (line_width * 0.5)**2) & (not bool(feature.get('_terrain_fill_only')))
                    if feature.get('_terrain_stroke_only'):
                        # Raw raster coverage already includes caps and joins.
                        stroke[:] = True
                        inside[:] = True
                    depth[yy[stroke], xx[stroke]] = stroke_depth[stroke]
                    inside |= stroke
                    rgba[yy[~inside], xx[~inside]] = 0
                else:
                    depth[yy, xx] = stroke_depth
                if feature.get('_terrain_draped'):
                    # The footprint bias comes only from terrain. A building
                    # silhouette cannot enlarge it and expose a drape through
                    # the building. Live GPU surface identity handles walls.
                    bias = (terrain_dx[yy, xx]*np.abs(offset_x)
                            + terrain_dy[yy, xx]*np.abs(offset_y) + 4*np.finfo(np.float32).eps)
                    hidden = depth[yy, xx] > terrain_depth[yy, xx] + bias
                    rgba[yy[hidden], xx[hidden]] = 0
            batches.append({'rgba': np.ascontiguousarray(rgba), 'depth': np.ascontiguousarray(depth),
                            'depth_test': projected is not recipe, 'layer': str(layer.layer_id),
                            'terrain_draped': bool(feature.get('_terrain_draped'))})
    return batches


def furniture_batches(recipe):
    from . import map_scene as ms
    if recipe.map_furniture is None:
        return []
    rgba = np.zeros((recipe.output.height, recipe.output.width, 4), dtype=np.uint8)
    ms._compose_furniture(rgba, recipe)
    return [{'rgba': rgba, 'depth_test': False, 'layer': 'map_furniture'}]


def building_world_triangles(projector):
    """Match scatter::pack_instance_transforms, including its local axis frame."""
    from . import map_scene as ms
    batches = ms._terrain_scatter_building_batches_for_recipe(projector.recipe, projector.heightmap,
        shared_frame=True)
    if batches is None:
        return []
    width = float(max(projector.heightmap.shape))
    scale = np.float32(projector.span / width)
    span = np.float32(projector.span)
    z_scale = np.float32(projector.params.z_scale)
    valid = projector.heightmap if projector.nodata_height_below is None else projector.heightmap[projector.heightmap >= projector.nodata_height_below]
    low, high = ms._heightmap_domain(valid)
    height_range = np.float32(high-low)
    result = []
    for batch in batches[0]:
        mesh = batch['levels'][0]['mesh']
        positions = np.asarray(mesh['positions'], dtype=np.float32)
        indices = np.asarray(mesh['indices'], dtype=np.uint32).reshape(-1, 3)
        for row in batch['transforms']:
            matrix = np.asarray(row, dtype=np.float32).reshape(4, 4)
            tx, ty, tz = matrix[:3, 3]
            translation = np.array([tx*scale-span*0.5, tz*scale-span*0.5,
                                    ty-height_range*z_scale*0.5], dtype=np.float32)
            world = positions @ (matrix[:3, :3] * scale).T + translation
            result.extend(world[indices].astype(np.float64).tolist())
    return result


def point_batches(recipe):
    """Project authored 3-D points once through the shared camera; keep depth."""
    from . import map_scene as ms
    from ._map_scene_projection import TerrainProjector
    from ._map_scene_validation import diagnostic_block
    layers=[layer for layer in recipe.layers if isinstance(layer,ms.PointCloudLayer)]
    if not layers:
        return []
    projector=TerrainProjector(recipe)
    width,height=recipe.output.width,recipe.output.height
    batches=[]
    for layer in layers:
        metadata=ms._metadata_dict(layer.metadata)
        if metadata.get('coordinate_space')!='world' or metadata.get('positions') is None:
            raise ms.MapSceneNativeUnavailable([diagnostic_block(layer=str(layer.layer_id),
                reason='point source has no world-space XYZ command stream in the terrain CRS; independent screen projection has no shared-depth meaning',
                required_native='CONCORDIA world-space point positions')])
        if layer.crs is not None:
            from ._map_scene_common import _same_crs
            if not _same_crs(layer.crs,recipe.target_crs or recipe.terrain.crs):
                raise ms.MapSceneNativeUnavailable([diagnostic_block(layer=str(layer.layer_id),
                    reason='point positions must share the terrain CRS before native projection',
                    required_native='CONCORDIA world-space point transform')])
        positions=np.asarray(metadata['positions'],dtype=np.float64)
        if positions.ndim!=2 or positions.shape[1]!=3 or not np.isfinite(positions).all():
            raise ValueError('native point positions must be finite Nx3 world XYZ')
        base=ms._rgba01(ms._render_color(metadata.get('color'),(255,255,255,220)))
        colors=metadata.get('colors')
        if colors is not None:
            colors=np.asarray(colors)
            if colors.shape!=(len(positions),3):
                raise ValueError('point colors must match XYZ count with three byte channels')
            if not np.isfinite(colors).all() or np.any(colors<0) or np.any(colors>255):
                raise ValueError('point colors must contain finite byte channels')
        size=float(metadata.get('point_size',4.0))
        if not np.isfinite(size) or size<=0:
            raise ValueError('point_size must be positive and finite')
        values=[]
        for index,point in enumerate(positions):
            try:
                pixel=projector.project(point,allow_outside=True)
            except ValueError as error:
                if 'behind' in str(error) or 'clip planes' in str(error):
                    continue
                raise
            color=base if colors is None else (*tuple(float(v)/255.0 for v in colors[index]),base[3])
            values.append([2.0*pixel[0]/width-1.0,1.0-2.0*pixel[1]/height,size,*color,pixel[2]])
        edl,strength,radius=ms._edl_settings(metadata)
        batches.append({'points':np.ascontiguousarray(values,dtype=np.float32).reshape(-1,8),
                        'edl':edl,'edl_strength':strength,'edl_radius_px':radius})
    if not batches:
        return []
    edl={(batch['edl'],batch['edl_strength'],batch['edl_radius_px']) for batch in batches}
    if len(edl)!=1:
        raise ms.MapSceneNativeUnavailable([diagnostic_block(layer='point_clouds',
            reason='overlapping transparent sources need one shared EDL configuration for their resolved representative depth',
            required_native='CONCORDIA shared point EDL settings')])
    # One accumulator covers every participating cloud, preserving OIT across
    # source boundaries as well as within each point source.
    first=dict(batches[0])
    first['points']=np.ascontiguousarray(np.concatenate([batch['points'] for batch in batches]))
    return [first]


def is_cartographic_landmark(layer):
    """The legacy screen image, including its default rect, has no world mesh."""
    from . import map_scene as ms
    metadata=ms._metadata_dict(layer.metadata)
    rect=metadata.get("screen_rect") or metadata.get("landmark_screen_rect")
    return (isinstance(layer,ms.BuildingLayer) and bool(ms._building_gltf_path(layer))
            and not layer.features and (rect is None or (isinstance(rect,(list,tuple)) and len(rect)==4)))


def landmark_batches(recipe):
    """Author the existing screen-image style before rendering; GPU blends it.

    These bytes come only from the asset and recipe, never a rendered frame.
    The existing vertical color ramp is an image style, not PBR lighting.
    """
    from . import map_scene as ms
    from pathlib import Path
    from . import io
    from ._png import load_png_rgba
    batches=[];materials_count=0;primitives_count=0;assets=[];stages={}
    width,height=recipe.output.width,recipe.output.height
    for layer in recipe.layers:
        if not isinstance(layer,ms.BuildingLayer) or not is_cartographic_landmark(layer):
            continue
        intents=ms._building_textured_material_intents(layer)
        if not intents:
            raise ValueError(f"landmark {layer.layer_id} has no authored texture")
        gltf_path=ms._building_gltf_path(layer)
        mesh,materials,primitives=io.import_gltf(gltf_path,with_materials=True)
        texture_path=intents[0].get("albedo_texture") or intents[0].get("texture_path")
        texture=np.asarray(load_png_rgba(texture_path),dtype=np.uint8)
        if texture.ndim!=3 or texture.shape[2]!=4 or not texture.shape[0] or not texture.shape[1]:
            raise ValueError("cartographic landmark texture must be nonempty RGBA")
        metadata=ms._metadata_dict(layer.metadata)
        rect=metadata.get("screen_rect") or metadata.get("landmark_screen_rect")
        if rect is not None and not np.isfinite(np.asarray(rect,dtype=float)).all():
            raise ValueError("cartographic landmark rectangle must be finite")
        left,top,right,bottom=ms._screen_rect_pixels(rect,width=width,height=height)
        target_h,target_w=bottom-top,right-left
        yy=np.linspace(0,texture.shape[0]-1,target_h).astype(np.int32)
        xx=np.linspace(0,texture.shape[1]-1,target_w).astype(np.int32)
        sampled=texture[np.ix_(yy,xx)].astype(np.float32)
        sampled[...,:3]*=np.linspace(1.08,0.78,target_h,dtype=np.float32)[:,None,None]
        opacity=float(intents[0].get("opacity",1.0))
        if not np.isfinite(opacity) or not 0.0<=opacity<=1.0:
            raise ValueError("cartographic landmark opacity must be finite in [0,1]")
        sampled[...,3]*=opacity
        rgba=np.zeros((height,width,4),dtype=np.uint8)
        rgba[top:bottom,left:right]=np.clip(sampled,0,255).astype(np.uint8)
        batches.append({"rgba":rgba,"depth_test":False,"layer":str(layer.layer_id)})
        materials_count+=len(materials)
        primitives_count+=len(primitives) if primitives else int(np.asarray(mesh.indices).size//3)
        assets.append(str(metadata.get("source_id") or Path(gltf_path).stem))
        stages[str(layer.layer_id)]="post_resolve_cartographic"
    if not batches:return [],{}
    return batches,{"gltf_textured_backend":"native_cartographic_landmark",
        "gltf_textured_layer_count":len(batches),"gltf_material_count":materials_count,
        "gltf_primitive_count":primitives_count,"gltf_asset_ids":assets,
        "gltf_layer_stages":stages,"gltf_material_model":"authored_screen_image"}
