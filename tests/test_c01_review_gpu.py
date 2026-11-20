"""C01 combined-meaning and full-frame palette acceptance (physical GPU required)."""
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

import forge3d as f3d


def _capture_visibility(monkeypatch):
    from forge3d import map_scene as ms
    from forge3d._map_scene_projection import frame_visibility, needs_frame_visibility
    original = ms._render_terrain_renderer_result
    captures = []
    def render(recipe, *args, **kwargs):
        result = original(recipe, *args, **kwargs)
        assert result is not None
        if needs_frame_visibility(recipe):
            captures.append(frame_visibility(recipe, result.aov_frame))
        return result
    monkeypatch.setattr(ms, '_render_terrain_renderer_result', render)
    return captures


def _scene():
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (1, 1), "m", "m")
    scene = f3d.SceneRecipe(f3d.TerrainSource(data=np.zeros((64, 64), np.float32), crs=grid.crs),
        f3d.OrbitCamera(distance=20, elevation_deg=89, azimuth_deg=180),
        f3d.LightingPreset("custom", sun_direction=(.8, .6, 0), settings={"cli_params": {"camera_mode": "mesh"}}),
        output=f3d.OutputSpec(width=128, height=128))
    return grid, scene


def test_every_nadir_class_pixel_including_corner_is_exact(tmp_path, monkeypatch):
    captures = _capture_visibility(monkeypatch)
    grid, scene = _scene()
    swatch = (40, 60, 20, 255)
    palette = f3d.CategoricalRasterStyle({3: swatch}, {3: "class"})
    material = f3d.TerrainMaterialRecipe.land_cover(palette, albedo=(170, 125, 80))
    scene = f3d.MapScene(recipe=replace(material.bind(scene, grid, classes=np.full((64, 64), 3)), map_furniture=None))
    path = tmp_path / "nadir.png"
    scene.render(path)
    assert scene.last_render_backend == "gpu_terrain"
    pixels = np.asarray(Image.open(path))
    terrain = captures[-1][2]
    assert terrain.any() and terrain[:8, :8].all()
    np.testing.assert_array_equal(pixels[terrain], np.tile(swatch, (terrain.sum(), 1)))
    np.testing.assert_array_equal(pixels[:8, :8], np.full((8, 8, 4), swatch, np.uint8))
    assert not np.any(np.all(pixels[~terrain] == swatch, axis=-1))


def test_protected_classes_and_physical_surfaces_share_one_map(tmp_path, monkeypatch):
    captures = _capture_visibility(monkeypatch)
    grid, base = _scene()
    swatch = (40, 60, 20, 255)
    palette = f3d.CategoricalRasterStyle({3: swatch}, {3: "class"}, nodata=0)
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (150, 140, 130), roughness=1),),
                                       categorical=palette, relief_strength=.6)
    classes = np.zeros((64, 64), np.int16)
    classes[:, :32] = 3
    bound = replace(material.bind(base, grid, classes=classes), map_furniture=None)
    frames = []
    for i, sun in enumerate(((.8, .6, 0), (-.8, .1, 0))):
        scene = f3d.MapScene(recipe=replace(bound, lighting=replace(bound.lighting, sun_direction=sun)))
        path = tmp_path / f"sun-{i}.png"
        scene.render(path)
        assert scene.last_render_backend == "gpu_terrain"
        frames.append(np.asarray(Image.open(path)))
    first_classes = np.all(frames[0] == swatch, axis=-1)
    assert first_classes.any() and (~first_classes).any()
    np.testing.assert_array_equal(first_classes, np.all(frames[1] == swatch, axis=-1))
    np.testing.assert_array_equal(frames[0][first_classes], frames[1][first_classes])
    visible_lit = captures[-1][2] & ~first_classes
    assert visible_lit.any()
    assert np.any(frames[0][visible_lit] != frames[1][visible_lit])
    assert scene.last_render_metadata['material_shading'] == 'physical lighting plus declared cartographic relief'


def test_buildings_occlude_class_and_vector_paint_from_same_frame(tmp_path, monkeypatch):
    captures = _capture_visibility(monkeypatch)
    grid, base = _scene()
    base = replace(base, camera=replace(base.camera, distance=110, elevation_deg=40, azimuth_deg=140),
                   output=replace(base.output, width=256, height=256))
    swatch = (40, 60, 20, 255)
    vector_swatch = (230,120,30,255)
    palette = f3d.CategoricalRasterStyle({3: swatch,4: vector_swatch}, {3: 'class',4:'vector'})
    material = f3d.GeospatialMaterialRecipe((f3d.MaterialLayer('ground', (150, 140, 130), roughness=.8),),
                                          categorical=palette)
    ring = [[500020,4599980],[500044,4599980],[500044,4599956],[500020,4599956],[500020,4599980]]
    feature = {'type':'Feature', 'id':'building', 'properties':{'height':25},
               'geometry':{'type':'Polygon', 'coordinates':[ring]}}
    buildings = f3d.MapSceneBuildingLayer('buildings', features=[feature], metadata={'crs':grid.crs})
    bound = replace(material.bind(base, grid, classes=np.full((64,64),3)), layers=(buildings,), map_furniture=None)
    first = f3d.MapScene(recipe=bound)
    first.render(tmp_path / 'building.png')
    pixels = np.asarray(Image.open(tmp_path / 'building.png'))
    scene_depth, uv, terrain = captures[-1]
    # Finite non-terrain scene pixels are actual nearer scatter fragments.
    building = (scene_depth < 1) & ~terrain
    assert building.any() and terrain.any() and np.any(scene_depth == 1)
    assert first.last_render_metadata['building_backend'] == 'terrain_scatter_instanced_mesh'
    assert not np.any(np.all(pixels[building] == swatch, axis=-1))
    np.testing.assert_array_equal(pixels[terrain], np.tile(swatch, (terrain.sum(),1)))
    outline = [[500001,4599999],[500062,4599999],[500062,4599938],[500001,4599938],[500001,4599999]]
    vector = f3d.VectorOverlay('paint', features=[{'type':'Feature','id':'paint','properties':{},
        'geometry':{'type':'Polygon','coordinates':[outline]}}], crs=grid.crs,
        metadata={'coordinate_space':'world'})
    vector = material.with_layer('ground',roughness=None).bind_vector(vector, [[500032,4599968]], units='m', classes=[4])
    second = f3d.MapScene(recipe=replace(bound, layers=(buildings,vector)))
    report = second.validate()
    assert report.status != 'error', [d.message for d in report.diagnostics]
    second.render(tmp_path / 'building-vector.png')
    painted = np.asarray(Image.open(tmp_path / 'building-vector.png'))
    assert np.any(np.all(painted[terrain] == vector_swatch,axis=-1))
    np.testing.assert_array_equal(painted[building], pixels[building])
    np.testing.assert_array_equal(painted[scene_depth == 1], pixels[scene_depth == 1])
    # Switching the shared renderer between product routes must restore the
    # ordinary forward pipeline state, including its established depth policy.
    from forge3d.map_scene import _shared_terrain_render_context
    physical = replace(bound,terrain=replace(bound.terrain,
        style=replace(bound.terrain.style,recipe=replace(material,presentation='physical'))))
    with _shared_terrain_render_context():
        for name,recipe in (('before',physical),('protected',bound),('after',physical)):
            f3d.MapScene(recipe=recipe).render(tmp_path / f'switch-{name}.png')
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'switch-before.png')),
                                  np.asarray(Image.open(tmp_path/'switch-after.png')))


@pytest.mark.parametrize("elevation,cell_width", ((40,4),(12,4),(12,1)))
def test_visible_ridge_classes_use_native_pixel_centre_cells(tmp_path, monkeypatch, elevation, cell_width):
    """Native physical sampling is independent of protected unprojection.

    Independent review measured 99.61--99.94% hue agreement for these views.
    Predeclared acceptance: >=99.6%, with discrepancies only at native class
    boundaries (an eight-connected immediate neighbour has the opposite hue).
    """
    grid, base = _scene()
    rows, cols = np.mgrid[:64,:64]
    heights = (25*np.exp(-((rows-32)/7)**2)+6*np.sin(cols/5)).astype(np.float32)
    base = replace(base, terrain=replace(base.terrain,data=heights),
        camera=replace(base.camera,distance=110,elevation_deg=elevation,azimuth_deg=200),
        lighting=replace(base.lighting,sun_direction=(.3,.3,.9)),
        output=replace(base.output,width=384,height=256))
    red, blue = (220,30,30,255), (30,30,220,255)
    classes = np.where(((cols//cell_width)+(rows//cell_width))%2==0,1,2)
    palette = f3d.CategoricalRasterStyle({1:red,2:blue},{1:'red',2:'blue'})
    material = f3d.TerrainMaterialRecipe.land_cover(palette,albedo=(150,140,130))
    frames = {}
    native_coverage = []
    from forge3d import map_scene as ms
    original_render = ms._render_terrain_renderer_result
    def observe(recipe, *args, **kwargs):
        result = original_render(recipe, *args, **kwargs)
        if result.aov_frame is not None:
            terrain_depth, scene_depth = result.aov_frame._visibility_depth()
            native_coverage.append((terrain_depth < 1) & (terrain_depth == scene_depth))
        return result
    monkeypatch.setattr(ms, "_render_terrain_renderer_result", observe)
    from forge3d.map_scene import _shared_terrain_render_context
    with _shared_terrain_render_context():
        for presentation in ('protected','physical'):
            bound = replace(material,presentation=presentation).bind(base,grid,classes=classes)
            scene = f3d.MapScene(recipe=replace(bound,map_furniture=None))
            path = tmp_path/f'ridge-{presentation}.png'
            scene.render(path)
            frames[presentation] = np.asarray(Image.open(path)).astype(np.int16)
    protected, physical = frames['protected'], frames['physical']
    is_red = np.all(protected==red,axis=-1)
    painted = is_red | np.all(protected==blue,axis=-1)
    native_red = physical[...,0]>physical[...,2]
    maximum = np.maximum(physical[...,0],physical[...,2])
    confident = (np.abs(physical[...,0]-physical[...,2])>.5*maximum) & (maximum>20)
    compared = painted & confident
    native_visible = native_coverage[0]
    coverage = float(np.count_nonzero(painted & native_visible)/np.count_nonzero(native_visible))
    print("NATIVE_PAINT_COVERAGE", elevation, cell_width, int(native_visible.sum()), coverage)
    # Every native-visible cell has a valid categorical value. Baseline measured
    # full coverage in all three cases; losing even one such pixel is a defect.
    assert coverage == 1.0, f"Protected paint coverage: {coverage:.6%}"
    assert compared.any()
    disagreement = compared & (is_red!=native_red)
    padded = np.pad(native_red,1,mode='edge')
    boundary = np.zeros(native_red.shape,bool)
    for dy in range(3):
        for dx in range(3):
            neighbour = padded[dy:dy+native_red.shape[0],dx:dx+native_red.shape[1]]
            boundary |= neighbour!=native_red
    fraction = float(disagreement.sum()/compared.sum())
    assert fraction<=.004, f'Native hue disagreement: {fraction:.6%}'
    assert not np.any(disagreement & ~boundary), 'Only immediate native class boundaries may disagree'
    import json
    import os
    from pathlib import Path
    if os.environ.get('FORGE3D_C01_ARTIFACT_DIR'):
        out = Path(os.environ['FORGE3D_C01_ARTIFACT_DIR'])/f'ridge-e{elevation}-c{cell_width}'
        out.mkdir(parents=True,exist_ok=True)
        marker = np.zeros((*disagreement.shape,3),np.uint8)
        marker[disagreement] = (255,255,0)
        Image.fromarray(np.concatenate((protected[...,:3],physical[...,:3],marker),axis=1).astype(np.uint8)).save(out/'comparison.png')
        (out/'measurement.json').write_text(json.dumps({'compared':int(compared.sum()),
            'disagreements':int(disagreement.sum()),'agreement':1-fraction,
            'nonboundary_disagreements':int(np.count_nonzero(disagreement & ~boundary)),
            'predeclared_minimum_agreement':.996,'native_visible':int(native_visible.sum()),'painted_native_visible':int(np.count_nonzero(painted & native_visible)),'paint_coverage':coverage,'minimum_paint_coverage':1.0},indent=2))
