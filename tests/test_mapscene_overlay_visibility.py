"""Stage-3 regressions: physical occluders, world label geometry and widths."""
import numpy as np
import pytest
from PIL import Image

import forge3d as f3d
from forge3d import map_scene
from forge3d._map_scene_projection import TerrainProjector, project_vector_recipe
from test_mapscene_geometric_integrity import _scene, N, requires_gpu


def _vector(coords, *, join="round", width_world=None):
    return f3d.VectorOverlay(layer_id="vector", crs="EPSG:32633", line_join=join,
        width_px=3 if width_world is None else None, width_world=width_world,
        features=[{"type":"Feature", "geometry":{"type":"LineString", "coordinates":coords}}],
        style={"layers":[{"type":"line","paint":{"line-color":"#ff0000"}}]})


@pytest.mark.parametrize("space", (None, "screen"))
def test_legacy_3d_point_labels_keep_screen_coordinates(monkeypatch, space):
    scene=_scene(np.zeros((N,N),np.float32),camera_mode="clipmap:2:32:32:10:0.3:zup")
    metadata={} if space is None else {"coordinate_space":space}
    scene.recipe.layers=[f3d.LabelLayer(layer_id="legacy",occlusion="none",
        labels=[{"id":"legacy","text":"P","geometry":{"type":"Point","coordinates":[48,32,0]}}],
        glyph_atlas={"glyphs":["P"]},metadata=metadata)]
    def no_projection(*args,**kwargs):
        pytest.fail("legacy screen coordinates must not enter world projection")
    monkeypatch.setattr(TerrainProjector,"__init__",no_projection)
    plan=scene.compile_plan()
    assert not plan.validation_report.render_blocked(),plan.validation_report.to_dict()
    assert tuple(plan.label_plans["legacy"].accepted[0].candidate.anchor)==(48.0,32.0,0.0)


@pytest.mark.parametrize("mode", ("mesh:zup", "clipmap:2:32:32:10:0.3:zup:globe"))
def test_projected_anchors_bypass_new_world_projection(monkeypatch, mode):
    dem=np.zeros((N,N),np.float32); dem[0,0]=np.nan
    scene=_scene(dem,camera_mode=mode)
    scene.recipe.layers=[f3d.LabelLayer(layer_id="labels",occlusion="none",
        labels=[{"id":"ready","text":"P","geometry":{"type":"Point","coordinates":[0,0]},
                 "projected_anchor":[128,128,0.5]}],glyph_atlas={"glyphs":["P"]},metadata={"source_id":"ready","coordinate_space":"world"})]
    def no_projection(*args,**kwargs):
        raise AssertionError("serialized projected anchors must keep their authority")
    monkeypatch.setattr(TerrainProjector,"__init__",no_projection)
    assert scene.compile_plan().label_plans["labels"].accepted


def test_nodata_projection_uses_renderer_filled_heights():
    dem=np.zeros((N,N),np.float32);dem[10:20,10:20]=np.nan
    scene=_scene(dem,layers=[_vector([[-20,0],[20,0]])])
    projector=TerrainProjector(scene.recipe)
    filled,threshold=map_scene._mapscene_nodata_heightmap(dem)
    np.testing.assert_array_equal(projector.heightmap,filled)
    assert projector.nodata_height_below==threshold
    assert not scene.validate().render_blocked()


@pytest.mark.parametrize("coords", ([[0,0,120],[20,0]], [[0,0],[20,0,120]]))
def test_mixed_elevation_path_preserves_explicit_endpoints(coords):
    projector=TerrainProjector(_scene(np.full((N,N),30,np.float32)).recipe)
    world=projector.world_path(coords)
    expected_start=coords[0][2] if len(coords[0])>2 else projector.height(projector.uv(coords[0]))
    assert world[0][2]==expected_start
    projected=projector.path(coords)
    assert projected[0]==projector.project(coords[0])
    assert projected[-1]==projector.project(coords[-1])
    assert all(len(point)==3 for point in world[:-1])


def test_world_label_layers_share_one_depth_compilation(monkeypatch):
    scene=_scene(np.zeros((N,N),np.float32))
    scene.recipe.layers=[f3d.LabelLayer(layer_id=name,occlusion="terrain",
        labels=[{"id":name,"text":"P","geometry":{"type":"Point","coordinates":[0,0]}}],
        glyph_atlas={"glyphs":["P"]},metadata={"source_id":name,"coordinate_space":"world"}) for name in ("first","second")]
    original=TerrainProjector.depth_image
    calls=[]
    def count(projector):
        calls.append(projector)
        return original(projector)
    monkeypatch.setattr(TerrainProjector,"depth_image",count)
    compiled=scene.compile_plan()
    assert not compiled.validation_report.render_blocked()
    assert all(plan.accepted for plan in compiled.label_plans.values())
    assert len(calls)==1


@pytest.mark.parametrize("coords", ([[10000,0],[10001,0]], [[0,0],[float("inf"),0]]))
def test_bad_vectors_report_before_drawing(coords,tmp_path,monkeypatch):
    scene=_scene(np.zeros((N,N),np.float32),layers=[_vector(coords)])
    report=scene.validate()
    assert report.render_blocked()
    assert any(d.code=="world_overlay_projection_unavailable" for d in report.diagnostics)
    assert scene.compile_plan().validation_report.render_blocked()
    monkeypatch.setattr(map_scene,"_render_terrain_renderer_result_impl",lambda *a,**k:pytest.fail("draw before validation"))
    with pytest.raises(RuntimeError,match="blocking diagnostics"):
        scene.render(str(tmp_path/"blocked.png"))
    assert not (tmp_path/"blocked.png").exists()


@pytest.mark.parametrize("mode,code", (("screen","label_plan_compile_unavailable"),
                                       ("mesh:zup","world_overlay_projection_unavailable")))
def test_invalid_label_depth_is_a_typed_validation_report(mode,code,tmp_path,monkeypatch):
    scene=_scene(np.zeros((N,N),np.float32),camera_mode=mode)
    scene.recipe.layers=[f3d.LabelLayer(layer_id="labels",occlusion="terrain",
        labels=[{"id":"depth","text":"P","geometry":{"type":"Point","coordinates":[0,0]}}],
        glyph_atlas={"glyphs":["P"]},metadata={"source_id":"depth","coordinate_space":"world",
            "depth_occlusion":{"image":np.full((4,4),0.5).tolist(),"source":"serialized_depth_proxy"}})]
    report=scene.validate()
    assert report.render_blocked()
    assert any(d.code==code and "depth_convention" in d.message for d in report.diagnostics)
    assert scene.compile_plan().validation_report.render_blocked()
    monkeypatch.setattr(map_scene,"_render_terrain_renderer_result_impl",lambda *a,**k:pytest.fail("draw before validation"))
    with pytest.raises(RuntimeError,match="blocking diagnostics"):
        scene.render(str(tmp_path/"blocked.png"))
    assert not (tmp_path/"blocked.png").exists()


@requires_gpu
@pytest.mark.parametrize("join", ("miter","round"))
@pytest.mark.parametrize("mode", ("mesh:zup","clipmap:2:64:64:10:0.3:zup"))
def test_ridge_hides_vectors_on_both_compositors(tmp_path,join,mode,record_property):
    rows,cols=np.mgrid[:128,:128]
    dem=(80*np.exp(-((rows-64)/5)**2)).astype(np.float32)
    scene=_scene(dem,camera_mode=mode,camera=f3d.OrbitCamera(
        target=(0,0,0),distance=260,azimuth_deg=-90,elevation_deg=70,fov_deg=40))
    scene.recipe.terrain.metadata=dict(scene.recipe.terrain.metadata,width=128,height=128)
    span=map_scene._terrain_scene_diagonal(scene.recipe.terrain)
    projector=TerrainProjector(scene.recipe);depth=projector.depth_image()
    hidden=[[-0.3*span,0.06*span],[0.3*span,0.06*span]]
    samples=projector.path(hidden)
    assert all(p[2]>depth[int(p[1]),int(p[0])] for p in samples)
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path/"bare.png"));bare=np.asarray(Image.open(tmp_path/"bare.png"))
        scene.recipe.layers=[_vector(hidden,join=join)]
        scene.render(str(tmp_path/"hidden.png"))
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/"hidden.png")),bare)
        # A second stroke above the occluder must remain visible.
        scene.recipe.layers=[_vector([[x,y,120] for x,y in hidden],join=join)]
        scene.render(str(tmp_path/"above.png"))
        assert np.any(np.asarray(Image.open(tmp_path/"above.png"))!=bare)
    record_property("occluded_line_samples",len(samples))
    record_property("polar_camera_degrees",70)


@requires_gpu
@pytest.mark.parametrize("direction,text", (
    ("horizontal","P"),("sloping","P"),("area","P"),
    ("horizontal","Ridge Road"),("sloping","Ridge Road"),("area","Ridge Road")),
    ids=("horizontal","sloping","area","horizontal-words","sloping-words","area-words"))
def test_automatic_world_line_and_area_label_pixels(tmp_path,direction,text,record_property,monkeypatch):
    from forge3d import recipe_manifest as rm
    from test_mapscene_sutura_integrity import _report_bytes, _ssim
    kind="Polygon" if direction=="area" else "LineString"
    coords=([[-60,-30],[60,30]] if direction=="sloping" else
            [[-60,0],[60,0]] if direction=="horizontal" else
            [[[-30,-30],[30,-30],[30,30],[-30,30],[-30,-30]]])
    scene=_scene(np.zeros((N,N),np.float32))
    # Bundles persist terrain asset references, rather than runtime arrays.
    terrain_path=tmp_path/"terrain.npy"
    np.save(terrain_path,scene.recipe.terrain.data)
    scene.recipe.terrain.path=str(terrain_path)
    scene.recipe.terrain.data=None
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path/"bare.png"));bare=np.asarray(Image.open(tmp_path/"bare.png"))
        scene.recipe.layers=[f3d.LabelLayer(layer_id="labels",occlusion="terrain",
            labels=[{"id":"world","text":text,"geometry":{"type":kind,"coordinates":coords}}],
            glyph_atlas={"glyphs":sorted(set(text))},metadata={"source_id":"world","coordinate_space":"world"})]
        compiled=scene.compile_plan();plan=compiled.label_plans["labels"]
        assert not compiled.validation_report.render_blocked(),compiled.validation_report.to_dict()
        assert len(plan.accepted)==1,plan.to_dict()
        report=scene.render(str(tmp_path/"label.png"))
        yy,xx=np.nonzero(np.any(np.asarray(Image.open(tmp_path/"label.png"))!=bare,axis=2))
        assert len(xx)
        label=plan.accepted[0];bounds=label.screen_bounds
        if direction=="sloping":
            assert any(glyph["rotation"]!=0 for glyph in label.positioned_glyphs)
        assert xx.min()>=np.floor(bounds[0])-1 and xx.max()<=np.ceil(bounds[2])+1
        assert yy.min()>=np.floor(bounds[1])-1 and yy.max()<=np.ceil(bounds[3])+1
        record_property("chosen_anchor",list(label.candidate.anchor))
        manifest=rm.manifest_to_json(compiled.manifest)
        scene.save_bundle(tmp_path/"bundle")
        with monkeypatch.context() as patch:
            def frozen(*args,**kwargs):
                pytest.fail("saved label geometry must be consumed without projection or relayout")
            patch.setattr(map_scene,"_label_plan_from_layer",frozen)
            loaded=f3d.MapScene.load_bundle(scene.last_bundle_path)
            rerender=loaded.render(str(tmp_path/"loaded.png"))
        assert rm.manifest_to_json(loaded.compiled_plan.manifest)==manifest
        assert _report_bytes(report)==_report_bytes(rerender)
        score=_ssim(np.asarray(Image.open(tmp_path/"label.png")),np.asarray(Image.open(tmp_path/"loaded.png")))
        assert score==1.0
        record_property("roundtrip_ssim",score)


@requires_gpu
@pytest.mark.parametrize("join", ("miter","round"))
def test_world_width_is_resolved_per_perspective_segment(tmp_path,join,record_property):
    scene=_scene(np.zeros((N,N),np.float32),camera=f3d.OrbitCamera(
        target=(0,0,0),distance=420,azimuth_deg=-90,elevation_deg=70,fov_deg=40),
        layers=[_vector([[-40,-60],[40,-60],[40,60],[-40,60]],join=join,width_world=8)])
    assert not scene.validate().render_blocked()
    projected=project_vector_recipe(scene.recipe).layers[0]
    widths=[feature['_projected_width_px'] for feature in projected.features]
    assert min(widths)>0 and max(widths)>min(widths)
    with map_scene._shared_terrain_render_context():
        layer=scene.recipe.layers[0];scene.recipe.layers=[]
        scene.render(str(tmp_path/"bare.png"))
        scene.recipe.layers=[layer];scene.render(str(tmp_path/"wide.png"))
    assert np.any(np.asarray(Image.open(tmp_path/"wide.png"))!=np.asarray(Image.open(tmp_path/"bare.png")))
    record_property("projected_width_range_px",[min(widths),max(widths)])


@requires_gpu
@pytest.mark.parametrize("geometry", (
    {"type":"MultiPoint","coordinates":[[-20,0],[20,0]]},
    {"type":"MultiLineString","coordinates":[[[-30,-20],[30,-20]],[[-30,20],[30,20]]]},
    {"type":"Polygon","coordinates":[[[-30,-30],[30,-30],[30,30],[-30,30],[-30,-30]]]},
    {"type":"MultiPolygon","coordinates":[[[[-30,-20],[-5,-20],[-5,20],[-30,20],[-30,-20]]],
                                             [[[5,-20],[30,-20],[30,20],[5,20],[5,-20]]]]},
), ids=lambda geometry:geometry["type"])
def test_world_width_covers_multi_geometry_and_polygon_outlines(tmp_path,geometry):
    layer=_vector([],width_world=8)
    layer.features=[{"type":"Feature","geometry":geometry}]
    scene=_scene(np.zeros((N,N),np.float32),layers=[layer])
    assert not scene.validate().render_blocked()
    projected=project_vector_recipe(scene.recipe).layers[0]
    strokes=[feature for feature in projected.features if not feature.get("_terrain_fill_only")]
    assert strokes and all(feature["_projected_width_px"]>0 for feature in strokes)
    assert all(feature["geometry"]["type"] in ("Point","LineString") for feature in strokes)
    with map_scene._shared_terrain_render_context():
        scene.recipe.layers=[];scene.render(str(tmp_path/"bare.png"))
        scene.recipe.layers=[layer];scene.render(str(tmp_path/"wide.png"))
    assert np.any(np.asarray(Image.open(tmp_path/"wide.png"))!=np.asarray(Image.open(tmp_path/"bare.png")))


@requires_gpu
def test_gappy_globe_scene_renders_world_vectors_and_precomputed_labels(tmp_path,monkeypatch):
    dem=np.zeros((N,N),np.float32);dem[10:20,10:20]=np.nan
    scene=_scene(dem,camera_mode="clipmap:2:32:32:10:0.3:zup:globe")
    with map_scene._shared_terrain_render_context():
        scene.render(str(tmp_path/"bare.png"))
        scene.recipe.layers=[_vector([[-20,0],[20,0]])]
        assert not scene.compile_plan().validation_report.render_blocked()
        scene.render(str(tmp_path/"vector.png"))
        assert np.any(np.asarray(Image.open(tmp_path/"vector.png"))!=np.asarray(Image.open(tmp_path/"bare.png")))
        scene.recipe.layers=[f3d.LabelLayer(layer_id="labels",occlusion="none",
            labels=[{"id":"ready","text":"P","geometry":{"type":"Point","coordinates":[0,0]},
                     "projected_anchor":[128,128,0.5]}],glyph_atlas={"glyphs":["P"]},metadata={"source_id":"ready","coordinate_space":"world"})]
        with monkeypatch.context() as patch:
            def frozen(*args,**kwargs):
                pytest.fail("explicit projected anchors must not invoke the terrain projector")
            patch.setattr(TerrainProjector,"__init__",frozen)
            assert scene.compile_plan().label_plans["labels"].accepted
            scene.render(str(tmp_path/"label.png"))
        assert np.any(np.asarray(Image.open(tmp_path/"label.png"))!=np.asarray(Image.open(tmp_path/"bare.png")))


@requires_gpu
def test_rotated_text_quad_keeps_its_geometry_at_viewport_edge():
    # An opaque atlas makes the entire GPU quad observable. Before rotation its
    # rectangle crosses y=0; after rotation all corners are inside the viewport.
    scene=f3d.Scene(64,64)
    scene.disable_terrain()
    bare=scene.render_rgba(certificate=False)
    scene.set_native_text_atlas(np.full((8,8,3),255,np.uint8),channels=3)
    scene.enable_native_text()
    scene.add_native_text_rect_uv_halo(10,-2,8,16,0,0,1,1,
        1,0,0,1,0,0,0,0,0,rotation=np.pi/2)
    pixels=scene.render_rgba(certificate=False)
    changed=np.any(pixels!=bare,axis=2)
    expected=np.zeros((64,64),bool);expected[2:10,6:22]=True
    np.testing.assert_array_equal(changed,expected)
