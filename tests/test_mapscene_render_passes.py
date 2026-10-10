"""D04 exact pixel math, graph validation and compiled bundle replay."""
import json
import hashlib
import struct
import zlib

import numpy as np
import pytest

import forge3d as f3d
from forge3d._png import load_png_rgba
from forge3d.recipe_manifest import manifest_from_json, manifest_to_json
from forge3d.render_pass import compile_passes, execute_passes

# Float fixtures use the platform's guaranteed decimal precision. PNG fixtures
# below compare integer channels exactly, without a tolerance.
FLOAT_ATOL = 10.0 ** -np.finfo(np.float64).precision


def _scene(width=1, height=1, **kwargs):
    return f3d.MapScene(
        terrain=f3d.TerrainSource(crs="EPSG:3857", metadata={"source_id": "fixture", "asset_status": "fixture"}),
        lighting=f3d.LightingPreset(), output=f3d.OutputSpec(width, height), **kwargs)


def _blend(bottom, top, operation, **options):
    shape = bottom.shape[:2]
    passes = [f3d.RenderPassSpec("blend", "overlay", ("bottom", "top"), operation,
                               color_space="srgb", **options)]
    inputs = {"bottom": f3d.RenderPassInput(bottom), "top": f3d.RenderPassInput(top)}
    return execute_passes(json.dumps(compile_passes(passes, inputs, shape)), shape)


@pytest.mark.parametrize("operation,expected", [
    ("multiply", [1/8, 1/8, 3/32, 1]),
    ("screen", [5/8, 5/8, 25/32, 1]),
    ("alpha_over", [1/2, 1/4, 1/8, 1]),
])
def test_opaque_hand_computed_fixture(operation, expected):
    bottom = np.array([[[1/4, 1/2, 3/4, 1], [0, 1, 0, 1]]])
    top = np.array([[[1/2, 1/4, 1/8, 1], [1, 0, 1, 1]]])
    second = {"multiply": [0, 0, 0, 1], "screen": [1, 1, 1, 1], "alpha_over": [1, 0, 1, 1]}
    np.testing.assert_allclose(_blend(bottom, top, operation), [[expected, second[operation]]], atol=FLOAT_ATOL, rtol=0)


@pytest.mark.parametrize("operation,expected", [
    ("multiply", [7/24, 7/24, 31/96, 3/4]),
    ("screen", [11/24, 11/24, 53/96, 3/4]),
    ("alpha_over", [5/12, 1/3, 1/3, 3/4]),
])
def test_transparent_hand_computed_fixture(operation, expected):
    bottom = np.array([[[1/4, 1/2, 3/4, 1/2], [1, 1, 1, 0], [1/4, 1/2, 3/4, 1/2], [1, 0, 0, 0]]])
    top = np.array([[[1/2, 1/4, 1/8, 1/2], [1/2, 1/4, 1/8, 1/2], [1, 1, 1, 0], [0, 1, 0, 0]]])
    np.testing.assert_allclose(_blend(bottom, top, operation),
                              [[expected, [1/2, 1/4, 1/8, 1/2], [1/4, 1/2, 3/4, 1/2], [0, 0, 0, 0]]], atol=FLOAT_ATOL, rtol=0)


@pytest.mark.parametrize("operation,pixel", [("multiply", [32, 32, 24, 255]),
                                             ("screen", [159, 159, 199, 255]),
                                             ("alpha_over", [128, 64, 32, 255])])
def test_public_api_writes_hand_computed_pixels(operation, pixel, tmp_path):
    scene = _scene()
    inputs = {"a": f3d.RenderPassInput(np.array([[[.25, .5, .75, 1.]]])),
              "b": f3d.RenderPassInput(np.array([[[.5, .25, .125, 1.]]]))}
    scene.render_passes([f3d.RenderPassSpec("out", "color", ("a", "b"), operation,
                                         color_space="srgb")], inputs, str(tmp_path / "out.png"))
    assert np.asarray(load_png_rgba(tmp_path / "out.png")).tolist() == [[pixel]]


def test_order_and_opacity_are_load_bearing():
    inputs = {name: f3d.RenderPassInput(np.array([[[value, value, value, 1.0]]]))
              for name, value in (("a", .25), ("b", .5), ("c", .5))}
    results = []
    for first, second in (("multiply", "screen"), ("screen", "multiply")):
        passes = [f3d.RenderPassSpec("one", "relief", ("a", "b"), first, color_space="srgb"),
                  f3d.RenderPassSpec("two", "overlay", ("one", "c"), second, color_space="srgb")]
        results.append(execute_passes(json.dumps(compile_passes(passes, inputs, (1, 1))), (1, 1)))
    np.testing.assert_allclose(results[0], [[[9/16, 9/16, 9/16, 1]]], atol=FLOAT_ATOL, rtol=0)
    np.testing.assert_allclose(results[1], [[[5/16, 5/16, 5/16, 1]]], atol=FLOAT_ATOL, rtol=0)
    a = np.array([[[1., 0, 0, .5]]])
    b = np.array([[[0., 0, 1, .5]]])
    np.testing.assert_allclose(_blend(a, b, "alpha_over"), [[[1/3, 0, 2/3, 3/4]]], atol=FLOAT_ATOL, rtol=0)
    np.testing.assert_allclose(_blend(b, a, "alpha_over"), [[[2/3, 0, 1/3, 3/4]]], atol=FLOAT_ATOL, rtol=0)
    for operation in ("multiply", "screen", "alpha_over"):
        np.testing.assert_allclose(_blend(a, b, operation, parameters={"opacity": 0}), a, atol=FLOAT_ATOL, rtol=0)


def test_linear_space_and_premultiplied_inputs():
    passes = [f3d.RenderPassSpec("result", "overlay", ("black", "white"))]
    inputs = {"black": f3d.RenderPassInput(np.array([[[0., 0, 0, 1]]])),
              "white": f3d.RenderPassInput(np.array([[[.5, .5, .5, .5]]]), alpha_mode="premultiplied")}
    result = execute_passes(json.dumps(compile_passes(passes, inputs, (1, 1))), (1, 1))
    # Linear .5 encoded by the IEC sRGB transfer function (not encoded .5).
    np.testing.assert_allclose(result, [[[0.7353569830524495] * 3 + [1]]], atol=FLOAT_ATOL, rtol=0)
    inputs["white"] = f3d.RenderPassInput(np.array([[[1., 1, 1, .5]]]), color_space="linear")
    np.testing.assert_allclose(execute_passes(json.dumps(compile_passes(passes, inputs, (1, 1))), (1, 1)), result)


@pytest.mark.parametrize("kwargs", [
    {"name": ""}, {"name": " "}, {"inputs": "image"}, {"inputs": ()},
    {"inputs": ("a", "b", "c")}, {"operation": "overlay"}, {"kind": "unknown"},
    {"color_space": "gamma"}, {"parameters": {"opacity": float("nan")}},
    {"parameters": {"opacity": -1}}, {"parameters": {"opacity": 2}},
    {"parameters": {"opacity": True}}, {"parameters": {"gain": 1}},
])
def test_invalid_spec(kwargs):
    values = dict(name="color", kind="color", inputs=("image",))
    values.update(kwargs)
    with pytest.raises(ValueError):
        f3d.RenderPassSpec(**values)


@pytest.mark.parametrize("data", [np.zeros((1, 1, 3)), np.zeros((0, 1, 4)),
                                 np.full((1, 1, 4), np.nan), np.full((1, 1, 4), np.inf),
                                 np.full((1, 1, 4), -1.), np.full((1, 1, 4), 2.), np.zeros((1, 1, 4), dtype=int)])
def test_invalid_input(data):
    with pytest.raises(ValueError):
        f3d.RenderPassInput(data)


def test_invalid_graph_rejected_before_output(tmp_path):
    scene = _scene()
    image = f3d.RenderPassInput(np.zeros((1, 1, 4)))
    cases = [([f3d.RenderPassSpec("x", "color", ("missing",))], {"a": image}),
             ([f3d.RenderPassSpec("x", "color", ("later",)), f3d.RenderPassSpec("later", "color", ("a",))], {"a": image}),
             ([f3d.RenderPassSpec("a", "color", ("a",))], {"a": image}),
             ([f3d.RenderPassSpec("x", "color", ("a",))] * 2, {"a": image}),
             ([f3d.RenderPassSpec("x", "color", ("a",))], {"a": f3d.RenderPassInput(np.zeros((2, 1, 4)))}),
             ([], {"a": image})]
    for passes, inputs in cases:
        with pytest.raises(ValueError):
            scene.render_passes(passes, inputs, str(tmp_path / "invalid.png"))
        assert not (tmp_path / "invalid.png").exists()
        assert not scene.recipe.pass_specs


def test_public_bundle_replay_is_bitexact_and_uses_frozen_plan(tmp_path, monkeypatch):
    source = np.array([[[.25, .5, .75, .5]]])
    inputs = {"a": f3d.RenderPassInput(source), "b": f3d.RenderPassInput(np.array([[[.5, .25, .125, .5]]]))}
    passes = [f3d.RenderPassSpec("color", "color", ("a",), color_space="srgb"),
              f3d.RenderPassSpec("relief", "relief", ("color", "b"), "multiply", {"opacity": .5}, "srgb"),
              f3d.RenderPassSpec("overlay", "overlay", ("relief", "a"), "screen", color_space="srgb")]
    scene = _scene(pass_specs=passes, pass_inputs=inputs)
    source[:] = 0  # Caller mutation cannot change the snapshot.
    with pytest.raises(ValueError):
        inputs["a"].data.setflags(write=True)
    first_report = scene.render_passes(path=str(tmp_path / "first.png"))
    scene.save_bundle(tmp_path / "first")
    loaded = f3d.MapScene.load_bundle(tmp_path / "first.forge3d")
    assert loaded.to_dict() == scene.to_dict()
    frozen = loaded.compiled_plan.render_passes_json
    monkeypatch.setattr(loaded, "compile_plan", lambda: pytest.fail("bundle replay recompiled"))
    assert loaded.render_passes(path=str(tmp_path / "replay.png")).to_dict() == first_report.to_dict()
    with pytest.raises(f3d.MapSceneNativeUnavailable, match="render_passes"):
        loaded.render(str(tmp_path / "render.png"))
    assert loaded.compiled_plan.render_passes_json == frozen
    assert scene.render_passes(path=str(tmp_path / "repeat.png")).to_dict() == first_report.to_dict()
    assert len({(tmp_path / f"{name}.png").read_bytes() for name in ("first", "repeat", "replay")}) == 1
    assert not (tmp_path / "render.png").exists()
    # Bundle review metadata records the destination. Match it before comparing
    # whole bundle bytes as well as the destination-independent compiled data.
    loaded.render_passes(path=str(tmp_path / "first.png"))
    loaded.save_bundle(tmp_path / "second")
    for rel in ("manifest.json", "scene/mapscene_recipe.json", "scene/compiled_plan.json", "scene/state.json"):
        assert (tmp_path / "first.forge3d" / rel).read_bytes() == (tmp_path / "second.forge3d" / rel).read_bytes()
    text = manifest_to_json(scene.compiled_plan.manifest)
    assert manifest_to_json(manifest_from_json(text)) == text
    assert loaded.last_render_backend == "python_ordered_rgba_composition"
    assert np.asarray(load_png_rgba(tmp_path / "first.png")).shape == (1, 1, 4)


def test_no_pass_serialization_and_render_delegate_unchanged(tmp_path, monkeypatch):
    scene = _scene()
    assert "render_passes" not in scene.recipe.to_dict()
    compiled = scene.compile_plan()
    assert compiled.render_passes_json is None
    assert "compiled_render_passes" not in json.loads(manifest_to_json(compiled.manifest))
    marker = object()
    monkeypatch.setattr(scene, "render", lambda path=None: marker)
    assert scene.render_passes(path=str(tmp_path / "native.png")) is marker


@pytest.mark.parametrize("options", [{"certificate": True}, {"cache": "native-cache"}])
def test_no_pass_forwards_native_controls(tmp_path, monkeypatch, options):
    scene = _scene()
    marker = object()
    received = {}

    def render(path=None, *, certificate=False, cache=None):
        received.update(path=path, certificate=certificate, cache=cache)
        return marker

    monkeypatch.setattr(scene, "render", render)
    path = str(tmp_path / "native.png")
    assert scene.render_passes(path=path, **options) is marker
    assert received == {"path": path, "certificate": False, "cache": None, **options}


def test_bundle_mismatched_pass_plan_rejected(tmp_path):
    scene = _scene(pass_specs=[f3d.RenderPassSpec("color", "color", ("image",))],
                   pass_inputs={"image": f3d.RenderPassInput(np.zeros((1, 1, 4)))})
    scene.save_bundle(tmp_path / "bundle")
    plan = tmp_path / "bundle.forge3d/scene/compiled_plan.json"
    data = json.loads(plan.read_text())
    data["compiled_render_passes"]["passes"][0]["parameters"]["opacity"] = .5
    plan.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="do not match"):
        f3d.MapScene.load_bundle(tmp_path / "bundle.forge3d")


def test_recipe_mutation_recompiles_passes(tmp_path):
    scene = _scene(pass_specs=[f3d.RenderPassSpec("color", "color", ("image",))],
                   pass_inputs={"image": f3d.RenderPassInput(np.ones((1, 1, 4)))})
    scene.render_passes(path=str(tmp_path / "first.png"))
    old = scene.compiled_plan
    scene.recipe.pass_specs = [f3d.RenderPassSpec("color", "color", ("image",), parameters={"opacity": 0})]
    scene.render_passes(path=str(tmp_path / "second.png"))
    assert scene.compiled_plan is not old
    assert not np.asarray(load_png_rgba(tmp_path / "second.png")).any()


def test_snapshot_and_optional_decode_are_canonical():
    image = f3d.RenderPassInput(np.array([[[-0., .5, 1., 0.]]]))
    payload = image.to_dict()
    assert "-0.0" not in json.dumps(payload)
    assert f3d.RenderPassInput.from_dict(payload).to_dict() == payload
    spec = f3d.RenderPassSpec("color", "color", ("image",))
    data = spec.to_dict()
    data["parameters"] = None
    assert f3d.RenderPassSpec.from_dict(data).to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match="premultiplied"):
        f3d.RenderPassInput(np.array([[[1., 0, 0, .5]]]), alpha_mode="premultiplied")


def test_unsupported_output_and_native_options_rejected(tmp_path):
    scene = _scene(pass_specs=[f3d.RenderPassSpec("color", "color", ("image",))],
                   pass_inputs={"image": f3d.RenderPassInput(np.zeros((1, 1, 4)))})
    for options in ({"certificate": True}, {"cache": str(tmp_path)}, {"emit_provenance": True}):
        with pytest.raises(f3d.MapSceneNativeUnavailable, match="render_passes"):
            scene.render(str(tmp_path / "invalid.png"), **options)
    for options in ({"certificate": True}, {"cache": str(tmp_path)}):
        with pytest.raises(ValueError, match="native certificates or render cache"):
            scene.render_passes(path=str(tmp_path / "invalid.png"), **options)
        original_recipe, original_plan = scene.recipe, scene.compiled_plan
        with pytest.raises(ValueError, match="native certificates or render cache"):
            scene.render_passes([f3d.RenderPassSpec("replacement", "color", ("new_image",))],
                                {"new_image": f3d.RenderPassInput(np.ones((1, 1, 4)))},
                                str(tmp_path / "invalid.png"), **options)
        assert scene.recipe is original_recipe
        assert scene.compiled_plan is original_plan
        assert scene.recipe.pass_specs[0].name == "color"
    scene.recipe.output.hdr = True
    with pytest.raises(ValueError, match="PNG"):
        scene.render_passes(path=str(tmp_path / "invalid.png"))
    assert not (tmp_path / "invalid.png").exists()


@pytest.mark.parametrize('operation,expected', [
    ('multiply', [7/24, 7/24, 31/96, 3/4]),
    ('screen', [11/24, 11/24, 53/96, 3/4]),
    ('alpha_over', [5/12, 1/3, 1/3, 3/4]),
])
def test_default_linear_blend_hand_computed(operation, expected):
    inputs = {'a': f3d.RenderPassInput(np.array([[[1/4, 1/2, 3/4, 1/2]]]), color_space='linear'),
              'b': f3d.RenderPassInput(np.array([[[1/2, 1/4, 1/8, 1/2]]]), color_space='linear')}
    compiled = compile_passes([f3d.RenderPassSpec('out', 'relief', ('a', 'b'), operation)], inputs, (1, 1))
    # These fractions are calculated in linear light; encode only the final RGB.
    encoded = [1.055 * value ** (1/2.4) - .055 for value in expected[:3]] + [expected[3]]
    np.testing.assert_allclose(execute_passes(json.dumps(compiled), (1, 1)), [[encoded]],
                               atol=FLOAT_ATOL, rtol=0)


def test_premultiplied_uint8_and_png16_exact(tmp_path):
    scene = _scene()
    premultiplied = f3d.RenderPassInput(np.array([[[32, 64, 96, 128]]], dtype=np.uint8),
                                      alpha_mode='premultiplied')
    scene.render_passes([f3d.RenderPassSpec('out', 'color', ('a',))], {'a': premultiplied},
                        str(tmp_path / 'premult.png'))
    assert np.asarray(load_png_rgba(tmp_path / 'premult.png')).tolist() == [[[64, 128, 191, 128]]]
    scene.recipe.output.bit_depth = 16
    scene.render_passes([f3d.RenderPassSpec('out', 'relief', ('a', 'b'), 'multiply', color_space='srgb')],
                        {'a': f3d.RenderPassInput(np.array([[[.25, .5, .75, 1.]]])),
                         'b': f3d.RenderPassInput(np.array([[[.5, .25, .125, 1.]]]))},
                        str(tmp_path / '16.png'))
    binary = (tmp_path / '16.png').read_bytes()
    assert binary[24] == 16
    offset, compressed = 8, bytearray()
    while offset < len(binary):
        length = struct.unpack('>I', binary[offset:offset+4])[0]
        if binary[offset+4:offset+8] == b'IDAT':
            compressed.extend(binary[offset+8:offset+8+length])
        offset += length + 12
    raw = zlib.decompress(compressed)
    assert raw[0] == 0  # Existing deterministic PNG encoder uses filter None.
    assert np.frombuffer(raw[1:], dtype='>u2').tolist() == [8192, 8192, 6144, 65535]


@pytest.mark.parametrize('failure', ['path', 'format', 'hdr', 'aovs', 'bit_depth', 'empty'])
def test_failed_replacement_preserves_recipe_and_plan(failure, tmp_path):
    original = f3d.RenderPassInput(np.ones((1, 1, 4)))
    scene = _scene(pass_specs=[f3d.RenderPassSpec('old', 'color', ('old_image',))],
                   pass_inputs={'old_image': original})
    old_plan = scene.compile_plan()
    if failure == 'format':
        scene.recipe.output.format = 'exr'
    elif failure == 'hdr':
        scene.recipe.output.hdr = True
    elif failure == 'aovs':
        scene.recipe.output.aovs = ('depth',)
    elif failure == 'bit_depth':
        scene.recipe.output.bit_depth = 12
    before = scene.recipe.to_dict()
    passes = [] if failure == 'empty' else [f3d.RenderPassSpec('new', 'color', ('new_image',))]
    inputs = {} if failure == 'empty' else {'new_image': f3d.RenderPassInput(np.zeros((1, 1, 4)))}
    with pytest.raises(ValueError):
        scene.render_passes(passes, inputs, None if failure == 'path' else str(tmp_path / 'invalid.png'))
    assert scene.recipe.to_dict() == before
    assert scene.compiled_plan is old_plan
    assert not (tmp_path / 'invalid.png').exists()


def test_snapshots_and_specs_have_value_equality_and_hashes():
    zero = f3d.RenderPassInput(np.zeros((1, 1, 4)))
    same = f3d.RenderPassInput(np.zeros((1, 1, 4)))
    one = f3d.RenderPassInput(np.ones((1, 1, 4)))
    assert zero == same and hash(zero) == hash(same)
    assert zero != one and len({zero, same, one}) == 2
    assert _scene(pass_inputs={'i': zero}, pass_specs=[f3d.RenderPassSpec('out', 'color', ('i',))]).recipe != (
        _scene(pass_inputs={'i': one}, pass_specs=[f3d.RenderPassSpec('out', 'color', ('i',))]).recipe)
    spec = f3d.RenderPassSpec('out', 'color', ('i',))
    same_spec = f3d.RenderPassSpec.from_dict(spec.to_dict())
    assert spec == same_spec and hash(spec) == hash(same_spec)
    assert len({spec, same_spec}) == 1


@pytest.mark.parametrize('unused_pass', [False, True])
def test_disconnected_passes_and_inputs_rejected(unused_pass):
    inputs = {'a': f3d.RenderPassInput(np.ones((1, 1, 4)))}
    passes = [f3d.RenderPassSpec('out', 'color', ('a',))]
    if unused_pass:
        passes.insert(0, f3d.RenderPassSpec('unused', 'relief', ('a',)))
    else:
        inputs['unused'] = inputs['a']
    with pytest.raises(ValueError, match='unused'):
        compile_passes(passes, inputs, (1, 1))


def test_v4_bundle_snapshots_are_deduplicated_and_canonical(tmp_path):
    image = f3d.RenderPassInput(np.random.default_rng(0).random((192, 256, 4)))
    passes = [f3d.RenderPassSpec('out', 'relief', ('a', 'b'), 'multiply')]
    scene = _scene(256, 192, pass_specs=passes, pass_inputs={'a': image, 'b': image})
    # Rendering, compilation and bundling must never materialize inline pixels.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(f3d.RenderPassInput, 'to_dict', lambda self: pytest.fail('inline pixel serialization'))
        scene.render_passes(path=str(tmp_path / 'original.png'))
        scene.save_bundle(tmp_path / 'first')
        loaded = f3d.MapScene.load_bundle(tmp_path / 'first.forge3d')
        loaded.render_passes(path=str(tmp_path / 'original.png'))
        loaded.save_bundle(tmp_path / 'second')
    bundle = tmp_path / 'first.forge3d'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['version'] == 4
    assets = list(bundle.glob('scene/render_pass_inputs/*.npy'))
    assert len(assets) == 1
    descriptor = image._asset_descriptor()
    assert manifest['checksums'][descriptor['asset']] == hashlib.sha256(assets[0].read_bytes()).hexdigest()
    for rel in ('scene/mapscene_recipe.json', 'scene/compiled_plan.json', descriptor['asset'], 'manifest.json'):
        assert (bundle / rel).read_bytes() == (tmp_path / 'second.forge3d' / rel).read_bytes()
    assert 'rgba' not in json.loads((bundle / 'scene/compiled_plan.json').read_text())['compiled_render_passes']['inputs']['a']
    assert loaded.to_dict() == scene.to_dict()
    assert loaded.recipe.pass_inputs['a'] == image


@pytest.mark.parametrize('tamper', ['bytes', 'path', 'shape', 'dtype', 'checksum', 'version'])
def test_snapshot_asset_boundary_rejects_tampering(tamper, tmp_path):
    image = f3d.RenderPassInput(np.ones((1, 1, 4)))
    scene = _scene(pass_specs=[f3d.RenderPassSpec('out', 'color', ('a',))], pass_inputs={'a': image})
    scene.save_bundle(tmp_path / 'bundle')
    root = tmp_path / 'bundle.forge3d'
    recipe = root / 'scene/mapscene_recipe.json'
    payload = json.loads(recipe.read_text())
    descriptor = payload['render_passes']['inputs']['a']
    if tamper == 'bytes':
        (root / descriptor['asset']).write_bytes(b'invalid')
    elif tamper == 'path':
        descriptor['asset'] = '../outside.npy'
    elif tamper == 'shape':
        descriptor['shape'] = [2, 1, 4]
    elif tamper == 'dtype':
        descriptor['dtype'] = 'object'
    else:
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if tamper == 'checksum':
            manifest['checksums'][descriptor['asset']] = '0' * 64
        else:
            manifest['version'] = 3
        manifest_path.write_text(json.dumps(manifest))
    recipe.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        f3d.MapScene.load_bundle(root)


@pytest.mark.parametrize('payload', [None, {}, {'passes': [], 'inputs': {}}])
def test_mapscene_rejects_v4_without_passes(payload, tmp_path):
    scene = _scene()
    scene.save_bundle(tmp_path / 'bundle')
    root = tmp_path / 'bundle.forge3d'
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['version'] == 3
    manifest['version'] = 4
    manifest_path.write_text(json.dumps(manifest))
    recipe_path = root / 'scene/mapscene_recipe.json'
    recipe = json.loads(recipe_path.read_text())
    if payload is not None:
        recipe['render_passes'] = payload
    recipe_path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='version 4.*nonempty render passes'):
        f3d.MapScene.load_bundle(root)


def test_summary_hash_matches_compiled_and_lists_passes_without_inline_pixels(monkeypatch):
    image = f3d.RenderPassInput(np.random.default_rng(0).random((384, 512, 4)))
    scene = _scene(512, 384, pass_specs=[f3d.RenderPassSpec('out', 'color', ('a',))],
                   pass_inputs={'a': image})
    plan = scene.compile_plan()
    with monkeypatch.context() as patch:
        patch.setattr(f3d.RenderPassInput, 'to_dict', lambda self: pytest.fail('inline summary pixels'))
        summary = f3d.recipe_manifest(scene)
        assert summary == f3d.recipe_manifest(scene.recipe)
        assert summary == f3d.recipe_manifest(scene.recipe.to_dict(include_pass_data=False))
    assert summary['recipe_hash'] == plan.recipe_hash
    assert summary['render_passes'] == json.loads(plan.render_passes_json)
    # Both standalone mapping forms yield the same hash without changing the caller.
    standalone = scene.to_dict()
    before = json.dumps(standalone, sort_keys=True)
    assert f3d.recipe_manifest(standalone) == summary
    assert json.dumps(standalone, sort_keys=True) == before
    assert f3d.recipe_manifest(scene.recipe.to_dict()) == summary
    scene.recipe.pass_inputs['a'] = f3d.RenderPassInput(np.zeros((384, 512, 4)))
    updated = f3d.recipe_manifest(scene)
    assert updated['recipe_hash'] != summary['recipe_hash']
    assert updated['recipe_hash'] == scene.compile_plan().recipe_hash


def test_no_pass_summary_keeps_existing_hash_and_fields():
    scene = _scene()
    summary = f3d.recipe_manifest(scene)
    assert 'render_passes' not in summary
    assert summary == f3d.recipe_manifest(scene.recipe.to_dict())
    assert summary['recipe_hash'] == scene.compile_plan().recipe_hash
