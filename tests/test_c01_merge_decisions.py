"""Owner-approved compatibility decisions for the C01 merge."""
from dataclasses import asdict,replace
import copy
import warnings

import numpy as np
import pytest
import forge3d as f3d
from forge3d import map_scene as ms
from forge3d.terrain_params import SamplingSettings


def _sampling():
    return SamplingSettings("Linear","Linear","Linear",1,"Repeat","Repeat","Repeat")


@pytest.mark.parametrize("axis", ("address_u","address_v","address_w"))
@pytest.mark.parametrize("value", ("ClampToEdge","MirrorRepeat"))
@pytest.mark.parametrize("construction", (True,False), ids=("construct","assign"))
def test_nondefault_address_settings_warn_and_preserve_values(axis,value,construction):
    original=_sampling()
    expected=asdict(original)
    expected[axis]=value
    with pytest.warns(DeprecationWarning,match=axis+".*deprecated.*no effect"):
        if construction:
            settings=replace(original,**{axis:value})
        else:
            settings=original
            setattr(settings,axis,value)
    assert asdict(settings)==expected
    with pytest.warns(DeprecationWarning,match="deprecated"):
        restored=SamplingSettings(**asdict(settings))
    assert asdict(restored)==expected


def test_default_address_settings_remain_silent_and_invalid_values_still_raise():
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        settings=_sampling()
        settings.address_u="Repeat"
    assert not observed
    with pytest.raises(ValueError,match="Invalid address_v"):
        replace(settings,address_v="invalid")


@pytest.mark.parametrize("cli", ({"msaa_samples":4},{"render_scale":.5},
    {"msaa_samples":4,"render_scale":.5},{"msaa_samples":1,"render_scale":1.0},
    {"msaa_samples":"ignored","render_scale":"ignored"}))
def test_mapscene_warns_for_present_sampling_keys_and_keeps_fixed_sampling(cli):
    from test_c01_review_gpu import _scene
    _,recipe=_scene()
    cli={"camera_mode":"mesh",**cli}
    recipe=replace(recipe,lighting=replace(recipe.lighting,settings={"cli_params":cli}))
    before=copy.deepcopy(recipe.lighting.settings)
    with pytest.warns(UserWarning,match="MapScene.*ignores.*msaa_samples=1.*render_scale=1") as caught:
        params=ms._build_mapscene_terrain_params(recipe,recipe.terrain.data,(128,128))
    for name in cli.keys()-{"camera_mode"}:
        assert name in str(caught[0].message)
    assert params.python_object.msaa_samples==1
    assert params.python_object.render_scale==1.0
    assert recipe.lighting.settings==before


def test_mapscene_without_sampling_keys_is_silent():
    from test_c01_review_gpu import _scene
    _,recipe=_scene()
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        params=ms._build_mapscene_terrain_params(recipe,recipe.terrain.data,(128,128))
    assert not observed
    assert params.python_object.msaa_samples==1 and params.python_object.render_scale==1.0


def test_ignored_recipe_sampling_keeps_native_provenance_and_pixels():
    from test_c01_review_regressions import _visibility_fixture
    _,recipe=_visibility_fixture(buildings=False)
    controlled=replace(recipe,lighting=replace(recipe.lighting,settings={**recipe.lighting.settings,
        "cli_params":{"camera_mode":"mesh","msaa_samples":4,"render_scale":.5}}))
    with ms._shared_terrain_render_context():
        with pytest.warns(UserWarning,match="MapScene.*ignores"):
            keyed=ms._render_terrain_renderer_result(controlled,controlled.terrain.data,emit_provenance=True)
        plain=ms._render_terrain_renderer_result(recipe,recipe.terrain.data,emit_provenance=True)
    assert keyed.source_map is not None and keyed.source_map.dtype==np.uint32
    assert keyed.source_map.shape==(recipe.output.height,recipe.output.width)
    assert keyed.aov_frame.has_source_id
    np.testing.assert_array_equal(keyed.source_map,plain.source_map)
    np.testing.assert_array_equal(keyed.rgba,plain.rgba)
    assert controlled.lighting.settings["cli_params"]["render_scale"]==.5



@pytest.mark.parametrize("axis", ("address_u", "address_v", "address_w"))
def test_copying_deprecated_address_values_is_silent_but_changes_warn(axis):
    with pytest.warns(DeprecationWarning, match=axis):
        settings = replace(_sampling(), **{axis: "ClampToEdge"})
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        copies = (replace(settings), replace(settings, anisotropy=2),
                  copy.copy(settings), copy.deepcopy(settings))
    assert not observed, "Copying an existing deprecated value must not warn again"
    for copied in copies:
        assert getattr(copied, axis) == "ClampToEdge"
        expected = asdict(settings)
        expected["anisotropy"] = copied.anisotropy
        assert asdict(copied) == expected
    with pytest.warns(DeprecationWarning, match=axis):
        changed = replace(settings, **{axis: "MirrorRepeat"})
    assert getattr(changed, axis) == "MirrorRepeat"
    with pytest.warns(DeprecationWarning, match=axis):
        setattr(settings, axis, "MirrorRepeat")
    with pytest.warns(DeprecationWarning, match=axis):
        SamplingSettings(**asdict(settings))


@pytest.mark.parametrize("samples,denoiser", ((1, "none"), (4, "atrous")))
def test_mapscene_ignored_sampling_warning_points_to_public_caller(tmp_path, samples, denoiser):
    import inspect
    from test_c01_review_regressions import _visibility_fixture
    _, recipe = _visibility_fixture(buildings=False)
    recipe = replace(recipe, output=replace(recipe.output, samples=samples, denoiser=denoiser),
        lighting=replace(recipe.lighting, settings={**recipe.lighting.settings,
            "cli_params": {"camera_mode": "mesh", "msaa_samples": 4, "render_scale": .5}}))
    scene = f3d.MapScene(recipe=recipe)
    with pytest.warns(UserWarning, match="MapScene.*ignores") as caught:
        call_line = inspect.currentframe().f_lineno + 1
        scene.render(tmp_path / f"warning-caller-{samples}.png")
    assert scene.last_render_backend == "gpu_terrain"
    for warning in caught:
        if "MapScene ignores" in str(warning.message):
            assert warning.filename == __file__
            assert warning.lineno == call_line
