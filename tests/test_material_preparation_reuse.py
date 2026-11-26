"""Source edits and caller mutation cannot make scoped preparation stale."""
from dataclasses import replace

import numpy as np
from PIL import Image

import forge3d as f3d
from forge3d._material_preparation import material_preparation_context


def test_scoped_preparation_observes_texture_edits_and_isolates_results(tmp_path, monkeypatch):
    path = tmp_path / "albedo.png"
    Image.new("RGB", (2, 2), (100, 60, 20)).save(path)
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    layer = f3d.TerrainMaterialLayer("ground", (255, 255, 255),
        texture=f3d.MaterialTexture(str(path), (4, 4), "srgb", "albedo", variation_seed=None))
    style = f3d.TerrainMaterialStyle(f3d.TerrainMaterialRecipe((layer,)), grid)
    heights = np.zeros((4, 4))
    reference = style.prepare(heights).rgba.copy()
    original = f3d.TerrainMaterialRecipe.prepare
    evaluations = []
    def observe(self, *args, **kwargs):
        evaluations.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(f3d.TerrainMaterialRecipe, "prepare", observe)
    with material_preparation_context():
        first = style.prepare(heights)
        np.testing.assert_array_equal(first.rgba, reference)
        first.rgba[:] = 0
        np.testing.assert_array_equal(style.prepare(heights).rgba, reference)
        np.testing.assert_array_equal(style.apply(heights, valid_mask=np.ones(heights.shape, bool)).rgba, reference)
        assert len(evaluations) == 1
        Image.new("RGB", (2, 2), (20, 80, 150)).save(path)
        changed = style.prepare(heights).rgba
        assert np.any(changed != reference)
    np.testing.assert_array_equal(changed, style.prepare(heights).rgba)


def test_scoped_preparation_observes_heights_validity_and_denied_retention(monkeypatch):
    grid = f3d.MaterialGrid("EPSG:32632", (500000, 4600000), (2, 2), "m", "m")
    mask = f3d.MaterialMask("ramp", field="elevation", units="m", lower=0, upper=10)
    material = f3d.TerrainMaterialRecipe((f3d.TerrainMaterialLayer("ground", (40, 60, 80)),
        f3d.TerrainMaterialLayer("high", (200, 180, 160), mask)))
    style = f3d.TerrainMaterialStyle(material, grid)
    source = np.zeros((4, 4))
    valid = np.ones(source.shape, bool)
    with material_preparation_context():
        low = style.apply(source, valid_mask=valid).rgba.copy()
        source[:] = 10
        high = style.apply(source, valid_mask=valid).rgba
        assert np.any(low != high)
        valid[0, 0] = False
        assert not style.apply(source, valid_mask=valid).valid_mask[0, 0]
    from forge3d._native import get_native_module
    native = get_native_module()
    def deny(*args):
        raise native.MemoryBudgetExceeded("optional retained preparation budget")
    monkeypatch.setattr(native, "_reserve_label_depth_host_allocation", deny)
    with material_preparation_context():
        np.testing.assert_array_equal(style.prepare(source).rgba,
                                      replace(style).prepare(source).rgba)
