"""Screen-frame parity and cloud-fixture measurements from physical frames."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


def screen_framing_scene(directory: Path, *, effects: bool):
    from test_recipe_goldens import _base_scene

    height = np.zeros((16, 16), dtype=np.float32)
    height[5:11, 5:11] = 0.35
    return _base_scene(directory, "screen-framing", width=128, height=80,
        heightmap=height,
        lighting_settings={"screen_space": {"ssr": {"enabled": effects, "intensity": 0.0}}})


def plateau_bounds(rgba):
    # This fixture's raised plateau and contour are red-dominant; its low
    # terrain is green. Measure its raster footprint, independent of brightness.
    mask = rgba[..., 0] > rgba[..., 1]
    y, x = np.nonzero(mask)
    assert len(x), "the authored plateau must be visible"
    return [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)]


def cloud_fixture_measurements(model, directory: Path | None = None):
    from _concordia_scene import capture
    import forge3d.map_scene as ms

    # Keep the route-control effect local to these measurement variants.
    # The canonical cloud recipe must attest only effects it actually uses.
    model = copy.deepcopy(model)
    model.recipe.lighting.settings.setdefault("screen_space", {})["ssr"] = {
        "enabled": True, "intensity": 0.0}
    clear_model = copy.deepcopy(model)
    clear_model.recipe.terrain.metadata["clouds"]["enabled"] = False
    # Clone recipe variants before compile_plan freezes mapping-proxy records.
    ambient_model = copy.deepcopy(clear_model)
    ambient_model.recipe.lighting.intensity = 0.0
    assert "ssr" in ms._mapscene_native_effects(model.recipe)
    assert "ssr" in ms._mapscene_native_effects(clear_model.recipe)
    cloudy, a = capture(model)
    clear, b = capture(clear_model)
    _, ambient = capture(ambient_model)
    np.testing.assert_array_equal(a["depth"], b["depth"])
    np.testing.assert_array_equal(a["surface_kind"], b["surface_kind"])
    assert cloudy.metadata["native_pipeline"]["effects"]["cloud_shadows"]
    assert not clear.metadata["native_pipeline"]["effects"]["cloud_shadows"]
    assert cloudy.metadata["native_pipeline"]["effects"]["ssr"]
    assert clear.metadata["native_pipeline"]["effects"]["ssr"]
    direct = b["geometry_hdr"][..., :3] - ambient["geometry_hdr"][..., :3]
    lit = (b["surface_kind"] == 1) & (direct.sum(axis=-1) > 0)
    assert lit.any(), "cloud sensitivity requires sunlit terrain"
    difference = np.abs(cloudy.rgba[..., :3].astype(float) - clear.rgba[..., :3].astype(float)).mean(axis=-1)
    from test_recipe_goldens import MEAN_ABS_MAX, SSIM_MIN
    from tests._ssim import ssim
    metrics = {
        "lit_pixels": int(lit.sum()),
        "changed_above_tolerance_pixels": int(np.count_nonzero(difference[lit] > MEAN_ABS_MAX)),
        "changed_above_tolerance_fraction": float(np.mean(difference[lit] > MEAN_ABS_MAX)),
        "lit_mean_abs_difference": float(difference[lit].mean()),
        "whole_image_mean_abs_difference": float(difference.mean()),
        "clear_cloudy_ssim": float(ssim(cloudy.rgba[..., :3], clear.rgba[..., :3], data_range=255.0)),
        "golden_mean_abs_max": MEAN_ABS_MAX, "golden_ssim_min": SSIM_MIN,
        "same_route": "native_interactive; zero-intensity SSR remains enabled in both frames",
        "depth_byte_identical": True, "surface_kind_byte_identical": True,
        "adapter": cloudy.metadata["native_pipeline"]["execution_report"]["adapter"],
    }
    if directory is not None:
        from PIL import Image
        directory.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cloudy.rgba).save(directory/"clouds-enabled.png")
        Image.fromarray(clear.rgba).save(directory/"clouds-disabled.png")
        Image.fromarray((lit*255).astype(np.uint8)).save(directory/"lit-terrain.png")
        Image.fromarray(np.clip(difference, 0, 255).astype(np.uint8)).save(directory/"absolute-difference.png")
        (directory/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
