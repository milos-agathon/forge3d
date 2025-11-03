# tests/test_renderer_apply_preset.py
# Unit tests for P7-04: Renderer.apply_preset merging and override precedence
# RELEVANT FILES: python/forge3d/__init__.py, python/forge3d/presets.py, python/forge3d/config.py
from __future__ import annotations

import pytest

import forge3d as f3d


@pytest.mark.parametrize(
    "preset_name, flat_overrides, nested_overrides, expect",
    [
        (
            "studio_pbr",
            {"brdf": "disney-principled", "shadows": "pcf", "shadow_map_res": 4096},
            {},
            {"shading.brdf": "disney-principled", "shadows.technique": "pcf", "shadows.map_size": 4096},
        ),
        (
            "outdoor_sun",
            {"brdf": "cooktorrance-ggx", "shadows": "csm", "cascades": 4, "hdr": "assets/sky.hdr"},
            {},
            {
                "shading.brdf": "cooktorrance-ggx",
                "shadows.technique": "csm",
                "shadows.cascades": 4,
                "atmosphere.sky": "hosek-wilkie",
                "atmosphere.hdr_path": "assets/sky.hdr",
            },
        ),
        (
            "toon_viz",
            {},
            {"shading": {"brdf": "toon"}, "shadows": {"technique": "hard", "map_size": 1024}},
            {"shading.brdf": "toon", "shadows.technique": "hard", "shadows.map_size": 1024},
        ),
    ],
)
def test_apply_preset_merge_and_overrides(preset_name, flat_overrides, nested_overrides, expect):
    r = f3d.Renderer(320, 200)
    r.apply_preset(preset_name, **nested_overrides, **flat_overrides)
    cfg = r.get_config()  # dict

    # Helper to retrieve nested keys from dict
    def _get(d: dict, dotted: str):
        cur = d
        for part in dotted.split("."):
            assert isinstance(cur, dict), f"Expected dict at {part} in {dotted}"
            cur = cur[part]
        return cur

    for dotted, val in expect.items():
        assert _get(cfg, dotted) == val


def test_apply_preset_invalid_name_raises():
    r = f3d.Renderer(64, 32)
    with pytest.raises(Exception):
        r.apply_preset("not_a_preset")
