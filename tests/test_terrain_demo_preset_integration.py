# tests/test_terrain_demo_preset_integration.py
# Acceptance one-liner: simulate `examples/terrain_demo.py --preset outdoor_sun` + overrides
# by constructing a RendererConfig via load_renderer_config(preset_map, overrides).
# This mirrors the merge order in examples/terrain_demo.py::_build_renderer_config.
from __future__ import annotations

from pathlib import Path

from forge3d import presets
from forge3d.config import load_renderer_config


def test_acceptance_one_liner_outdoor_sun_presets_merge() -> None:
    # Equivalent of:
    #   python examples/terrain_demo.py --preset outdoor_sun \
    #       --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr
    preset_map = presets.get("outdoor_sun")
    cfg = load_renderer_config(
        preset_map,
        overrides={
            "brdf": "cooktorrance-ggx",
            "shadows": "csm",
            "cascades": 4,
            "hdr": "assets/sky.hdr",
        },
    )

    # Expectations: overrides win; preset-provided atmosphere sky is preserved
    assert cfg.shading.brdf == "cooktorrance-ggx"
    assert cfg.shadows.technique == "csm"
    assert cfg.shadows.cascades == 4
    assert cfg.atmosphere.sky == "hosek-wilkie"
    assert cfg.atmosphere.hdr_path == "assets/sky.hdr"

