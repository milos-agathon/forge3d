# tests/test_terrain_demo_cli_smoke.py
# End-to-end smoke test for examples/terrain_demo.py CLI config wiring.
# Loads the module by path, constructs a minimal argparse.Namespace and invokes
# _build_renderer_config(args) to validate preset merge order and overrides.
from __future__ import annotations

import types
from pathlib import Path
import importlib.util
import argparse

from forge3d.config import RendererConfig


def _load_module_by_path(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("terrain_demo", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_terrain_demo_preset_cli_smoke() -> None:
    repo = Path(__file__).resolve().parents[1]
    mod = _load_module_by_path(repo / "examples" / "terrain_demo.py")

    # Build a minimal argparse.Namespace matching _build_renderer_config usage
    args = argparse.Namespace(
        light=[],
        exposure=1.0,
        brdf="cooktorrance-ggx",
        shadows="csm",
        shadow_map_res=None,
        cascades=4,
        pcss_blocker_radius=None,
        pcss_filter_radius=None,
        shadow_light_size=None,
        shadow_moment_bias=None,
        gi=None,
        sky=None,
        hdr="assets/sky.hdr",
        volumetric=None,
        preset="outdoor_sun",
    )

    cfg: RendererConfig = mod._build_renderer_config(args)  # type: ignore[attr-defined]

    # Expectations identical to acceptance test
    assert cfg.shading.brdf == "cooktorrance-ggx"
    assert cfg.shadows.technique == "csm"
    assert cfg.shadows.cascades == 4
    assert cfg.atmosphere.sky == "hosek-wilkie"
    assert cfg.atmosphere.hdr_path == "assets/sky.hdr"
