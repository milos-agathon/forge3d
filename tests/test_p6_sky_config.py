# tests/test_p6_sky_config.py
# P6 Phase 1: Sky configuration round-trip tests
# Ensures that sky model selection is plumbed through RendererConfig.atmosphere.sky
# and that flat overrides (sky=...) map correctly into nested config.

from __future__ import annotations

import forge3d as f3d
from forge3d.config import load_renderer_config


def test_p6_sky_renderer_roundtrip_variants() -> None:
    """Round-trip atmosphere.sky through the Python Renderer config.

    This verifies that the flat `sky=` keyword is normalized and ends up
    in the nested `atmosphere.sky` field returned by `Renderer.get_config()`
    for the three primary sky modes.
    """

    # Hosek-Wilkie analytic sky
    renderer_hw = f3d.Renderer(16, 16, sky="hosek-wilkie")
    cfg_hw = renderer_hw.get_config()
    assert cfg_hw["atmosphere"]["sky"] == "hosek-wilkie"

    # Preetham analytic sky
    renderer_pre = f3d.Renderer(16, 16, sky="preetham")
    cfg_pre = renderer_pre.get_config()
    assert cfg_pre["atmosphere"]["sky"] == "preetham"

    # HDRI sky requires an HDR path but should round-trip the mode
    renderer_hdri = f3d.Renderer(16, 16, sky="hdri", hdr="env.hdr")
    cfg_hdri = renderer_hdri.get_config()
    assert cfg_hdri["atmosphere"]["sky"] == "hdri"
    assert cfg_hdri["atmosphere"]["hdr_path"] == "env.hdr"


def test_p6_sky_load_renderer_config_overrides() -> None:
    """Directly exercise load_renderer_config sky overrides.

    This validates that flat overrides using `sky=` are correctly
    normalized and stored on AtmosphereParams.sky, including aliases.
    """

    # Alias "hosekwilkie" should normalize to "hosek-wilkie"
    cfg_hw = load_renderer_config(None, {"sky": "hosekwilkie"})
    assert cfg_hw.atmosphere.sky == "hosek-wilkie"

    # Preetham by canonical name
    cfg_pre = load_renderer_config(None, {"sky": "preetham"})
    assert cfg_pre.atmosphere.sky == "preetham"

    # HDRI sky with HDR path provided via hdr= alias
    cfg_hdri = load_renderer_config(None, {"sky": "hdri", "hdr": "env.hdr"})
    assert cfg_hdri.atmosphere.sky == "hdri"
    assert cfg_hdri.atmosphere.hdr_path == "env.hdr"
