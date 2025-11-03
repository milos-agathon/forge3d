.. docs/user/presets_overview.rst
.. High-level presets for lighting/shading (P7)
.. RELEVANT FILES: python/forge3d/presets.py, python/forge3d/__init__.py, examples/terrain_demo.py

=================
Presets Overview
=================

Forge3D ships a small set of high-level presets to quickly configure lighting,
shading, and sky settings. Presets are provided as Python mappings compatible
with :class:`forge3d.config.RendererConfig`, and can also be applied directly
to a live :class:`forge3d.Renderer`.

What presets provide
--------------------

- Studio PBR (``studio_pbr``):
  Directional key, Disney Principled BRDF, PCF shadows. Atmosphere disabled by
  default. GI modes are empty to avoid asset requirements unless an HDR is provided.

- Outdoor Sun (``outdoor_sun``):
  Hosek–Wilkie sky, directional "sun", Cascaded Shadows (CSM), GGX BRDF.
  Atmosphere sky is enabled; GI modes left empty by default.

- Toon Visualization (``toon_viz``):
  Toon BRDF, hard shadows, no GI, flat background (atmosphere disabled).

Quick start (Python)
--------------------

List available presets and load one as a mapping::

    from forge3d import presets
    from forge3d.config import RendererConfig

    print(presets.available())
    cfg = RendererConfig.from_mapping(presets.get("outdoor_sun"))
    cfg.validate()  # Raises if invalid (e.g., missing HDR when required)

Apply preset to a live renderer and merge overrides::

    import forge3d as f3d

    r = f3d.Renderer(1280, 720)
    # Merge order: current → preset → overrides
    r.apply_preset(
        "outdoor_sun",
        brdf="cooktorrance-ggx",   # flat override mapped to shading.brdf
        shadows="csm",             # flat override mapped to shadows.technique
        cascades=4,                 # flat override mapped to shadows.cascades
        hdr="assets/sky.hdr",      # flat override mapped to atmosphere.hdr_path
    )

CLI usage (terrain demo)
------------------------

``examples/terrain_demo.py`` wires a ``--preset`` flag. Preset settings are
applied first; subsequent CLI flags override the preset::

    python examples/terrain_demo.py --preset outdoor_sun \
        --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr

Rules and merge precedence
--------------------------

- Preset is the base, overrides always win.
- Flat keys (``brdf``, ``shadows``, ``cascades``, ``hdr``, ``gi``) are mapped
  into their schema locations through :func:`forge3d.config.load_renderer_config`.
- Validation enforces sensible ranges (e.g. CSM cascades [2..4]) and asset
  requirements (e.g. IBL needs an environment HDR).

Examples and galleries
----------------------

Presets are a convenient starting point for the gallery scripts:

- ``examples/lighting_gallery.py`` — direct lighting and IBL variants
- ``examples/shadow_gallery.py`` — Hard, PCF, PCSS, VSM, EVSM, MSM, CSM compared
- ``examples/ibl_gallery.py`` — native terrain IBL rotation + mesh-tracer sweeps

API reference
-------------

Presets are defined in :mod:`forge3d.presets`::

    from forge3d import presets

    presets.available()   # ["studio_pbr", "outdoor_sun", "toon_viz"]
    presets.get("studio_pbr")  # Mapping compatible with RendererConfig.from_mapping()

They cooperate with :func:`forge3d.config.load_renderer_config` and
:class:`forge3d.Renderer.apply_preset` for end-to-end configuration.
