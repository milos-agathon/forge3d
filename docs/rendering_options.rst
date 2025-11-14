.. docs/rendering_options.rst
.. Rendering configuration reference for forge3d
.. Exists to document RendererConfig defaults and CLI wiring for lighting and GI
.. RELEVANT FILES: python/forge3d/config.py, python/forge3d/__init__.py, examples/terrain_demo.py, src/render/params.rs

Rendering Options Overview
==========================

``RendererConfig`` encapsulates the lighting, shading, shadow, GI, and atmospheric controls that are now threaded through the Python API.
The Rust crate mirrors these structures in ``src/render/params.rs`` so JSON round-trips and unit tests can keep both sides aligned.

RendererConfig Summary
----------------------

.. list-table::
   :widths: 22 18 60
   :header-rows: 1

   * - Section
     - Default
     - Notes
   * - ``lighting.lights``
     - 1 × directional ``(-0.35,-1.0,-0.25)``
     - Supports ``directional``, ``point``, ``spot``, ``area-rect``, ``area-disk``, ``area-sphere``, and ``environment`` types. 
       Environment lights require ``hdr_path`` unless the atmosphere supplies one.
   * - ``lighting.exposure``
     - ``1.0``
     - ACES exposure multiplier. Propagates to terrain pipelines and fallback renderer.
   * - ``shading.brdf``
     - ``cooktorrance-ggx``
     - Enum matches Rust ``BrdfModel``. Additional scalars expose metallic, roughness, sheen, clearcoat, subsurface, and anisotropy mixes.
   * - ``RendererConfig.brdf_override``
     - ``None``
     - Forces the chosen BRDF globally when set (e.g. ``toon`` or ``ward``).
   * - ``shadows``
     - Enabled, ``pcf`` @ ``2048``px, ``4`` cascades
     - Filtered techniques enforce a minimum map size. ``csm`` requires at least two cascades.
   * - ``gi.modes``
     - ``["none"]``
     - Accepts lists such as ``["ibl","ssao"]``. Enabling ``ibl`` requires an environment HDR.
   * - ``atmosphere``
     - ``hosek-wilkie`` sky, fog disabled
     - Optional volumetric block with ``density``, ``phase`` (``isotropic`` or ``henyey-greenstein``), and anisotropy ``g``.

CLI Flags
---------

The terrain demo exposes configuration toggles that map directly into ``RendererConfig`` via ``examples/terrain_demo.py``:

.. list-table::
   :widths: 18 22 60
   :header-rows: 1

   * - Flag
     - Example
     - Effect
   * - ``--light``
     - ``--light type=directional,dir=0.2,0.8,-0.55,intensity=8,color=1,0.96,0.9``
     - Append one or more light definitions (repeatable). Keys map to the ``lighting.lights`` entries.
   * - ``--brdf``
     - ``--brdf disney-principled``
     - Overrides ``shading.brdf``.
   * - ``--preset``
     - ``--preset outdoor_sun --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr``
     - Initialize from a high-level preset, then apply CLI overrides. Preset merge precedes overrides; overrides always take precedence.
   * - ``--shadows`` / ``--shadow-map-res`` / ``--cascades``
     - ``--shadows pcf --shadow-map-res 2048 --cascades 4``
     - Select technique and atlas parameters.
   * - ``--pcss-blocker-radius`` / ``--pcss-filter-radius`` / ``--shadow-light-size`` / ``--shadow-moment-bias``
     - ``--pcss-blocker-radius 6.0 --pcss-filter-radius 12.0 --shadow-light-size 0.4 --shadow-moment-bias 0.0005``
     - Fine-tune PCSS penumbra and moment-based shadow filtering.
   * - ``--gi``
     - ``--gi ibl,ssao``
     - Enables one or more GI modes.
   * - ``--sky`` / ``--hdr``
     - ``--sky hdri --hdr assets/sky.hdr``
     - Sets sky model and HDR path. Required when ``--gi`` includes ``ibl`` without an environment light.
   * - ``--volumetric``
     - ``--volumetric density=0.02,phase=hg,g=0.7``
     - Supplies volumetric fog parameters.

Preset Examples
---------------

One-command acceptance (P7) using the terrain demo:

.. code-block:: bash

   python examples/terrain_demo.py --preset outdoor_sun \
       --brdf cooktorrance-ggx --shadows csm --cascades 4 --hdr assets/sky.hdr

Additional Viewer Fog Controls (P6‑10)
--------------------------------------

The interactive viewer exposes performance and quality toggles for the volumetric fog pass:

* ``:fog-half on|off`` — compute fog at half resolution and upsample to full-res for composition.
* ``:fog-edges on|off`` — enable depth-aware bilateral upsample to preserve edges (on by default).
* ``:fog-upsigma <float>`` — set bilateral depth sigma (default ``0.02``). Lower = sharper edges; higher = smoother.

Notes:

* In half-res mode, the renderer reduces internal ``max_steps`` heuristically to maintain quality at lower cost.
* Upsample is guided by the full-resolution depth buffer. When ``:fog-edges off``, a fast bilinear upsample is used instead.

Python Usage
------------

``Renderer`` now accepts config dictionaries, JSON files, or keyword mirrors::

    import forge3d as f3d

    renderer = f3d.Renderer(
        256, 256,
        light=[{
            "type": "directional",
            "direction": [0.0, -1.0, 0.0],
            "intensity": 2.0,
        }],
        brdf="lambert",
        gi=["ibl"],
        hdr="assets/sky.hdr",
        exposure=0.85,
    )

    config_dict = renderer.get_config()
    print(config_dict["lighting"]["lights"][0])

Lights can also be updated after construction::

    renderer.set_lights([
        {"type": "directional", "direction": [0.1, -1.0, 0.2], "intensity": 3.0},
        {"type": "point", "position": [2.0, 4.0, -1.0], "range": 15.0, "intensity": 1.5},
    ])

To load from disk, write a JSON document that matches the ``RendererConfig`` structure::

    {
        "lighting": {
            "lights": [
                { "type": "directional", "direction": [0.0, -1.0, 0.0] }
            ]
        },
        "shading": { "brdf": "toon", "roughness": 0.2, "metallic": 0.1 },
        "brdf_override": "toon"
    }

    renderer = f3d.Renderer(256, 256, config="render_config.json")

Both Python and Rust validators ensure directional lights provide ``direction``, positional lights include ``position``, filtered shadows retain practical resolutions, and shadow atlases stay within the 256 MiB budget with power-of-two sizes.

Viewer GI Commands Quick Reference
----------------------------------

The interactive viewer exposes a compact, single-terminal command interface for GI and debugging. See the project root ``p5.md`` for a full guide.

Basic toggles:

* ``:gi ssao on|off``
* ``:gi ssgi on|off``
* ``:gi ssr on|off``
* ``:viz material|normal|depth|gi``
* ``:viz-depth-max <float>``

SSAO/GTAO:

* ``:ssao-technique ssao|gtao``
* ``:ssao-radius <float>``
* ``:ssao-intensity <float>``
* ``:ssao-composite on|off``
* ``:ssao-mul <0..1>``

SSGI:

* ``:ssgi-steps <u32>``  ``:ssgi-radius <float>``  ``:ssgi-half on|off``
* ``:ssgi-temporal-alpha <0..1>``
* ``:ssgi-edges on|off``  ``:ssgi-upsample-sigma-depth <float>``  ``:ssgi-upsample-sigma-normal <float>``

SSR:

* ``:ssr-max-steps <u32>``  ``:ssr-thickness <float>``

Environment & capture:

* ``:ibl <path.hdr|path.exr>``
* ``:snapshot [path]``

Camera & framing (useful for golden images):

* ``:fov <degrees>``
* ``:cam-lookat ex ey ez tx ty tz [ux uy uz]``
* ``:size <width> <height>``

See also: the acceptance workflow in ``p5.md`` and the generator ``scripts/generate_golden_images.py``.
