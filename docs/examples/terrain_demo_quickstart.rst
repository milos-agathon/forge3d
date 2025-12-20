.. docs/examples/terrain_demo_quickstart.rst
.. Terrain demo quickstart mirroring examples/terrain_demo.py
.. Exists to document required imports and parameter defaults for the public API
.. RELEVANT FILES: examples/terrain_demo.py, python/forge3d/__init__.py, python/forge3d/terrain_params.py, src/terrain_renderer.rs

============================
Terrain Demo Quickstart
============================

This quickstart mirrors ``examples/terrain_demo.py`` so you can verify that the public ``forge3d`` API exposes every class required for an offscreen terrain render.

Imports and Availability
-------------------------

``examples/terrain_demo.py`` checks that the following symbols exist on the top-level package:

``Session``, ``TerrainRenderer``, ``MaterialSet``, ``IBL``, ``Colormap1D``, ``TerrainRenderParams``, ``OverlayLayer``, ``LightSettings``, ``IblSettings``, ``ShadowSettings``, ``TriplanarSettings``, ``PomSettings``, ``LodSettings``, ``SamplingSettings``, ``ClampSettings``.

Build the PyO3 extension via ``maturin develop --release`` and confirm the bindings:

.. code-block:: bash

   python -c "import forge3d as f; required = [
       'Session','TerrainRenderer','MaterialSet','IBL','Colormap1D','TerrainRenderParams',
       'OverlayLayer','LightSettings','IblSettings','ShadowSettings','TriplanarSettings',
       'PomSettings','LodSettings','SamplingSettings','ClampSettings'
   ]; print(all(hasattr(f, n) for n in required))"

Parameters and Defaults
------------------------

Create dataclasses from ``forge3d.terrain_params`` to mirror the example defaults:

.. code-block:: python

   from forge3d import (
       Session,
       TerrainRenderer,
       MaterialSet,
       IBL,
       Colormap1D,
       OverlayLayer,
       TerrainRenderParams,
   )
   from forge3d.terrain_params import (
       TerrainRenderParams as TerrainRenderParamsConfig,
       LightSettings,
       IblSettings,
       ShadowSettings,
       TriplanarSettings,
       PomSettings,
       LodSettings,
       SamplingSettings,
       ClampSettings,
   )

   cmap = Colormap1D.from_stops(
       [(0.0, "#1e3a5f"), (0.5, "#6ca365"), (1.0, "#f5f1d0")],
       (0.0, 1.0),
   )
   overlay = OverlayLayer.from_colormap1d(cmap, strength=0.5)

   params = TerrainRenderParamsConfig(
       size_px=(256, 256),
       render_scale=1.0,
       msaa_samples=4,
       z_scale=1.0,
       cam_target=[0.0, 0.0, 0.0],
       cam_radius=6.0,
       cam_phi_deg=135.0,
       cam_theta_deg=40.0,
       fov_y_deg=55.0,
       clip=(0.1, 500.0),
       light=LightSettings("Directional", 135.0, 40.0, 3.0, [1.0, 0.97, 0.92]),
       ibl=IblSettings(True, 1.0, 0.0),
       shadows=ShadowSettings(
           True, "PCSS", 1024, 2, 500.0, 1.0, 0.8,
           0.002, 0.001, 0.3, 1e-4, 0.5, 2.0, 0.9,
       ),
       triplanar=TriplanarSettings(6.0, 4.0, 1.0),
       pom=PomSettings(True, "Occlusion", 0.05, 12, 40, 4, True, True),
       lod=LodSettings(0, 0.0, -0.5),
       sampling=SamplingSettings("Linear", "Linear", "Linear", 8, "Repeat", "Repeat", "Repeat"),
       clamp=ClampSettings((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
       overlays=[overlay],
       exposure=1.0,
       gamma=2.2,
       colormap_strength=0.5,
   )

Render Workflow
----------------

#. Instantiate ``Session`` with ``window=False`` for headless usage.

#. Build ``MaterialSet.terrain_default()`` and ``IBL.from_hdr`` for lighting.

#. Create a synthetic or DEM-backed heightmap as a ``float32`` NumPy array.

#. Wrap the dataclass inside ``TerrainRenderParams(params)`` and dispatch ``TerrainRenderer.render_terrain_pbr_pom``.

#. Save the generated ``Frame`` via ``frame.save("out.png")`` for RGBA8 or ``frame.save("out.exr")`` for RGBA16F (HDR, when the images feature is enabled), or convert to NumPy with ``frame.to_numpy()``.

This mirrors ``examples/terrain_demo.py`` and exercises every exported symbol listed above.

MSAA Compatibility
------------------
Forge3D automatically selects the highest supported sample count for the render target. When ``--msaa`` requests an unsupported value (for example, 8 on hardware that only exposes {1, 4}), the renderer downgrades to the nearest supported count, logs the decision, and always resolves to a single-sample texture for readback.

