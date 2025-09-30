Vector Picking & OIT
====================

This page shows how to render vector primitives (points, lines) using Weighted Order-Independent Transparency (OIT) and how to produce a picking (ID buffer) render to identify objects by clicking.

Prerequisites
-------------
- A GPU/device that supports the required features. You can check availability from Python via ``forge3d.is_weighted_oit_available()``.
- Forge3D built with the ``weighted-oit`` Cargo feature (wheel builds may include this).

Quick Python Demo
-----------------
The simplest way to try OIT and picking is to call the built-in helper:

.. code-block:: python

    import forge3d as f3d

    # Optional: control point impostor mode (5 = sphere impostor) and LOD threshold in pixels
    f3d.set_point_shape_mode(5)         # 0=circle, 4=texture, 5=sphere impostor
    f3d.set_point_lod_threshold(24.0)   # fall back to circle when point size < 24px

    if not f3d.is_weighted_oit_available():
        print("Weighted OIT is not available on this build/platform")
    else:
        rgba, pick_id = f3d.vector_oit_and_pick_demo(640, 360)
        print("center pick id:", pick_id)
        f3d.numpy_to_png("vector_oit_demo.png", rgba)

Single-call API (RGBA + Pick Map)
---------------------------------
If you already have your own points and polylines, you can render both the OIT image and a full pick map in one call:

.. code-block:: python

    import forge3d as f3d

    points = [(-0.5, -0.5), (0.4, 0.2)]
    colors = [(1.0, 0.2, 0.2, 0.9), (0.2, 0.8, 1.0, 0.7)]
    sizes  = [24.0, 32.0]
    polylines = [[(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)]]
    poly_colors = [(0.1, 0.9, 0.3, 0.6)]
    poly_widths = [8.0]

    rgba, pick = f3d.vector_render_oit_and_pick_py(
        640, 360,
        points_xy=points,
        point_rgba=colors,
        point_size=sizes,
        polylines=polylines,
        polyline_rgba=poly_colors,
        stroke_width=poly_widths,
        base_pick_id=1,
    )
    f3d.numpy_to_png("vector_oit_combined.png", rgba)

High-level Convenience Wrapper
------------------------------
For ergonomic batching, use the ``VectorScene`` helper to collect primitives and render OIT and/or pick maps:

.. code-block:: python

    from forge3d import VectorScene, set_point_shape_mode, set_point_lod_threshold, numpy_to_png

    set_point_shape_mode(5)
    set_point_lod_threshold(24.0)

    vs = VectorScene()
    vs.add_point(-0.5, -0.5, (1.0, 0.2, 0.2, 0.9), 24.0)
    vs.add_point(0.4, 0.2, (0.2, 0.8, 1.0, 0.7), 32.0)
    vs.add_polyline([(-0.8, -0.8), (0.8, 0.5), (0.4, 0.8)], (0.1, 0.9, 0.3, 0.6), 8.0)

    rgba, pick = vs.render_oit_and_pick(640, 360, base_pick_id=1)
    numpy_to_png("vector_oit_vs.png", rgba)

What it does
------------
- Renders a sample polyline and two instanced points into two MRT targets using Weighted OIT:
  - Accumulation target: ``Rgba16Float`` (weighted color + weight sum in alpha)
  - Revealage target: ``R16Float`` (product of (1 - alpha))
- Composes the two targets to an ``RGBA8`` output image.
- Renders a second pass to a ``R32Uint`` ID buffer and returns the pick ID at the center pixel.

Manual Setup (Advanced)
-----------------------
If you wish to build your own OIT/picking passes, the core building blocks are:

- Renderers
  - ``forge3d.vector.PointRenderer``
  - ``forge3d.vector.LineRenderer``

- OIT utilities
  - ``forge3d.vector.oit.WeightedOIT`` (creates MRT textures and compose pipeline)

- Typical flow

.. code-block:: text

    Accumulation RenderPass (color0=Rgba16Float, color1=R16Float)
      -> LineRenderer.render_oit(...)
      -> PointRenderer.render_oit(...)

    Compose RenderPass (color=Rgba8)
      -> WeightedOIT.compose(...)

    Picking RenderPass (color=R32Uint)
      -> LineRenderer.render_pick(...)
      -> PointRenderer.render_pick(...)

Performance Tips
----------------
- Use sphere impostors for large points for good quality; set a reasonable ``lod_threshold`` to switch to circle rendering when points are small.
- Keep line alpha in a sensible range (e.g., 0.4–0.8) for stable transparency in weighted blending.
- For interactive picking, render the ID buffer at the same resolution as your main target and read a single pixel under the cursor.

Troubleshooting
---------------
- If ``vector_oit_and_pick_demo`` raises an error about OIT availability, the build/platform may not include the required GPU features. Use ``f3d.is_weighted_oit_available()`` to gate tests/demos.
- Some CI environments do not expose a GPU; skip tests that require OIT in such cases (see the perf test in ``tests/perf/test_vector_oit.py``).
