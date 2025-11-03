IBL Gallery
==========

The IBL gallery demonstrates image-based lighting using an HDR environment. It
supports three modes and renders real images in all cases:

- Rotation: native terrain pipeline when available, otherwise mesh tracer fallback
- Roughness: mesh tracer sweep with HDR tinting
- Metallic: mesh tracer comparison of metallic vs dielectric across roughness

Script
------

Path: ``examples/ibl_gallery.py``

What it demonstrates
--------------------

- ``IBL.from_hdr`` (rotation sweep) on a synthetic DEM when native symbols are present
- Mesh tracer HDR integration (CPU-friendly) for roughness/metallic sweeps
- Cross-platform operation with robust PNG saving and optional per-tile outputs

Usage
-----

.. code-block:: bash

   # Rotation sweep (terrain native when available)
   python examples/ibl_gallery.py --mode rotation --hdr assets/snow_field_4k.hdr

   # Roughness sweep using mesh tracer
   python examples/ibl_gallery.py --mode roughness --hdr assets/snow_field_4k.hdr

   # Metallic vs dielectric comparison using mesh tracer
   python examples/ibl_gallery.py --mode metallic --hdr assets/snow_field_4k.hdr

   # All modes; auto-pick HDR from assets if not provided
   python examples/ibl_gallery.py --mode all

Options
-------

- ``--hdr``: path to HDRI (optional; defaults to ``assets/snow_field_4k.hdr`` if present)
- ``--outdir``: directory for per-tile images (default: ``examples/out``)
- ``--output``: output PNG path (when using a single mode)
- ``--mode``: one of ``rotation``, ``roughness``, ``metallic``, or ``all``
- ``--tile-size``: tile width/height in pixels (default: 400)
- ``--rotation-steps``: number of rotation angles (default: 8)
- ``--roughness-steps``: number of steps in roughness sweep (default: 10)

How it picks native vs fallback
-------------------------------

- If a GPU is available and the required native symbols exist (``Session``,
  ``TerrainRenderer``, ``MaterialSet``, ``IBL``, ``TerrainRenderParams``), the
  rotation mode uses the terrain renderer for each tile with a different IBL
  rotation.
- Otherwise it uses the high-level mesh tracer with HDR tinting (CPU-friendly).

Assets
------

- ``assets/snow_field_4k.hdr`` – default HDRI for the rotation and mesh tracer modes.
- ``assets/bunny.obj`` – default mesh for the mesh tracer.
