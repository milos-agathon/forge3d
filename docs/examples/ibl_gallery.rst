IBL Gallery
==========

The IBL gallery demonstrates image-based lighting using an HDR environment and a
BRDF tile-based renderer. It supports three modes and builds all panels from the
same M4-style IBL resources (prefiltered environment, irradiance, and DFG LUT):

- Rotation: sweep HDR environment rotation across tiles
- Roughness: sweep material roughness (0.0 → 1.0) for dielectric and metallic rows
- Metallic: compare metallic vs dielectric at fixed roughness samples

Script
------

Path: ``examples/ibl_gallery.py``

What it demonstrates
--------------------

- CPU-side IBL evaluation using precomputed M4 resources (equirect → cubemap,
  prefilter chain, irradiance cubemap, DFG LUT)
- BRDF tile-style panels for rotation, roughness, and metallic comparisons
- Robust PNG/NPY saving via :func:`forge3d.numpy_to_png` fallbacks

Usage
-----

.. code-block:: bash

   # Rotation sweep (BRDF tile IBL panels)
   python examples/ibl_gallery.py --mode rotation --hdr assets/snow_field_4k.hdr

   # Roughness sweep (dielectric + metallic rows)
   python examples/ibl_gallery.py --mode roughness --hdr assets/snow_field_4k.hdr

   # Metallic vs dielectric comparison
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

Requirements and fallbacks
--------------------------

- The gallery uses the BRDF tile renderer path exposed by :mod:`forge3d`. It
  requires a GPU, a working native module, and ``forge3d.render_brdf_tile_full``
  to be available.
- M4 IBL helper functions are loaded from ``examples/m4_generate.py``. If this
  file is missing or the helpers cannot be imported, ``ibl_gallery.py`` fails
  early with a clear ``SystemExit`` message.
- When the BRDF tile renderer or M4 helpers are unavailable, tests and CI treat
  the IBL gallery as an optional example and will skip its smoke test.

Assets
------

- ``assets/snow_field_4k.hdr`` – default HDRI for all IBL gallery modes.
