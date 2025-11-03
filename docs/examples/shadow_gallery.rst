Shadow Gallery
=============

The shadow gallery renders a mosaic that compares multiple shadow techniques side
by side using the terrain renderer when available, with graceful placeholder
fallbacks in CPU-only environments.

Script
------

Path: ``examples/shadow_gallery.py``

What it demonstrates
--------------------

- Shadow techniques: Hard, PCF, PCSS, VSM, EVSM, MSM, CSM
- Consistent lighting and camera setup for visual comparison
- Terrain-based native pipeline when GPU/native types are available
- Labeled tiles, and optional per-tile saves

Usage
-----

.. code-block:: bash

   # Render all supported techniques to a single mosaic
   python examples/shadow_gallery.py

   # Focus on a subset of techniques
   python examples/shadow_gallery.py --techniques Hard PCF PCSS

   # Larger shadow maps and tiles
   python examples/shadow_gallery.py --map-res 4096 --tile-size 800

Options
-------

- ``--output``: path of the mosaic PNG (default: ``shadow_gallery.png``)
- ``--outdir``: directory for optional per-tile images
- ``--hdr``: path to an environment HDR (default: ``assets/snow_field_4k.hdr``)
- ``--techniques``: subset of techniques to render
- ``--map-res``: shadow map resolution (power of two; default 2048)
- ``--tile-size``: tile width/height in pixels (default 512)
- ``--cols``: number of columns in the mosaic (default 3)

Notes
-----

- When GPU/native symbols are available (``Session``, ``TerrainRenderer``, ``MaterialSet``, ``IBL``, ``TerrainRenderParams``), the script
  renders real terrain tiles. Otherwise, it produces labeled placeholders so the
  page is still generated.
- To keep CI fast, prefer ``--map-res 2048`` or lower and ``--tile-size 384``.
