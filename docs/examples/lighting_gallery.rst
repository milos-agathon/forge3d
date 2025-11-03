Lighting Gallery
================

The lighting gallery renders a labeled mosaic of tiles showcasing direct-lighting
models and IBL variants on a simple mesh. It uses the high-level mesh tracer by
default and degrades gracefully to CPU when a GPU is unavailable.

Script
------

Path: ``examples/lighting_gallery.py``

What it demonstrates
--------------------

- Direct lighting models: Lambertian, Phong, Blinn-Phong
- IBL rotations when an HDR is present (e.g., ``assets/snow_field_4k.hdr``)
- CPU/GPU fallback via the high-level tracer (no native symbols required)
- Labeled tiles saved to a mosaic; optional per-tile PNGs

Usage
-----

.. code-block:: bash

   # Fast defaults (CPU friendly)
   python examples/lighting_gallery.py

   # Save to a custom output and per-tile directory
   python examples/lighting_gallery.py \
       --output examples/out/lighting_gallery.png \
       --outdir examples/out \
       --save-tiles

   # Specify a mesh and increase samples
   python examples/lighting_gallery.py \
       --mesh assets/bunny.obj \
       --tile-size 384 \
       --frames 6

Options
-------

- ``--output``: path of the mosaic PNG (default: ``lighting_gallery.png``)
- ``--outdir``: directory for optional per-tile images (default: ``examples/out``)
- ``--mesh``: OBJ mesh to render (default tries ``assets/bunny.obj``)
- ``--tile-size``: tile width/height in pixels (default: 320)
- ``--frames``: accumulation frames for the tracer (default: 4)
- ``--save-tiles``: also write individual tile PNGs

Assets
------

- Mesh: ``assets/bunny.obj`` (provided)
- Optional HDR: ``assets/snow_field_4k.hdr`` (provided); enables IBL tiles

Tips
----

- Lower ``--tile-size`` and ``--frames`` for faster, CI-friendly runs.
- Increase ``--frames`` for smoother results on GPUs.
