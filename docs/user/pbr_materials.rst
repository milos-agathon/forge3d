PBR Materials
=============

The current Python package does not expose a ``forge3d.pbr`` module. Use these
modules instead:

* ``forge3d.materials`` for ``PbrMaterial``
* ``forge3d.textures`` for texture-set construction
* ``forge3d.path_tracing`` for the deterministic CPU fallback renderer
* ``forge3d.buildings`` for building-material helpers

Example
-------

.. code-block:: python

   import numpy as np
   from forge3d.materials import PbrMaterial
   from forge3d.textures import build_pbr_textures

   albedo = np.full((4, 4, 4), [180, 120, 80, 255], dtype=np.uint8)
   texset = build_pbr_textures(base_color=albedo)
   material = PbrMaterial(
       base_color_factor=(1.0, 1.0, 1.0, 1.0),
       metallic_factor=0.0,
       roughness_factor=0.7,
   ).with_textures(texset)

   print(material)
