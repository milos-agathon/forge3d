Shadow Mapping
==============

There is no public ``forge3d.shadows`` Python module in the current package.

## Current public controls

Use ``forge3d.Scene`` for shadow-related toggles:

* ``set_shadow_quality("off" | "low" | "medium" | "high")``
* ``enable_cloud_shadows()`` / ``disable_cloud_shadows()``
* ``set_cloud_shadow_intensity(...)``
* ``set_cloud_shadow_softness(...)``

Example
-------

.. code-block:: python

   import forge3d as f3d

   scene = f3d.Scene(800, 600)
   scene.set_shadow_quality("high")
   scene.enable_cloud_shadows()
   scene.set_cloud_shadow_intensity(0.6)
   scene.set_cloud_shadow_softness(0.4)

## Internal status

Lower-level CSM helpers exist in the native extension and Rust renderer, but
they are not wrapped as a stable user-facing ``forge3d.shadows`` package.
