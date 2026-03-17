Post-Processing Effects
=======================

There is no public ``forge3d.postfx`` Python module in the current package.

## Current public controls

Post-processing features are exposed primarily on ``forge3d.Scene``:

* ``enable_bloom()`` / ``disable_bloom()`` / ``set_bloom_settings(...)``
* ``enable_dof()`` / ``disable_dof()``
* ``enable_ssr()`` / ``disable_ssr()``
* ``enable_ssgi()`` / ``disable_ssgi()``
* ``set_ssao_enabled(...)`` / ``set_ssao_parameters(...)``

The interactive viewer also exposes TAA-related IPC helpers in
``forge3d.viewer_ipc`` such as ``set_taa_enabled(...)`` and
``set_taa_params(...)``.

Example::

   import forge3d as f3d

   scene = f3d.Scene(800, 600)
   scene.enable_bloom()
   scene.set_bloom_settings(threshold=1.1, strength=0.6, radius=1.2)
   scene.enable_dof("medium")
   scene.enable_ssr()
   rgba = scene.render_rgba()

## Internal status

The Rust renderer does have a post-processing chain and resource-pool
implementation, but it is not wrapped as a stable standalone Python API.
