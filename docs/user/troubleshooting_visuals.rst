Troubleshooting Visual Issues
============================

This guide uses only the APIs that are currently public.

General Checks
--------------

* Verify GPU availability with ``forge3d.has_gpu()`` and ``forge3d.device_probe()``
* Inspect runtime counters with ``Scene.get_stats()`` or ``ViewerHandle.get_stats()``
* Start from a minimal scene and enable one feature at a time

Scene Features You Can Toggle Publicly
--------------------------------------

* SSAO: ``set_ssao_enabled(...)`` and ``set_ssao_parameters(...)``
* SSGI: ``enable_ssgi()`` / ``disable_ssgi()``
* SSR: ``enable_ssr()`` / ``disable_ssr()``
* Bloom: ``enable_bloom()`` / ``set_bloom_settings(...)``
* DOF: ``enable_dof()`` / ``disable_dof()``
* OIT: ``enable_oit()`` / ``disable_oit()``
* Shadows: ``set_shadow_quality(...)``
* IBL: ``enable_ibl()`` / ``disable_ibl()``

Examples
--------

.. code-block:: python

   scene.set_ssao_enabled(True)
   scene.set_ssao_parameters(radius=2.0, intensity=1.0, bias=0.02)

   scene.enable_bloom()
   scene.set_bloom_settings(threshold=1.1, strength=0.6, radius=1.2)

   scene.enable_dof("medium")
   scene.set_shadow_quality("high")

Current Limitation
------------------

The native extension does define extra settings classes such as
``SSGISettings`` and ``SSRSettings``, but they are not currently re-exported on
the curated top-level Python surface. If you use them via
``forge3d._native.get_native_module()``, treat that path as internal.
