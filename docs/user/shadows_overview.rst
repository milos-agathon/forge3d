Shadow System Overview
======================

The current public Python shadow surface is centered on ``forge3d.Scene`` and
related scene controls. Older documentation that referenced a standalone
``forge3d.shadows`` package, ``ShadowManager``, or ``ShadowRenderer`` is
outdated.

## Public controls

* ``Scene.set_shadow_quality("off" | "low" | "medium" | "high")``
* ``Scene.enable_cloud_shadows()`` / ``disable_cloud_shadows()``
* ``Scene.is_cloud_shadows_enabled()``
* ``Scene.set_cloud_shadow_intensity(...)``
* ``Scene.set_cloud_shadow_softness(...)``

## Practical guidance

* Use ``set_shadow_quality(...)`` for the supported high-level shadow toggle.
* Use cloud-shadow controls when you want large-scale animated shadowing.
* For lower-level cascaded-shadow tuning, work through the native extension or
  the Rust renderer directly.

## Internal status

The renderer still contains lower-level CSM helpers and related implementation
code, but those APIs are not wrapped as a stable user-facing Python package at
this time.
