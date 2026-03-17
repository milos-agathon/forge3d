GPU Memory Management Guide
===========================

This page reflects the **current public Python surface**, not the older
experimental memory APIs.

## Public Python telemetry

Use these helpers today:

* ``forge3d.memory_metrics()``
* ``forge3d.budget_remaining()``
* ``forge3d.utilization_ratio()``
* ``forge3d.override_memory_limit(...)``
* ``forge3d.device_probe()``

Example::

   import forge3d as f3d

   print(f3d.device_probe())
   print(f3d.memory_metrics())
   print(f3d.budget_remaining())

## Scope

Older references to modules such as ``forge3d.async_readback``,
``forge3d.memory``, or ``forge3d.shadows`` are not part of the current public
Python API. The corresponding mechanisms are implemented in Rust internals and
feature-gated subsystems.
