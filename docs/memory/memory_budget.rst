Memory Budget Management
========================

The current Python API exposes budget control through ``forge3d.mem`` and the
top-level re-exports in ``forge3d``.

## Public helpers

* ``forge3d.memory_metrics()``
* ``forge3d.budget_remaining()``
* ``forge3d.utilization_ratio()``
* ``forge3d.override_memory_limit(bytes_or_none)``

Example::

   import forge3d as f3d

   print(f3d.memory_metrics())
   f3d.override_memory_limit(256 * 1024 * 1024)
   print(f3d.budget_remaining())

Passing ``None`` to ``override_memory_limit`` restores the default behavior.
