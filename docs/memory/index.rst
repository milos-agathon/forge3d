Memory Management
=================

The current **public Python** memory surface in forge3d is intentionally small.

## Public API

- ``forge3d.memory_metrics()``
- ``forge3d.budget_remaining()``
- ``forge3d.utilization_ratio()``
- ``forge3d.override_memory_limit(...)``
- ``forge3d.device_probe()`` for adapter/device capabilities

Example::

   import forge3d as f3d

   print(f3d.memory_metrics())
   print(f3d.budget_remaining())
   print(f3d.utilization_ratio())

## Internal systems

The pages in this section describe Rust-side memory subsystems and design work
such as staging rings, memory pools, compressed texture loading, and virtual
texturing. They should be read as **implementation notes**, not as public
Python modules like ``forge3d.memory``, ``forge3d.compressed``, or
``forge3d.streaming``.

.. toctree::
   :maxdepth: 1

   staging_rings
   memory_pools
   compressed_textures
   virtual_texturing
   texture_memory_accounting
