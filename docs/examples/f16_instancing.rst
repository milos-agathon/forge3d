F16 Mesh Instancing Demo
========================

.. image:: ../assets/thumbnails/f16_instancing.svg
   :alt: F16 Mesh Instancing
   :width: 480px

Demonstrates duplicating a base mesh using a list of 4x4 row-major transforms.
This is a CPU fallback that is portable to environments without GPU instancing.

Run the example::

    python examples/f16_instancing_demo.py

Key API:

- ``forge3d.geometry.instance_mesh(mesh, transforms)``

GPU Instancing
---------------

If native GPU instancing is available, you can detect it via::

    from forge3d.geometry import gpu_instancing_available
    if gpu_instancing_available():
        print("GPU instancing available (native path)")
    else:
        print("Using CPU instancing fallback")

Snippet:

.. code-block:: python

    import numpy as np
    from forge3d.geometry import primitive_mesh, instance_mesh

    base = primitive_mesh("box")
    T = np.eye(4, dtype=np.float32)
    T[0,3] = 1.0  # translate +X
    transforms = np.stack([
        np.eye(4, dtype=np.float32).reshape(-1),
        T.reshape(-1),
    ], axis=0)
    inst = instance_mesh(base, transforms)
