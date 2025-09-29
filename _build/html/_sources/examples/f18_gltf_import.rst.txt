F18 glTF Import Demo
====================

.. image:: ../assets/thumbnails/f18_gltf.svg
   :alt: F18 glTF Import
   :width: 480px

Demonstrates importing the first mesh primitive from a glTF 2.0 file (``.gltf`` or ``.glb``).

Run the example::

    python examples/f18_gltf_import_demo.py path/to/model.gltf

Key API:

- ``forge3d.io.import_gltf(path: str) -> MeshBuffers``

Snippet:

.. code-block:: python

    from forge3d.io import import_gltf
    mesh = import_gltf("DamagedHelmet.gltf")
    print(mesh.vertex_count, mesh.triangle_count)
