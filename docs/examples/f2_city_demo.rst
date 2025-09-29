F2 OSM City Demo
================

.. image:: ../assets/thumbnails/f2_city_demo.svg
   :alt: F2 OSM City Demo
   :width: 480px

This example demonstrates extruding a small set of OSM-like building footprints into
a single merged mesh and exporting to OBJ.

Run the example::

    python examples/f2_city_demo.py

The script writes ``city_demo_buildings.obj``.

Key APIs:

- ``forge3d.io.import_osm_buildings_extrude(features, default_height=10.0, height_key=None)``
- ``forge3d.io.import_osm_buildings_from_geojson(geojson, default_height=10.0, height_key=None)``

Snippet:

.. code-block:: python

    import numpy as np
    from forge3d.io import import_osm_buildings_extrude

    rect1 = np.array([[0,0],[2,0],[2,1],[0,1]], dtype=np.float32)
    rect2 = np.array([[3,0],[4,0],[4,2],[3,2]], dtype=np.float32)
    features = [{"coords": rect1, "height": 10.0}, {"coords": rect2, "height": 15.0}]
    mesh = import_osm_buildings_extrude(features, default_height=8.0)
