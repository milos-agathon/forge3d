Vector Graphics
===============

forge3d currently has **two** distinct vector workflows:

* ``forge3d.vector`` for low-level native point/line/polygon state helpers
* ``forge3d.export.VectorScene`` for 2D SVG/PDF export

For live viewer overlays, use ``forge3d.viewer_ipc.add_vector_overlay(...)``.

Low-Level Native Vector State
-----------------------------

.. code-block:: python

   import numpy as np
   from forge3d import vector

   point_ids = vector.add_points(
       np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64),
       fill_color=(1.0, 0.2, 0.2, 1.0),
       point_size=6.0,
   )
   line_ids = vector.add_lines(
       np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]], dtype=np.float64),
       stroke_color=(0.2, 0.8, 1.0, 1.0),
       stroke_width=2.0,
   )

   print(point_ids, line_ids)
   print(vector.get_vector_counts())
   vector.clear_vectors()

Vector Export
-------------

.. code-block:: python

   import forge3d as f3d

   scene = f3d.VectorScene()
   scene.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
   scene.add_label("AOI", (0.5, 0.5))
   svg = f3d.generate_svg(scene)
   print(svg[:80])

Status
------

Older examples that used top-level ``add_points_py`` / ``add_lines_py`` /
``add_polygons_py`` directly or claimed that the fallback ``Renderer`` rendered
native vector state are outdated. Prefer ``forge3d.vector`` for the low-level
native helpers and ``forge3d.export`` for portable 2D output.
