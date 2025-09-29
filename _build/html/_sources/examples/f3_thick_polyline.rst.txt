F3 Thick Polyline Demo
======================

.. image:: ../assets/thumbnails/f3_thick_polyline.svg
   :alt: F3 Thick Polyline
   :width: 480px

This example demonstrates generating thick 3D polylines (ribbon-like meshes) with
miter, bevel, and round join styles. It also shows how to convert a constant pixel width
on screen to a world-space width given a camera distance and FOV.

Run the example::

    python examples/f3_thick_polyline_demo.py

The script writes three OBJ files:

- ``polyline_bevel.obj``
- ``polyline_miter.obj``
- ``polyline_round.obj``

Key API:

- ``forge3d.geometry.generate_thick_polyline(path, width_world, *, depth_offset=0.0, join_style="miter", miter_limit=4.0)``

Snippet:

.. code-block:: python

    import math, numpy as np
    from forge3d.geometry import generate_thick_polyline

    def pixel_to_world_width(pixel_width: float, z: float, fov_y_deg: float, image_height_px: int) -> float:
        f = math.tan(math.radians(fov_y_deg) * 0.5)
        return pixel_width * (2.0 * z * f / image_height_px)

    path = np.array([[0,0,0],[2,0,0],[2,1,0]], dtype=np.float32)
    width = pixel_to_world_width(3.0, z=4.0, fov_y_deg=45.0, image_height_px=1080)
    mesh = generate_thick_polyline(path, width_world=width, depth_offset=0.002, join_style="round")
