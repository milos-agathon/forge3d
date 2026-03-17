Terrain Rendering
=================

The current terrain workflows in forge3d are:

1. The **interactive viewer** via ``open_viewer_async()``
2. The **native scene API** via ``Scene`` and ``render_rgba()``
3. Dataset helpers such as ``mini_dem()``, ``mini_dem_path()``, and
   ``fetch_dem(...)``

Interactive Viewer
------------------

.. code-block:: python

   import forge3d as f3d

   with f3d.open_viewer_async(
       terrain_path=f3d.fetch_dem("rainier"),
       width=1400,
       height=900,
   ) as viewer:
       viewer.set_orbit_camera(phi_deg=28, theta_deg=48, radius=5500)
       viewer.set_sun(azimuth_deg=300, elevation_deg=24)
       viewer.snapshot("rainier.png")

Native Scene
------------

.. code-block:: python

   import forge3d as f3d

   height = f3d.mini_dem()
   scene = f3d.Scene(800, 600)
   scene.set_height_from_r32f(height)
   scene.set_camera_look_at(
       eye=(2.0, 2.0, 2.0),
       target=(0.0, 0.0, 0.0),
       up=(0.0, 1.0, 0.0),
       fovy_deg=45.0,
       znear=0.1,
       zfar=100.0,
   )
   rgba = scene.render_rgba()
   f3d.numpy_to_png("mini-dem-native.png", rgba)

Data Helpers
------------

.. code-block:: python

   import forge3d as f3d

   print(f3d.dem_stats(f3d.mini_dem()))
   print(f3d.dataset_info()["rainier"])

Status
------

Older examples that used ``set_height_data()``, ``render_terrain_rgba()``, or
global palette setters are outdated. Current terrain control is handled through
the viewer workflow, scene methods, dataset helpers, and terrain-parameter
objects elsewhere in the package.
