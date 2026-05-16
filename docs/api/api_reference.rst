API Reference
=============

This reference covers the current public Python surface exposed by the package.
It is grouped by workflow so the module layout is easier to navigate.

Top-Level Package
-----------------

.. automodule:: forge3d
   :members:
   :no-index:

Viewer, Notebook, And IPC
-------------------------

.. automodule:: forge3d.viewer
   :members:
   :no-index:

Label API Truth
~~~~~~~~~~~~~~~

The public label workflow is exposed through ``forge3d.viewer.ViewerHandle``.
For feature ``002-label-api-truth``, use ``ViewerHandle.add_label``,
``ViewerHandle.add_labels``, ``ViewerHandle.add_line_label``,
``ViewerHandle.add_curved_label``, ``ViewerHandle.add_callout``,
``ViewerHandle.add_vector_overlay``, ``ViewerHandle.load_label_atlas``,
``ViewerHandle.set_labels_enabled``, ``ViewerHandle.clear_labels``,
``ViewerHandle.remove_label``, ``ViewerHandle.set_label_typography``, and
``ViewerHandle.set_declutter_algorithm``.

Successful point, batch, line, callout, and vector-overlay create operations
return stable ids where later inspection, removal, export, or review needs
them. Curved labels, typography controls, decluttering controls, and
terrain-elevated line labels are currently ``experimental`` public paths that
return typed ``experimental_feature`` diagnostics rather than unqualified
success. Empty text and invalid line geometry return ``placeholder_fallback``
diagnostics; known glyph coverage gaps return ``missing_glyphs`` diagnostics.

Deterministic LabelPlan
~~~~~~~~~~~~~~~~~~~~~~~

``forge3d.label_plan`` exposes the offline deterministic label compiler. Use
``LabelPlan.compile`` with labels, camera/output context, terrain samples,
keepouts, priority classes, typography, glyph coverage, and a seed to produce
accepted labels, rejected labels, diagnostics, bounds, and render/export
payload data. Unsupported payload backends return typed
``placeholder_fallback`` diagnostics rather than empty success.

.. automodule:: forge3d.label_plan
   :members:
   :no-index:

.. automodule:: forge3d.viewer_ipc
   :members:
   :no-index:

.. automodule:: forge3d.widgets
   :members:
   :no-index:

.. automodule:: forge3d.interactive
   :members:
   :no-index:

Terrain Configuration And Automation
------------------------------------

.. automodule:: forge3d.terrain_params
   :members:
   :no-index:

.. automodule:: forge3d.presets
   :members:
   :no-index:

.. automodule:: forge3d.animation
   :members:
   :no-index:

.. automodule:: forge3d.camera_rigs
   :members:
   :no-index:

.. automodule:: forge3d.terrain_scatter
   :members:
   :no-index:

Data Access And Scene Inputs
----------------------------

.. automodule:: forge3d.datasets
   :members:
   :no-index:

.. automodule:: forge3d.crs
   :members:
   :no-index:

.. automodule:: forge3d.cog
   :members:
   :no-index:

.. automodule:: forge3d.pointcloud
   :members:
   :no-index:

.. automodule:: forge3d.tiles3d
   :members:
   :no-index:

Scene Assets And Packaging
--------------------------

.. automodule:: forge3d.buildings
   :members:
   :no-index:

.. automodule:: forge3d.bundle
   :members:
   :no-index:

.. automodule:: forge3d.style
   :members:
   :no-index:

.. automodule:: forge3d.style_expressions
   :members:
   :no-index:

Cartography And Export
----------------------

.. automodule:: forge3d.map_plate
   :members:
   :no-index:

.. automodule:: forge3d.legend
   :members:
   :no-index:

.. automodule:: forge3d.scale_bar
   :members:
   :no-index:

.. automodule:: forge3d.north_arrow
   :members:
   :no-index:

.. automodule:: forge3d.export
   :members:
   :no-index:

Native Rendering And Quality
----------------------------

.. automodule:: forge3d.offline
   :members:
   :no-index:

.. automodule:: forge3d.denoise_oidn
   :members:
   :no-index:

.. automodule:: forge3d.path_tracing
   :members:
   :no-index:

.. automodule:: forge3d.sdf
   :members:
   :no-index:

.. automodule:: forge3d.lighting
   :members:
   :no-index:

Geometry, Mesh, Vector, And IO
------------------------------

.. automodule:: forge3d.geometry
   :members:
   :no-index:

.. automodule:: forge3d.io
   :members:
   :no-index:

.. automodule:: forge3d.mesh
   :members:
   :no-index:

.. automodule:: forge3d.vector
   :members:
   :no-index:

Materials, Textures, And Utilities
----------------------------------

.. automodule:: forge3d.materials
   :members:
   :no-index:

.. automodule:: forge3d.textures
   :members:
   :no-index:

.. automodule:: forge3d.colors
   :members:
   :no-index:

.. automodule:: forge3d.mem
   :members:
   :no-index:

.. automodule:: forge3d.config
   :members:
   :no-index:

CLI-Oriented Helpers
--------------------

.. automodule:: forge3d.terrain_demo
   :members:
   :no-index:

.. automodule:: forge3d.terrain_pbr_pom
   :members:
   :no-index:
