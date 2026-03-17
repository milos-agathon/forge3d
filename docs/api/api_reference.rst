API Reference
=============

This page documents the current public Python surface. The recommended path for
most users is still the viewer-first workflow: datasets, the interactive
viewer, notebook widgets, bundles, map plates, buildings, and export.

Stable Workflow Surface
-----------------------

The most curated modules today are:

* ``forge3d.viewer``
* ``forge3d.viewer_ipc``
* ``forge3d.datasets``
* ``forge3d.widgets``
* ``forge3d.map_plate``
* ``forge3d.legend``
* ``forge3d.scale_bar``
* ``forge3d.north_arrow``
* ``forge3d.bundle``
* ``forge3d.buildings``
* ``forge3d.export``

Viewer And IPC
--------------

.. automodule:: forge3d.viewer
   :members: ViewerError, ViewerHandle, open_viewer_async, open_viewer, set_msaa
   :undoc-members:

.. automodule:: forge3d.viewer_ipc
   :members: send_ipc, launch_viewer, close_viewer, add_label, add_line_label, add_callout, add_vector_overlay, save_bundle, load_bundle
   :undoc-members:

Datasets
--------

.. automodule:: forge3d.datasets
   :members: available, bundled, remote, dataset_info, list_datasets, mini_dem, mini_dem_path, sample_boundaries, sample_boundaries_path, fetch, fetch_dem, fetch_cityjson, fetch_copc
   :undoc-members:

Notebook Widgets
----------------

.. automodule:: forge3d.widgets
   :members: widgets_available, ViewerWidget
   :undoc-members:

Cartography And Packaging
-------------------------

.. automodule:: forge3d.map_plate
   :members: BBox, MapPlateConfig, PlateRegion, MapPlate
   :undoc-members:

.. automodule:: forge3d.legend
   :members: LegendConfig, Legend
   :undoc-members:

.. automodule:: forge3d.scale_bar
   :members: ScaleBarConfig, ScaleBar
   :undoc-members:

.. automodule:: forge3d.north_arrow
   :members: NorthArrowConfig, NorthArrow
   :undoc-members:

.. automodule:: forge3d.bundle
   :members: BUNDLE_VERSION, TerrainMeta, CameraBookmark, BundleManifest, LoadedBundle, save_bundle, load_bundle, is_bundle
   :undoc-members:

Buildings And Vector Export
---------------------------

.. automodule:: forge3d.buildings
   :members: BuildingMaterial, Building, BuildingLayer, infer_roof_type, material_from_tags, material_from_name, add_buildings, add_buildings_cityjson, add_buildings_3dtiles
   :undoc-members:

.. automodule:: forge3d.export
   :members: VectorStyle, LabelStyle, Polygon, Polyline, Label, Bounds, VectorScene, generate_svg, export_svg, export_pdf, validate_svg
   :undoc-members:

Additional Current Modules
--------------------------

These modules are available today, but they are less curated than the viewer
workflow above and some are thin wrappers over native functionality:

* ``forge3d.cog`` for COG streaming
* ``forge3d.geometry`` for extrusion, primitives, mesh transforms, and validation
* ``forge3d.io`` for DEM and mesh IO helpers
* ``forge3d.mesh`` for TBN generation and BVH helpers
* ``forge3d.materials`` and ``forge3d.textures`` for PBR material containers
* ``forge3d.pointcloud`` for LAZ/COPC/EPT loading helpers
* ``forge3d.path_tracing`` for the deterministic CPU path-tracing fallback
* ``forge3d.sdf`` for SDF scene construction and hybrid rendering helpers
* ``forge3d.style`` and ``forge3d.vector`` for style/vector processing
* ``forge3d.lighting`` and ``forge3d.animation`` for lower-level utilities
* ``forge3d.terrain_pbr_pom`` for the terrain PBR/POM rendering workflow

Related Pages
-------------

* :doc:`../quickstart`
* :doc:`../tutorials/index`
* :doc:`../gallery/index`
