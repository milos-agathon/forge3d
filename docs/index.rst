forge3d Developer Platform
==========================

forge3d is a Rust-powered terrain and scene viewer with a Python surface built
around one primary workflow:

1. Resolve a dataset.
2. Launch the interactive viewer or notebook widget.
3. Send small, explicit scene updates over IPC.
4. Capture snapshots, package scenes, or compose publication output.

The Phase 2 platform work adds the missing developer layer around that runtime:
sample datasets, notebook widgets, a clearer docs spine, and higher-level
workflows for overlays, point clouds, map plates, buildings, and bundles.

.. toctree::
   :maxdepth: 2
   :caption: Start Here

   quickstart
   architecture
   api/api_reference

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   gallery/index

.. toctree::
   :maxdepth: 2
   :caption: Viewer And Scene Ops

   viewer/index

.. toctree::
   :maxdepth: 1
   :caption: Terrain Workflows

   terrain-tv1-atmosphere
   terrain-tv2-aovs
   terrain-tv3-scatter

.. toctree::
   :maxdepth: 1
   :caption: Product Notes

   pro-boundary-notes
   launch-blog

Platform Overview
-----------------

The current public workflow is intentionally small:

* ``forge3d.open_viewer_async()`` launches the Rust ``interactive_viewer`` binary.
* ``forge3d.ViewerHandle`` exposes the live control surface for terrain, overlays,
  point clouds, camera, lighting, and snapshots.
* ``forge3d.ViewerWidget`` wraps that same viewer flow for notebooks and falls
  back to an inline preview inside the widget when a full viewer process is
  unavailable.
* ``forge3d.datasets`` provides bundled samples and on-demand fetch helpers.
* ``forge3d.MapPlate``, ``forge3d.bundle``, and ``forge3d.export`` cover
  Pro-gated production packaging workflows around the live scene.

If you are new to the project, start with :doc:`quickstart`, then walk one of
the two tracks in :doc:`tutorials/index`.
