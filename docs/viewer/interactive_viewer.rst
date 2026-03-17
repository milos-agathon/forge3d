Interactive Viewer
==================

This file exists as a compatibility stub for older links. The current
viewer-facing workflow is:

- ``forge3d.open_viewer_async()`` for the non-blocking subprocess + IPC path
- ``forge3d.open_viewer()`` for the blocking desktop viewer path
- ``forge3d.ViewerHandle`` for runtime control
- ``forge3d.ViewerWidget`` for notebook integration with inline fallback

For current examples and supported commands, use the Markdown guide in the same
directory (`interactive_viewer.md`).
