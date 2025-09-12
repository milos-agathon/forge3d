# docs/ingest/overviews.md
# Overview level selection to reduce bytes for coarse zooms.
# Exists to document S6 selection heuristics and integration with reads.
# RELEVANT FILES:python/forge3d/adapters/rasterio_tiles.py,tests/test_overview_selection.py,examples/overview_selection_demo.py

`select_overview_level(dataset, target_resolution, band=1)` chooses the most suitable overview using `dataset.overviews(band)` and the affine resolution.

`windowed_read_with_overview(...)` integrates selection to read from overviews at lower zoom levels.

Acceptance is validated by tests that mock dataset properties to keep the dependency surface small.

