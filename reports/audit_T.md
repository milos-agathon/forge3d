## Audit Report — Workstream T (Vector IO & Geometry)

### Scope
- Workstream selector: ID "T" / Title "Vector IO & Geometry"
- Matched tasks: 5 (T1–T5)
- CSV headers validated (exact match to schema; trailing "Unnamed: 12" present and ignorable)

### CSV Hygiene Summary
- Priority values: all within {High, Medium, Low}
- Phase values: all within {MVP, Beyond MVP}
- Required cells (Task ID/Title/Deliverables/Acceptance Criteria): present for all T-rows
- No anomalies detected for selected rows

### Readiness Verdicts
- T1 GeoDataFrame ingestion (points/lines/polygons): Absent
- T2 Fast vector I/O via pyogrio + bbox filters: Absent
- T3 CRS transforms for vectors (pyproj): Absent
- T4 Polygon triangulation with holes (earcut): Absent
- T5 Large vector batching & instancing: Present but Partial

### Evidence Map
- Repository scan (ripgrep) findings relevant to T-series:
  - No matches for geopandas/pyogrio/shapely adapters or files:
    - Missing: python/forge3d/ingest/geopandas_adapter.py
    - Missing: python/forge3d/io/vector_io.py
    - Missing: python/forge3d/geo/crs.py
    - Missing: geometry/triangulate.rs
  - Instancing present (supports T5 in part):
    - examples/bundles_demo.py: instanced bundle creation (create_instanced_scene, create_simple_instanced_bundle)
    - docs/_build/html/_sources/render_bundles.md.txt: multiple references to instanced rendering and instance buffers
    - WGSL docs include instanced layouts (point_instanced.wgsl)

### Blocking Gaps
- T1: No GeoPandas ingest adapter or geometry-to-primitive routing in Python layer; no CRS handling.
- T2: No pyogrio fast path or bbox/mask filters; no vector I/O utility module; no benchmarks.
- T3: No dedicated CRS transform utilities for vectors; no transformer caching; no tests.
- T4: No triangulation (earcut) in Rust crate; no Python wrapper or tests for polygons with holes.
- T5: Instancing exists at rendering layer, but missing vector-specific batching utilities, chunking for memory bounds, and 1M points performance validation.

### Minimal Change Plan (No edits performed — audit-only)
- T1 (Absent → Present & Wired)
  - Add python/forge3d/ingest/geopandas_adapter.py with:
    - GeoDataFrame validation, CRS extraction (via GeoPandas/Shapely), mapping to primitives (points/lines/polygons) with style schema (color, width, height)
    - Lazy imports and clear error on missing extras
  - Add tests under tests/ for points/lines/polygons placement and style application; CRS respected
  - Example script examples/geopandas_ingest_demo.py
  - Docs page docs/ingest/geopandas.md
- T2 (Absent → Present & Wired)
  - Add python/forge3d/io/vector_io.py implementing pyogrio read_dataframe with bbox/mask filters; fallback to Fiona
  - Bench harness under bench/vector_io/ comparing Fiona vs pyogrio
  - Example examples/vector_io_bbox_demo.py; Docs docs/ingest/vector_io.md
- T3 (Absent → Present & Wired)
  - Add python/forge3d/geo/crs.py encapsulating pyproj.Transformer.from_crs with caching and EPSG:4326↔3857/UTM tests
  - Unit tests for round-trip within <1e-6 deg or <0.5m; simple perf sampling
- T4 (Absent → Present & Wired)
  - Add src/geometry/triangulate.rs using earcut; expose via PyO3 wrapper
  - Tests for multipolygons and holes; golden-image checks; Docs docs/geometry/triangulation.md
- T5 (Partial → Present & Wired)
  - Extend Python utilities to batch large vector datasets into instance buffers with chunking to keep memory bounded (<1 GB)
  - Add tests on synthetic 1M points for frame time (<33 ms headless) and memory
  - Wire examples to use batching path; document usage in docs/render_bundles.md

### Validation Runbook
- Build
  - maturin develop --release
  - cargo build --release
- Tests (vector tasks)
  - pytest -q tests/test_geopandas_ingest.py tests/test_vector_io.py tests/test_vector_crs.py tests/test_triangulate.py
- Demos (headless)
  - python examples/geopandas_ingest_demo.py --out reports/t1_geopandas.png
  - python examples/vector_io_bbox_demo.py --out reports/t2_vectorio.png
- Docs
  - cd docs && make html

Note: This report is audit-only. The Minimal Change Plan lists concrete file-level actions to reach “Present & Wired.”

