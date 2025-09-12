## Audit Report — Workstream U (Basemaps & Tiles)

### Scope
- Workstream selector: ID "U" / Title "Basemaps & Tiles"
- Matched tasks: 3 (U1–U3)
- CSV headers validated (match schema; trailing "Unnamed: 12" present and ignorable)

### CSV Hygiene Summary
- Priority values: within {High, Medium, Low}
- Phase values: within {MVP, Beyond MVP}
- Required fields (Task ID/Title/Deliverables/Acceptance Criteria): present for U rows
- No anomalies detected for selected rows

### Readiness Verdicts
- U1 XYZ/WMTS tile fetch + cache: Absent
- U2 Attribution overlay & extent management: Absent
- U3 Cartopy GeoAxes interop example: Absent

### Evidence Map
- Expected deliverables per roadmap.csv:
  - U1: python/forge3d/tiles/client.py; cache dir; WMTS/XYZ support; attribution API; tests (mocked network)
  - U2: Overlay renderer; position presets; docs on provider requirements; tests for visibility
  - U3: examples/cartopy_overlay.py; docs/integration/cartopy.md
- Repository scan results:
  - No `python/forge3d/tiles/` package present (tile client, cache, WMTS/XYZ not found)
  - No overlay/attribution renderer or provider-checklist docs located
  - No `examples/cartopy_overlay.py` or `docs/integration/cartopy.md`

### Blocking Gaps
- Missing tile client implementation and on-disk/offline cache layer
- No attribution overlay utilities (text/logo placement, presets, DPI/extent mapping)
- No Cartopy interop example or integration docs

### Minimal Change Plan (No edits performed — audit-only)
- U1 (Absent → Present & Wired)
  - Add `python/forge3d/tiles/client.py`:
    - Lazy optional deps (`requests`/`httpx`); WMTS/XYZ URL templating; tile math (z/x/y)
    - Cache directory layout (e.g., `%LOCALAPPDATA%/forge3d/tiles/{provider}/{z}/{x}/{y}.png`)
    - Retry/backoff; offline cache hits; attribution metadata surface
  - Add tests using mocked network (responses/pytest-httpx) validating correct URLs, cache hits
  - Example `examples/xyz_tile_compose_demo.py` composing viewport mosaic; Docs `docs/tiles/xyz_wmts.md`
- U2 (Absent → Present & Wired)
  - Add `python/forge3d/tiles/overlay.py` for attribution overlay (text/logo), position presets, DPI/extent mapping helpers
  - Tests verifying overlay visibility and positioning across resolutions; Docs on provider requirements
- U3 (Absent → Present & Wired)
  - Add `examples/cartopy_overlay.py` integrating a forge3d snapshot/overlay with Cartopy GeoAxes
  - Docs `docs/integration/cartopy.md`; ensure Agg backend in CI

### Validation Runbook
- Build
  - maturin develop --release
  - cargo build --release
- Tests (tiles & overlay)
  - pytest -q tests/test_tiles_client.py tests/test_tiles_overlay.py
- Demos (headless)
  - python examples/xyz_tile_compose_demo.py --out reports/u1_tiles.png
  - python examples/cartopy_overlay.py --out reports/u3_cartopy.png
- Docs
  - cd docs && make html

Note: This report is audit-only. The Minimal Change Plan lists concrete file-level actions to reach “Present & Wired.”

