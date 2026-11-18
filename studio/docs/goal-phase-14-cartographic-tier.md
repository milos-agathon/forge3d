# Goal: bring Forge3D's cartographic tier into Studio

## Why this exists

Studio renders a terrain and can map it. Forge3D can also draw **vectors on
that terrain** — the rail network over Luxembourg, the rivers, the boundaries —
and Studio cannot, because the offscreen terrain path has no vector geometry at
all. `VectorOverlaySettings` carries only depth-test, halo and contour flags;
the geometry never reaches the renderer. Measured on the Luxembourg DEM
(640×480): a `VectorOverlayConfig` with `drape=True` changes **0** samples.

So this phase is one engine increment, then the shell work that exposes it.

## Non-negotiable rules

1. **One implementation.** The viewer already draws vector overlays
   (`src/viewer/terrain/vector_overlay.rs`, 609 lines, plus `core.rs` and
   `pipelines.rs`) — that is the reference, not a design to re-invent. If the
   offscreen path can reuse it, reuse it; if it cannot, say so in the commit
   and measure the difference.
2. **No control ships without a measured pixel change.** Render once with and
   once without, quote the changed-sample count. Zero means delete the
   control, not document it. The CI gate
   (`tests/test_studio_surface_audit.py`) enforces the rest.
3. **Measure under the user's conditions** — their DEM, camera, sun and frame
   size, not a convenient one.
4. **Every number in prose must be re-derivable.** Quote the command.
5. **Every UI range sits inside the engine's range**, and every allocation goes
   through `src/core/resource_tracker.rs` (the allocation gate enforces it).
6. **One increment per commit**, with tests, and the tree green: `cargo test`,
   clippy 0, `cargo fmt --check`, `tsc`, `npm test`, `pytest`.

## The increments, in order

**14.1 — the engine seam.** A vector overlay pass in the offscreen terrain
renderer: geometry and index buffers, a pipeline, drape from the height
texture, depth test against the terrain's depth, and the viewer's lighting.
Prefer a **separate pipeline and shader** (the viewer's way): new WGSL inside
`terrain_pbr_pom.wgsl` would move `PINNED_TERRAIN_SOURCE_HASH` and need the
determinism rewrite rerun and owner approval. *Measure:* a synthetic overlay
changes N samples — never 0.

**14.2 — the data.** The rail data is ready: `assets/gpkg/luxembourg_rail.gpkg`,
2,031 line features, EPSG:4326 — the same CRS as
`assets/tif/luxembourg_dem.tif`. The venv has **no** geopandas, fiona or
pyproj (verified), so the reader is either dependency-free (`sqlite3` plus
GeoPackage WKB — the GPKG *is* a SQLite file) or the dependency is declared
and paid for. Reproject only when the CRSs differ; the Luxembourg pair does
not. Wide lines are triangle quads — wgpu has no wide lines (the gallery
example `examples/luxembourg_rail_overlay.py` does this). *Measure:* the
features load, and the drawn lines land on the DEM — a pixel measurement, not
a claim.

**14.3 — the shell.** A vector layer on the terrain (asset, colour, width,
drape), the payload key, the Inspector control, and the service that turns
the file into the engine's batch shape
(`python/forge3d/map_scene.py::_native_vector_payload_for_layer` is the
existing producer of that payload — reuse it rather than inventing a second
one). *Measure:* the control changes N samples; the control that draws nothing
is deleted by the gate.

**14.4 — the plate.** Vector layers in the map plate's cut-out and in its
legend, with the scale bar that already exists.

**14.5 — labels and furniture**, the rest of the tier, only after the lines
are right.

## The audit, before the phase closes

* Every `RenderSettings`/scene field reaches the payload, and the payload and
  the service agree in both directions (the gate does this).
* Every exposed parameter was measured on this path; the 0-sample registry in
  `docs/feature-coverage.md` is the list of things Studio must not offer.
* No second implementation of an engine mechanism.

## What not to do

* **Do not swap Studio's stills onto the interactive path.** Its CONCORDIA
  contract was stale and is now repaired (one line), so it renders — but it
  differs from the mesh path in **221,837 of 230,400** samples (mean 33.18).
  It is a capability to measure, not a migration.
* Do not re-implement the drape, the palette or the camera in the shell.
* Do not add a control that draws nothing.

## Definition of done

The rail network draws on the Luxembourg DEM in a Studio still, from a project
the user set up; the plate shows it with a legend; every new control has a
measured pixel delta; the audit gate is green; the tree is green; and
`docs/feature-coverage.md` records what the tier still lacks.
