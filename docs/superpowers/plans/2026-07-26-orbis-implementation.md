# ORBIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development` for each task, `superpowers:verification-before-completion` before every green claim, and the Codex CLI autoreview requested by the user before each commit. Execute this plan inline; do not delegate it.

**Goal:** Add an opt-in `enable-globe` clipmap mode that can descend from ISS altitude to Mount Rainier over a COG height pyramid with camera-relative planetary precision, readiness-safe LOD transitions, bounded non-blocking streaming, measurable render evidence, and an honest Python `GlobeScene` API.

**Architecture:** Reuse the existing `Anchor`, clipmap rings, terrain shader, GPU LOD selector, COG reader, async tile loader, height mosaic, page table, and resource tracker. `GlobeFrame` owns the geodetic/ECEF transform and routes all narrowing through `Anchor`; `ClipmapLevel` remains the one mesh generator and supports flat or globe frames. `GlobeScene` is a thin native orchestration layer over those existing pieces, not a second renderer framework.

**Tech Stack:** Rust, PyO3, wgpu/WebGPU, WGSL, glam `DVec3`/`DMat4`, existing COG/GIS modules, Python/pytest, maturin, GitHub Actions.

---

## Global constraints

- Work only on branch/worktree `05-orbis`.
- Keep the default flat clipmap output and API behavior unchanged.
- Add no dependency: the required math, COG, async loading, readback, image, and tracking facilities already exist.
- Keep the single sanctioned f64-to-f32 world-coordinate boundary in `camera::Anchor`; extend its fail-closed inventory when `GlobeFrame` adopts it.
- Route persistent and transient GPU allocations through `tracked_create_*` helpers and an ORBIS allocation owner.
- A missing/unsupported GPU path returns diagnostic data or skips by the existing hosted-GPU convention; it never reports fabricated metrics or an image.
- Use the current CI feature matrix, including `cog_streaming`, `gis-remote`, `geos-topology`, and `shader-contract-asserts`. Do not restore the prompt's deleted CENSOR-era feature names.
- After every numbered task: run its focused tests, rebuild the Python extension after Rust/WGSL changes, run `cargo fmt --check`, run the relevant contract gates, write a review bundle outside the repository, run `codex review --uncommitted`, correct every Critical/Important finding, rerun the tests and review, then commit and push. The first push creates the draft PR named `05-orbis`; later pushes update it.

## Task 1: Planetary f64 frame

**Files**

- Create `src/terrain/clipmap/globe.rs`
- Modify `src/terrain/clipmap/mod.rs`
- Modify `Cargo.toml`
- Modify `pyproject.toml`
- Modify `tests/test_world_coord_f32_gate.py`
- Modify `tests/test_m06_single_rebase_contract.py`
- Add this executable plan to the branch

**Implementation**

- Add `enable-globe = ["cog_streaming"]`; include it in maturin features without changing the default feature set.
- Implement:

```rust
pub struct GlobeFrame {
    radius: f64,
    anchor: Anchor,
    ecef_to_enu: DMat4,
    mode: GlobeMode,
}

pub struct CameraRelative {
    pub position: Vec3,
    pub up: Vec3,
}

impl GlobeFrame {
    pub const WGS84_MEAN_RADIUS_M: f64 = 6_371_000.0;
    pub fn globe(radius: f64, camera_anchor: DVec3) -> Result<Self, String>;
    pub fn flat(camera_anchor: DVec3) -> Self;
    pub fn lonlat_alt_to_ecef(&self, lon_deg: f64, lat_deg: f64, altitude_m: f64) -> DVec3;
    pub fn ecef_to_lonlat_alt(&self, ecef: DVec3) -> DVec3;
    pub fn camera_relative(&self, ecef: DVec3) -> CameraRelative;
}
```

- Validate radius and finite inputs. Compute longitude/latitude in radians, use a spherical mean-radius model, and round-trip longitude canonically.
- Build the tangent transform from the camera anchor. Subtract ECEF positions in f64, rotate the delta in f64, and route the final narrowing through `Anchor::to_render_vec3`; do not add another raw `as f32`.
- Add Rust tests for cardinal ECEF points, lon/lat/alt round trip, invalid inputs, flat compatibility, and planetary-scale subtraction error.
- Update the two fail-closed precision inventories with the new intentional `Anchor` owner and no new conversion sites.

**Gate**

```powershell
cargo test --features enable-globe terrain::clipmap::globe
python -m pytest tests/test_world_coord_f32_gate.py tests/test_m06_single_rebase_contract.py -q
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 1 for absolute-f32 leakage, transform errors, feature-gate drift, and contract regressions."
```

**Commit:** `feat(globe): add camera-relative planetary frame`

## Task 2: Rebase `ClipmapLevel`

**Files**

- Modify `src/terrain/clipmap/level.rs`
- Modify `src/terrain/clipmap/ring.rs`
- Modify `src/terrain/clipmap/vertex.rs` only if task 3 proves the current format insufficient
- Modify `src/terrain/renderer/geometry.rs`
- Add focused Rust tests in `src/terrain/clipmap/level.rs`

**Implementation**

- Store the level center as `DVec3` and the last camera anchor as `DVec3`.
- Preserve `ClipmapLevel::new(config, Vec2, extent)` as the flat constructor. Add an `enable-globe` constructor taking `GlobeFrame`, center lon/lat, and extent.
- Make `generate()` dispatch to the existing flat ring generator or the globe-aware ring path.
- Implement `recenter(camera_anchor: DVec3) -> bool`; regenerate only when anchor displacement exceeds half the finest cell.
- Keep flat renderer cache keys and meshes byte-equivalent. Add deterministic flat-vs-prechange fixture assertions plus a globe anchor recenter threshold test.

**Gate**

```powershell
cargo test --features enable-globe terrain::clipmap
python -m pytest tests/test_terrain_clipmap_streaming.py tests/test_geomorph_seams.py -q
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 2 for flat-path breakage, premature f32 narrowing, cache invalidation bugs, and unnecessary vertex changes."
```

**Commit:** `feat(globe): rebase clipmap levels around camera`

## Task 3: Curved rings and shader displacement

**Files**

- Modify `src/terrain/clipmap/ring.rs`
- Modify `src/terrain/clipmap/vertex.rs` if required
- Modify `src/shaders/terrain_pbr_pom.wgsl`
- Modify `src/terrain/renderer/geometry.rs`
- Add Rust geometry tests and shader contract assertions

**Implementation**

- Interpret globe grid offsets in the center tangent plane, convert them to angular offsets, and map every center/ring/skirt vertex to f64 ECEF before camera-relative conversion.
- Derive geodetic up from ECEF. Reuse existing vertex fields only if morphing remains independently representable; otherwise add one octahedral-normal `Float32x2` attribute and update the one clipmap vertex layout.
- In `vs_clipmap_main`, decode/derive up and displace the camera-relative base position by sampled height along geodetic up. Keep flat vertices at `z=0` with flat `+Z` up so the default shader output is unchanged.
- Test radius, tangent-distance error, finite normals, winding, maximum edge length, ring bounds, and flat vertex compatibility.

**Gate**

```powershell
cargo test --features enable-globe terrain::clipmap
cargo test --features enable-globe,shader-contract-asserts shader
maturin develop --release --features enable-globe
python -m pytest tests/test_terrain_clipmap_streaming.py tests/test_geomorph_seams.py -q
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 3 for incorrect curvature, normal/displacement errors, vertex-layout mismatches, long seam triangles, and flat rendering drift."
```

**Commit:** `feat(globe): curve clipmap rings onto planet`

## Task 4: Readiness-gated geomorph and skirts

**Files**

- Modify `src/terrain/clipmap/geomorph.rs`
- Modify `src/terrain/clipmap/streaming.rs`
- Modify `src/terrain/clipmap/ring.rs`
- Modify `src/terrain/renderer/streaming.rs`
- Extend `tests/test_geomorph_seams.py`

**Implementation**

- Add:

```rust
pub struct TileReadiness {
    pub fine_resident: bool,
    pub coarse_resident: bool,
}
```

- Keep the current distance morph calculation as the ready case. When either ring side is unavailable, return the coarse-snapped state; never expose a T-junction while tiles arrive.
- Derive per-ring readiness from the existing loaded/pending tile sets and required tile keys.
- Compute skirt depth as `max(configured_depth, curvature_sagitta + altitude_allowance)` with finite monotonic bounds. Do not create cross-row skirt triangles.
- Add throttled-arrival tests that enumerate readiness transitions and assert boundary parity/hole count remains zero.

**Gate**

```powershell
cargo test --features enable-globe terrain::clipmap::geomorph terrain::clipmap::streaming
python -m pytest tests/test_geomorph_seams.py tests/test_terrain_clipmap_streaming.py -q
maturin develop --release --features enable-globe
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 4 for inverted readiness semantics, transition cracks, non-monotonic skirts, and streamer state races."
```

**Commit:** `feat(globe): gate geomorphing on tile readiness`

## Task 5: Planetary GPU LOD

**Files**

- Modify `src/terrain/clipmap/gpu_lod.rs`
- Modify `src/shaders/clipmap_lod_select.wgsl`
- Add CPU/GPU parity tests in `src/terrain/clipmap/gpu_lod.rs`

**Implementation**

- Extend the existing uniform, without a new bind group, with planet radius, camera altitude, and camera-up.
- Supply camera-relative tile centers. Use their distance for screen-space error.
- Horizon-cull with the stable tangent-plane test equivalent to `dot(camera_up, tile_from_planet_center) >= radius / (radius + altitude)`, expanded conservatively by tile angular radius.
- Make CPU selection use the same equations and preserve the current visible-tile/LOD result shape.
- Replace the current fake GPU result with actual counter/output readback using existing async readback helpers; tests compare CPU/GPU selected tile IDs on a real adapter and skip only when the adapter is unavailable.

**Gate**

```powershell
cargo test --features enable-globe,async_readback terrain::clipmap::gpu_lod
maturin develop --release --features enable-globe
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 5 for horizon inequality errors, absolute coordinates in WGSL, uniform layout drift, fake GPU results, and CPU/GPU divergence."
```

**Commit:** `feat(globe): add altitude-aware GPU LOD`

## Task 6: Bounded planetary COG streaming

**Files**

- Modify `src/terrain/clipmap/streaming.rs`
- Modify `src/terrain/renderer/streaming.rs`
- Modify `src/terrain/page_table/height_loader.rs`
- Modify `src/terrain/page_table/gpu.rs`
- Modify `src/terrain/stream/height.rs`
- Modify existing COG metadata parsing only where needed for lon/lat tile mapping
- Add native and wasm compile-contract tests

**Implementation**

- Compose, rather than duplicate, `CogHeightReader`, `AsyncTileLoader`, `HeightMosaic`, and `PageTable`.
- Keep native loads on the existing bounded worker queue. Add a wasm poll/request-completion queue with the same bounded in-flight and dedup semantics; no blocking executor or wait is reachable on wasm.
- Cap ORBIS GPU-visible residency below 512 MiB. Before upload, evict least-recently-used leaf tiles until the tracked mosaic/page-table/staging total fits.
- Populate page-table ancestor entries and resolve the nearest resident ancestor for every missing selected tile. A request miss must return coarse coverage, never an empty mapping.
- Assign an `AllocationOwner` to mosaic, page table, upload/readback staging, and cache-visible GPU resources; expose their real tracked high-water mark.
- Test bounded queue behavior, LRU order, ancestor selection, tracked allocation release, native non-blocking polling, and `cargo check --target wasm32-unknown-unknown --features enable-globe`.

**Gate**

```powershell
cargo test --features enable-globe,cog_streaming terrain::clipmap::streaming terrain::renderer::streaming
cargo check --target wasm32-unknown-unknown --features enable-globe
maturin develop --release --features enable-globe
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 6 for frame-thread blocking, unbounded queues, wrong LRU eviction, holes instead of ancestors, untracked allocations, and wasm-only failures."
```

**Commit:** `feat(globe): stream bounded COG terrain`

## Task 7: `GlobeScene` Python/native API

**Files**

- Add `src/terrain/clipmap/globe_scene.rs`
- Modify the existing clipmap PyO3 registrar under `src/terrain/clipmap/py_bindings.rs`
- Modify `src/py_module/functions/rendering.rs`
- Modify `python/forge3d/__init__.py`
- Modify `python/forge3d/__init__.pyi`
- Modify `tests/test_api_contracts.py`
- Add `tests/test_globe_floating_origin.py`

**Implementation**

- Add `GlobeScene(cog_source, target_lon, target_lat, target_name)` with validated finite coordinates, non-empty target name, and diagnostic-bearing source errors.
- Expose `fly_to(lon, lat, altitude)`, `scripted_descent(waypoints=None)`, `snapshot()`, and `metrics()`.
- Use Mount Rainier (`assets/tif/dem_rainier.tif`, `-121.7603`, `46.8523`) as the deterministic named peak. The default descent spans 408,000 m to ground with logarithmic altitude sampling and polls streaming without blocking a frame.
- Return a typed PyO3 metrics class and Python dict-compatible properties for:
  - `max_vertex_jitter_px`
  - `peak_gpu_visible_bytes`
  - `lod_crack_pixels`
- Register the class, package export, `__all__`, stub, and positive API contracts together.

**Gate**

```powershell
maturin develop --release --features enable-globe
python -m pytest tests/test_globe_floating_origin.py tests/test_api_contracts.py -q --tb=short
cargo test --features enable-globe terrain::clipmap::globe_scene
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 7 for PyO3 registration drift, stale extension risk, fake rendering/metrics, invalid waypoint handling, and diagnostic contract gaps."
```

**Commit:** `feat(globe): expose scripted GlobeScene`

## Task 8: Honest metrics, golden, and CI gate

**Files**

- Complete `tests/test_globe_floating_origin.py`
- Add the final Mount Rainier golden under `tests/golden/terrain/`
- Modify `.github/workflows/ci.yml`
- Modify golden allowlists/contracts where required
- Add a concise ORBIS evidence document under `docs/`

**Implementation**

- Jitter: project the same tracked ground vertices in two physical frames separated by a camera micro-step, measure the maximum screen-space delta after subtracting the analytically expected camera motion, and compare with the deliberately naive absolute-f32 control.
- Memory: bracket the full descent in an ORBIS allocation-ledger capture and report the actual GPU-visible high-water mark.
- Cracks: read depth plus coverage, restrict evaluation to generated ring-boundary segments, and count uncovered interior pixels between valid samples. Store no all-zero shortcut.
- Gate one physical scripted descent on jitter `< 0.5`, peak `< 512 * 1024 * 1024`, and cracks `== 0`; assert throttled ancestor fallback and non-blocking frame progress separately.
- Compare the final ground snapshot with the committed golden. Use the repository's existing opt-in/unsupported-GPU skip convention and require real non-software adapter metadata before accepting evidence.
- Verify the task 1 `enable-globe` CI feature wiring and add the focused Python gate. Keep the contract that `.github/workflows/ci.yml` is the feature-list source of truth.

**Gate**

```powershell
cargo test --workspace --features default,async_readback,copc_laz,cog_streaming,gis-remote,geos-topology,weighted-oit,wsI_bigbuf,wsI_double_buf,enable-pbr,enable-tbn,enable-normal-mapping,enable-hdr-offscreen,enable-renderer-config,enable-staging-rings,shader-contract-asserts,enable-globe -- --test-threads=1 --skip gpu_extrusion --skip brdf_tile
maturin develop --release --features enable-globe
python -m pytest tests/test_globe_floating_origin.py tests/test_api_contracts.py -v --tb=short
cargo fmt --check
cargo forge3d-clippy
codex review --uncommitted "Review ORBIS task 8 and the entire branch for measurable-gate circumvention, nonphysical metrics, golden-update loopholes, CI feature drift, hidden fallbacks, and incomplete prompt coverage."
```

**Commit:** `test(globe): gate ORBIS descent metrics`

## Final branch gate

Run the task 8 Rust/Python/lint commands again from a clean tree, plus:

```powershell
git diff --check origin/main...HEAD
git status --short
codex review --base origin/main "Final ORBIS review: verify every requirement in docs/prompts/fable5-moonshots/05-orbis.md is implemented with real evidence and no placeholder or silent fallback."
```

Correct findings in a final reviewed commit only if necessary, rerun every affected gate, push, mark the `05-orbis` PR ready, and report exact measured jitter, GPU-visible peak bytes, crack pixels, tests, commits, and PR URL.
