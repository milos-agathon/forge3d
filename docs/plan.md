# forge3d Roadmap: Effort/Impact + Codebase-Grounded Implementation Plan

## 0) Ground truth from the current forge3d tree (what already exists)

### 0.1 Transparency / OIT (already implemented, needs productization)

* **Transparency/OIT**: Weighted blended OIT exists for vectors (`src/vector/oit/*`, feature `weighted-oit`).
* There’s also a **dual-source OIT renderer with WBOIT fallback**:

  * [src/core/dual_source_oit.rs](cci:7://file:////forge3d/src/core/dual_source_oit.rs:0:0-0:0)
  * Exposed on [Scene](cci:2://file:////forge3d/src/scene/mod.rs:47:0-161:1)
  * Python binding in [src/scene/mod.rs](cci:7://file:////forge3d/src/scene/mod.rs:0:0-0:0)

**Evidence excerpt (preserved):**

```rust
/src/core/dual_source_oit.rs:1-18
//! B16: Dual-source blending Order Independent Transparency
//! High-quality OIT using dual-source color blending with WBOIT fallback

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Dual-source OIT rendering mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DualSourceOITMode {
    /// Disabled - use standard alpha blending
    Disabled,
    /// Dual-source blending (requires hardware support)
    DualSource,
    /// Weighted Blended OIT fallback
    WBOITFallback,
    /// Automatic mode - dual-source if supported, otherwise WBOIT
    Automatic,
}
```

**Reported remaining work (preserved):**

* Integration test, Python binding, docs update.
* Hardware detection may be conservative across Vulkan/Metal/D3D12 (see Risks).

---

### 0.2 Shadows (VSM/EVSM/MSM already have plumbing; needs blur/leak/validation/API coherence)

The engine already has **VSM/EVSM/MSM plumbing**:

* `ShadowTechnique` includes `VSM/EVSM/MSM`:

  * [src/lighting/shadow.rs](cci:7://file:////forge3d/src/lighting/shadow.rs:0:0-0:0)
* Moment generation compute pass exists:

  * [src/shadows/moment_pass.rs](cci:7://file:////forge3d/src/shadows/moment_pass.rs:0:0-0:0)
  * [src/shaders/moment_generation.wgsl](cci:7://file:////forge3d/src/shaders/moment_generation.wgsl:0:0-0:0)
* Filtering code lives in:

  * [src/shaders/shadows.wgsl](cci:7://file:////forge3d/src/shaders/shadows.wgsl:0:0-0:0)
* Runtime manager exists:

  * [src/shadows/manager/system.rs](cci:7://file:////forge3d/src/shadows/manager/system.rs:0:0-0:0)

**Evidence excerpts (preserved):**

```rust
/src/shadows/manager/system.rs:27-32
        let requires_moments = matches!(
            config.technique,
            ShadowTechnique::VSM | ShadowTechnique::EVSM | ShadowTechnique::MSM
        );
        config.csm.enable_evsm = requires_moments;
```

```rust
/src/shadows/moment_pass.rs:149-197
    pub fn execute(
        &self,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        technique: ShadowTechnique,
        cascade_count: u32,
        shadow_map_size: u32,
        evsm_positive_exp: f32,
        evsm_negative_exp: f32,
    ) {
```

**Reported remaining work (preserved):**

* Blur pass integration (separable Gaussian)
* Light bleeding reduction parameters
* Integration tests for each technique
* Align Python config validation with actual Rust support (stop rejecting VSM/EVSM/MSM incorrectly)

---

### 0.3 Terrain streaming infrastructure exists, but not geometry clipmaps

* Tile/mosaic/page-table scaffolding exists:

  * `src/terrain/stream/*`
  * [src/terrain/page_table.rs](cci:7://file:////forge3d/src/terrain/page_table.rs:0:0-0:0)
* **Not geometry clipmaps** (explicitly missing nested regular grid, seam handling, morphing).

---

### 0.4 Temporal building blocks exist, but no full-frame TAA module

* Temporal AO resolve exists:

  * [src/shaders/temporal/resolve_ao.wgsl](cci:7://file:////forge3d/src/shaders/temporal/resolve_ao.wgsl:0:0-0:0)
* There are various temporal SSR/SSGI codepaths, but **no full-frame TAA** module.

**Evidence excerpt (preserved):**

```wgsl
/src/shaders/temporal/resolve_ao.wgsl:1-12
// Temporal accumulation for AO (P5.1)
// Reprojection with neighborhood clamping to reduce ghosting

struct TemporalParams {
    temporal_alpha: f32,  // Blend factor [0=no history, 1=full history]
    _pad: vec3<f32>,
}

@group(0) @binding(0) var current_ao: texture_2d<f32>;
@group(0) @binding(1) var history_ao: texture_2d<f32>;
@group(0) @binding(2) var output_ao: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> params: TemporalParams;
```

**Missing for full TAA (preserved):**

* Jitter sequence (Halton/blue noise)
* Motion vector generation
* Main color buffer history
* Proper reprojection with velocity buffer
* YCoCg color space conversion for better clamping

---

## 1) Consolidated implementation state + corrected effort/impact

### 1.1 Legend and intent

* **Impl** scale from GPT report: 0=none, 1=scaffold, 2=partial, 3=integrated+tested.
* Claude report provides “actual state” % estimates and revised effort.

We preserve both views and present a unified table.

### 1.2 Unified feature table (merged)

| #  | Feature                                                 | Impl (GPT) |     “Actual State” (Claude) | Corrected Impact |                  Revised Effort | Notes / why correction matters                                                                                     |
| -- | ------------------------------------------------------- | ---------: | --------------------------: | ---------------: | ------------------------------: | ------------------------------------------------------------------------------------------------------------------ |
| 1  | **Weighted blended OIT for water/volumetrics/overlays** |          2 |               ~90% complete |                4 |             **1–2** (Claude: 1) | Core OIT already exists; main work is wiring into passes + validation + bindings/docs.                             |
| 2  | **Sun position (lat/lon/date/time → az/el)**            |          1 | ~20% (manual controls only) |            **4** |             **2–3** (Claude: 2) | Needs deterministic ephemeris + timezone semantics + UX + reference tests.                                         |
| 3  | **Map plate compositor (title/legend/scale/inset)**     |          1 |         ~5% (basic helpers) |                4 |             **3–4** (Claude: 3) | Overlay+text primitives exist but true layout/typography/legend are nontrivial.                                    |
| 4  | **TAA (jitter + reprojection + clamp)**                 |          1 |      ~30% (AO/SSR temporal) |                4 |                **3–4** (GPT: 4) | AO temporal ≠ full-frame TAA; needs history buffers + reprojection + rejection; motion vectors are blocker.        |
| 5  | **Shadow modes VSM/EVSM/MSM (+ blur/leak control)**     |          2 |               ~85% complete |                4 | **1–3** (Claude: 1–2; GPT: 2–3) | Plumbing exists; remaining is blur path + leak control + API coherence + forced-edge tests.                        |
| 6  | **Terrain LOD: geometry clipmaps / nested grids**       |          1 |   ~40% (screen-space error) |                5 |             **4–5** (Claude: 4) | Harder than 3–4: touches mesh strategy, seams, morphing, streaming policy, tests.                                  |
| 7  | **COG streaming (HTTP range + overviews + cache)**      |          0 | ~25% (local tile streaming) |                5 |                         **4–5** | GPT said 0% because no HTTP range infra; Claude notes local tile streaming exists. Plan phases local → HTTP range. |
| 8  | **PMTiles vector streaming + styling/declutter**        |          0 |                          0% |                4 |                         **4–5** | Needs tile index reading, decode, styling DSL, label declutter integration.                                        |
| 9  | **3D Tiles import/export**                              |          0 |                          0% |                4 |                           **5** | Large: tileset.json, payload formats, metadata, bounding volumes, LOD; b3dm/pnts parsing etc.                      |
| 10 | **Point clouds (EPT/COPC) + splats**                    |          0 |   ~10% (basic point render) |                5 |                          **5+** | Full subsystem: streaming, LOD, rendering, picking, memory budgets; optional Gaussian splat path.                  |

---

## 2) Accuracy analysis & high-signal deltas (merged)

### 2.1 Overestimated effort (already implemented)

* **OIT is not a “new feature”** in forge3d; it’s a *wiring + productization* task.
* **Shadow upgrade path is already present in code**; remaining work is quality/blur/leak control + forced-edge tests.

### 2.2 Underestimated effort (needs foundations)

* **TAA is usually harder than “3”** unless you accept a limited “static-geometry reprojection only” v1.
* **Terrain clipmaps** are harder due to mesh generation, seams, morphing, streaming policy and tests.
* **3D Tiles complexity** is high (b3dm Draco compression, batched textures, metadata propagation, etc.).

---

## 3) Missing “must-have” differentiators (merged union)

### 3.1 From GPT report (preserved)

* **Declarative styling + legend synthesis** (Impact 5 / Effort 3–4): Mapbox-ish rules layer driving:

  * vector symbology + label rules + legend output + map plates.
* **Web export / WASM WebGPU viewer** (Impact 5 / Effort 5): Rust-only “shareable viewer” target is a moat.
* **Measurement & analysis tools** (Impact 4 / Effort 2–3): distance/area/elevation profile driven by existing picking.

### 3.2 From Claude report (preserved)

* **Motion Vectors / Velocity Buffer** (Impact 5 / Effort 2): foundation for TAA, motion blur, temporal effects.
* **GPU-Driven Rendering / Indirect Draw** (Impact 4 / Effort 3):

  * Note: partial support in [src/vector/indirect.rs](cci:7://file:////forge3d/src/vector/indirect.rs:0:0-0:0) (21KB)
* **Async Compute Overlap** (Impact 3 / Effort 2): better GPU utilization, parallel shadow/main pass.
* **Deferred Decals** (Impact 3 / Effort 2): terrain annotations / road markings / labels without geometry.
* **Exposure / Auto-Exposure Histogram** (Impact 3 / Effort 2):

  * Note: tonemap exists in [src/core/tonemap.rs](cci:7://file:////forge3d/src/core/tonemap.rs:0:0-0:0)

---

## 4) Strict implementation plan (surgically scoped milestones / phases)

### Global rules (preserved from GPT report)

All milestones must follow forge3d rules:

* **defaults unchanged**
* **new features opt-in**
* **Python tests define behavior**

---

## Phase 0 / Quick Wins (unlock “already done” features)

### P0.1 / M1 — “Make existing transparency real” (OIT productization)

**Deliverables (merged):**

* Add a single “transparency strategy” selection (standard / WBOIT / dual-source-auto) in the main scene/viewer pipeline.
* Route *vector overlays* and *any multi-layer transparent draw calls* through the same compositor.
* Integration test, Python binding `enable_oit()` (as explicitly proposed), docs update.
* Add a forced-overlap transparency test scene (two crossing translucent layers) + Python pixel-diff assertion.

**Exit / Acceptance criteria (preserved + clarified):**

* OIT visibly changes output vs sorted alpha in the forced scene; tests prove it.

**Risks (preserved):**

* Dual-source blending hardware detection may need backend-specific checks for Vulkan/Metal/D3D12.

---

### P0.2 / M3 — Shadow filtering “done for real” (VSM/EVSM/MSM polish)

**Deliverables (merged):**

* Align Python config validation ([python/forge3d/config.py](cci:7://file:////forge3d/python/forge3d/config.py:0:0-0:0)) with actual Rust support (stop incorrectly rejecting VSM/EVSM/MSM).
* Implement/verify **moment blur path** + leak controls:

  * EVSM exponents
  * moment bias
  * memory budget checks
* Add CLI switch (explicitly proposed): `--shadow-technique`
* Forced-edge regression tests: same scene, only technique changes, assert penumbra/leak metrics.
* Produce shadow technique A/B comparison renders (explicit remaining checklist item).

**Exit / Acceptance criteria (merged):**

* Technique toggles produce non-trivial numeric diffs; no new warnings; memory within budget.
* Integration tests exist per technique (or one parameterized test covering all techniques).

---

### P0.3 / M2 — Sun ephemeris + time-of-day controls

**Deliverables (merged):**

* Deterministic ephemeris function (prefer pure math, no heavy deps).
* API in Python:

  * `sun_position(lat, lon, datetime) → (azimuth, elevation)`
* Viewer/preset config keys:

  * `sun_lat`, `sun_lon`, `sun_datetime` (explicit timezone/UTC semantics).
* Python tests vs known reference cases (NOAA-style reference numbers).
* “Sun ephemeris validation against NOAA calculator” included as a remaining checklist item.

**Exit / Acceptance criteria (preserved + clarified):**

* Given (lat, lon, datetime UTC), az/el matches reference within tolerance.
* Renders show measurable shadow direction change.

---

## Phase 1 / TAA Foundation (1–2 weeks)

### P1.1 — Motion Vectors / Velocity Buffer (dependency blocker for TAA)

**Deliverables (preserved):**

* GBuffer velocity channel, shader output.
* Test: `test_motion_vectors.py`

**Why it’s first (preserved):**

* Without proper motion vectors, fast camera motion will ghost. P1.1 is critical.

---

### P1.2 — Jitter Sequence

**Deliverables (preserved):**

* Halton 2,3 or blue noise jitter; projection matrix jitter.
* Test: `test_jitter_sequence.py`

---

### P1.3 / M4 — TAA Resolve (static-geometry reprojection v1 acceptable)

**Deliverables (merged):**

* Halton jitter + history color buffer + reprojection using depth and prev/current matrices.
* History clamp/rejection:

  * neighborhood clamp
  * reactive mask for overlays/water if needed
* Mentioned improvements (preserved):

  * YCoCg color space conversion for better clamping (optional but recommended)
* Tests:

  * `test_taa_convergence.py`
  * Variance reduction vs no-TAA baseline (numeric)

**Exit criteria (preserved):**

* Demonstrable shimmer reduction metric in a thin-feature scene.
* Remaining checklist item: “TAA convergence test (static scene, verify jitter-induced noise reduction)”.

---

### P1.4 — TAA Integration

**Deliverables (preserved):**

* `--taa` CLI flag, preset support
* `test_taa_toggle.py`

---

## Phase 2 / Terrain Scale (2–3 weeks)

### P2.1 / M5 — Clipmap Structure (true scalability)

**Deliverables (merged):**

* Replace single-grid terrain draw with nested-ring clipmap mesh (with skirts/morphing).
* Connect to existing height mosaic/page table to choose LOD + streaming requests.
* Tests:

  * `test_clipmap_structure.py`

---

### P2.2 — Geo-morphing / seam correctness

**Deliverables (preserved):**

* Vertex blending at LOD boundaries
* Test: `test_geomorph_seams.py`

**Risks (preserved):**

* T-junction artifacts at LOD boundaries require careful vertex blending; risk of visual seams.

---

### P2.3 — GPU LOD Selection

**Deliverables (preserved):**

* Compute shader frustum cull + LOD
* Test: `test_gpu_lod_selection.py`

**Exit criteria (merged):**

* No cracks across LOD boundaries; stable triangle count vs camera height.
* Remaining checklist item: “Clipmap triangle budget verification (≥40% reduction at distance)”.

---

## Phase 3 / Cloud-Native Data (3–4+ weeks)

### M6 — Data-at-scale direction (pick one first, to avoid thrash)

**Preserved decision gate (GPT):**

* Choose **COG vs PMTiles first** to prevent wasted integration work.

#### Option A: COG

**GPT direction (preserved):**

* Start with local-file COG tile decode + cache; then add HTTP range.

**Claude breakdown (preserved):**

* **P3.1** COG Range Reads — HTTP range request adapter

  * `test_cog_range_read.py`
* **P3.2** COG Overview Integration — IFD parsing, overview selection

  * `test_cog_overviews.py`
* **P3.3** COG Tile Cache — LRU cache with memory budget

  * `test_cog_cache_eviction.py`

**Risks (preserved):**

* Range reads add latency; may need speculative prefetch and aggressive caching.

---

#### Option B: PMTiles

**GPT direction (preserved):**

* Start with local PMTiles decode + style rules; then add HTTP range.

**Claude breakdown (preserved):**

* **P3.4** PMTiles Reader — header/directory parsing

  * `test_pmtiles_parse.py`
* **P3.5** PMTiles Tile Fetch — range read + decompress

  * `test_pmtiles_fetch.py`
* **P3.6** Vector Tile Decode — MVT → geometry

  * `test_mvt_decode.py`

---

**Exit criteria (preserved):**

* Streamed dataset renders without pre-tiling
* cache hit/miss stats exposed
* tests for determinism

---

## Phase 4 / Creator Workflow (1–2 weeks)

### P4.1–P4.3 — Map Plate Compositor (classic creator moat)

**Claude breakdown (preserved):**

* **P4.1** Map Plate Layout — `MapPlate` class with regions

  * `test_map_plate_layout.py`
* **P4.2** Legend/Scale Bar — auto-generated from colormap

  * `test_legend_generation.py`
* **P4.3** Export Pipeline — PNG/PDF with layout

  * `test_plate_export.py`

---

## Phase 5 / Platform Moat (4–6+ weeks)

### 3D Tiles

**Claude breakdown (preserved):**

* **P5.1** 3D Tiles Parser — tileset.json, b3dm, pnts

  * `test_3dtiles_parse.py`
* **P5.2** 3D Tiles Traversal — SSE-based tile selection

  * `test_3dtiles_sse.py`
* **P5.3** 3D Tiles Render — batched mesh/point rendering

  * `test_3dtiles_render.py`

**Risks (preserved):**

* b3dm Draco compression, batched textures, metadata propagation are significant.
* Consider phasing: basic b3dm first, then compressed formats.

---

### Point clouds (EPT/COPC) + Splats

**Claude breakdown (preserved):**

* **P5.4** EPT/COPC Parser — octree, LAZ decode

  * `test_copc_parse.py`
* **P5.5** Point Cloud LOD — octree traversal + budget

  * `test_pointcloud_lod.py`
* **P5.6** Splat Rendering — Gaussian splat path (optional)

  * `test_splat_render.py`

---

## 5) Recommended priority (merged)

### GPT priority (preserved)

1. M1 OIT
2. M2 Sun ephemeris
3. M3 Shadows
4. M4 TAA v1
5. M5 Clipmaps
6. M6 Data-at-scale (COG vs PMTiles)
7. Backlog: 3D Tiles / Point clouds / Web export

### Claude priority (preserved)

```
1. P0.1-P0.3 (Quick Wins)     → Unlock "already done" features (1 week)
2. P1.1-P1.4 (TAA)            → Biggest visual stability jump (1 week)
3. P2.1-P2.3 (Clipmaps)       → Unlocks large datasets (2-3 weeks)
4. P3.1-P3.3 (COG)            → Cloud-native differentiation (2 weeks)
5. P4.1-P4.3 (Map Plates)     → Creator workflow moat (1-2 weeks)
```

**Merged recommendation:** adopt Claude’s phase grouping but keep GPT’s explicit “M6 choose one direction first” to prevent thrash.

---

## 6) Remaining checklist (merged union)

* [ ] P0.1: OIT integration test proving water/volumetric rendering
* [ ] M1: forced-overlap transparency test scene + Python pixel-diff assertion
* [ ] P0.2/M3: Shadow technique A/B comparison renders
* [ ] P0.3/M2: Sun ephemeris validation against NOAA calculator
* [ ] P1.x/M4: TAA convergence test (static scene, verify jitter-induced noise reduction)
* [ ] P2.x/M5: Clipmap triangle budget verification (≥40% reduction at distance)
* [ ] P3.x/M6: COG/PMTiles fetch latency benchmarks

---

## 7) What remains / risks (merged union)

1. **OIT dual-source blending**

   * Hardware detection is conservative; may need backend-specific checks for Vulkan/Metal/D3D12.

2. **TAA ghosting**

   * Without proper motion vectors, fast camera motion will ghost. P1.1 is critical.

3. **Clipmap geo-morphing**

   * T-junction artifacts at LOD boundaries require careful vertex blending; risk of visual seams.

4. **COG HTTP overhead**

   * Range reads add latency. May need speculative prefetch and aggressive caching.

5. **3D Tiles complexity**

   * b3dm Draco compression, batched textures, metadata propagation are significant effort. Consider phasing: basic b3dm first, then compressed formats.

---

## 8) Status / next step (preserved)

* **Completed**: codebase-grounded reality check and corrected effort/impact.
* **Next decision needed from you**: choose the **M6 direction** (COG vs PMTiles first), because it drives the styling/label/LOD roadmap and prevents wasted integration work.

---

# Appendix A — “Implementation Level + corrected effort/impact” table (original GPT section preserved)

Legend: **Impl** 0=none, 1=scaffold, 2=partial, 3=integrated+tested.

| #  | Feature                                             | Impl | Corrected Impact | Corrected Effort | Why the correction matters                                                                   |
| -- | --------------------------------------------------- | ---: | ---------------: | ---------------: | -------------------------------------------------------------------------------------------- |
| 1  | Weighted blended OIT for water/volumetrics/overlays |    2 |                4 |          **1–2** | Core OIT already exists; main work is *wiring into passes* + validation.                     |
| 2  | Sun position (lat/lon/date/time → az/el)            |    1 |            **4** |          **2–3** | No ephemeris/timezone API yet; needs deterministic math + UX + tests.                        |
| 3  | Map plate compositor (title/legend/scale/inset)     |    1 |                4 |          **3–4** | You have overlay + text overlay primitives, but layout/typography/legend are nontrivial.     |
| 4  | TAA (jitter + reprojection + clamp)                 |    1 |                4 |            **4** | AO temporal ≠ full-frame TAA; needs history buffers + reprojection + rejection logic.        |
| 5  | Shadow modes VSM/EVSM/MSM (+ blur/leak control)     |    2 |                4 |          **2–3** | The shader + moment pass exist; the remaining cost is *quality/blur/leak* + API coherence.   |
| 6  | Terrain LOD “geometry clipmaps / nested grids”      |    1 |                5 |          **4–5** | Current LOD code is mostly “paper LOD”; true clipmaps need new mesh strategy + seams/morphs. |
| 7  | COG streaming (HTTP range + overviews + cache)      |    0 |                5 |            **5** | No HTTP-range infra in Rust tree; decoding + caching + tests are big.                        |
| 8  | PMTiles vector streaming + styling/declutter        |    0 |                4 |          **4–5** | Needs tile index reading, feature decode, styling DSL, and label declutter integration.      |
| 9  | 3D Tiles import/export                              |    0 |                4 |            **5** | Interop is large: tileset.json, multiple payload formats, metadata, bounding volumes, LOD.   |
| 10 | Point clouds (EPT/COPC) + splats                    |    0 |                5 |               5+ | This is a full subsystem: streaming, LOD, rendering, picking, memory budgets.                |

---

# Appendix B — “Claude implementation state assessment” table (original Claude section preserved)

| #  | Feature                         | Report Estimate       | **Actual State**            | **Revised Effort** |
| -- | ------------------------------- | --------------------- | --------------------------- | ------------------ |
| 1  | **OIT (Weighted Blended)**      | Impact 4 / Effort 2-3 | **~90% complete**           | **Effort 1**       |
| 2  | **Sun Position (Ephemeris)**    | Impact 3 / Effort 1-2 | ~20% (manual controls only) | Effort 2 ✓         |
| 3  | **Map Plate Compositor**        | Impact 4 / Effort 2-3 | ~5% (basic helpers)         | Effort 3 ✓         |
| 4  | **TAA**                         | Impact 4 / Effort 3   | ~30% (AO/SSR temporal)      | **Effort 3-4**     |
| 5  | **Shadow Modes (VSM/EVSM/MSM)** | Impact 4 / Effort 3-4 | **~85% complete**           | **Effort 1-2**     |
| 6  | **Terrain LOD (Clipmaps)**      | Impact 5 / Effort 3-4 | ~40% (screen-space error)   | Effort 4 ✓         |
| 7  | **COG Streaming**               | Impact 5 / Effort 4   | ~25% (local tile streaming) | Effort 4 ✓         |
| 8  | **PMTiles**                     | Impact 4 / Effort 4   | 0%                          | Effort 4 ✓         |
| 9  | **3D Tiles**                    | Impact 4 / Effort 4-5 | 0%                          | **Effort 5**       |
| 10 | **Point Cloud (EPT/COPC)**      | Impact 5 / Effort 5   | ~10% (basic point render)   | Effort 5 ✓         |