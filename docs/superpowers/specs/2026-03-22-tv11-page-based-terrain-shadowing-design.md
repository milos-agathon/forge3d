# TV11 — Page-Based Terrain Shadowing

**Date:** 2026-03-22
**Epic:** TV11 from `docs/plans/2026-03-16-terrain-viz-epics.md`
**Scope:** TV11.1–TV11.5: stable per-cascade page domains, shadow page cache, CPU-driven paged rendering, stability tests, terrain API and example
**Target path:** Terrain renderer only (`src/terrain/renderer/`, `src/shaders/terrain_pbr_pom.wgsl`). The separate viewer terrain shader (`src/viewer/terrain/shader_pbr/terrain_pbr.wgsl`) is not in scope for v1.

---

## 1. Problem

The terrain shadow depth pass computes one terrain-covering light matrix and reuses it for every cascade (`src/viewer/terrain/render/shadow.rs:56`, `:89`, `:143`). Although the surrounding architecture is cascade-shaped — split generation exists in `src/core/cascade_split.rs:74` and `:123`, and the pass iterates per-cascade views — every cascade renders the same shadow coverage. The result is shadow shimmer and swimming when the camera pans across large terrains, and visible banding at cascade transition boundaries.

The existing stabilization (texel-grid snapping in `pipeline_init.rs`) is insufficient because it operates on a shared matrix rather than per-cascade domains, and the fixed 512-vertex shadow depth grid lacks the spatial granularity to benefit from page-based allocation.

**Value proposition:** Eliminate terrain shadow shimmer/swim and cascade-transition artifacts through page-based shadow allocation on stable per-cascade light-space domains, building toward a C-ready architecture for future demand-driven resolution scaling.

---

## 2. Non-Goals (v1)

- **GPU feedback-driven page requests.** V1 uses CPU-generated request sets. GPU feedback is the natural second phase once page domain math, residency accounting, and eviction behavior are proven.
- **Parent-page hierarchy.** No mip-chain or fallback-to-coarser-page in v1. All requested pages must be resident before the beauty pass; a missing page at render time is an error/debug condition.
- **Moment-based paged shadows.** Paged v1 is depth-atlas only. VSM, EVSM, and MSM require moment maps that the paged atlas does not store.
- **Default flip.** `shadow_backend` defaults to `"csm"`. The default changes to `"paged"` only in a later release once stability, performance, and artifact evidence justify it.

---

## 3. Task Decomposition

| Task | Scope | Definition of done |
|---|---|---|
| **TV11.1 Fix per-cascade light-space domains** | Replace the shared terrain light matrix with proper per-cascade frustum fitting. Each cascade gets its own light-space projection covering only its depth range. | Each cascade renders a distinct depth range; `SHADOW_DEBUG_CASCADES` visualization shows four clearly different coverage regions; existing PCF/PCSS quality is preserved or improved. |
| **TV11.2 Establish stable virtual page domains and shadow page cache** | Define a per-cascade virtual page grid in light space with texel-grid-snapped boundaries. Build a shadow-specific page cache with LRU eviction modeled on the VT cache/LRU pattern (`src/core/tile_cache/cache.rs`), budget partitioning (fixed overhead + resident pool), and residency stats. Include `debug_all_pages_resident=true` mode. | Page domains are stable under camera motion; `debug_all_pages_resident` produces shadow factors equivalent to the repaired per-cascade CSM path within tolerance (full paged pipeline exercised, not bypassed); page cache respects `shadow_page_budget_mb` (default 64, cap 256); residency stats (resident count, hits, misses, evictions) are queryable. |
| **TV11.3 Add CPU-driven page request generation and paged shadow rendering** | Generate page requests from CPU visibility: receiver-driven (camera-frustum terrain) plus caster margin (off-screen terrain that can cast into the visible region). Allocate/evict pages, render only requested pages into the shadow atlas. Wire paged sampling into the terrain shader behind `shadow_backend = "paged"`. GPU feedback-driven requests are deferred to a future phase. | Paged backend preserves occluder coverage and passes the TV11 stability metrics while rendering only requested pages; memory usage scales with visible page count, not total virtual space. |
| **TV11.4 Add stability and residency tests** | Implement the three core DoD criteria: world-space translation stability (reprojected shadow factor on matched receivers), cascade transition continuity (no derivative spike at cascade boundaries), and residency churn (bounded after warm-up). Add secondary rotation stability test. | All three core tests pass on representative DEM scenes (Fuji, Rainier); rotation test passes with looser thresholds; tests are metric-based, no golden images required. |
| **TV11.5 Expose terrain shadow settings and create example** | Add `shadow_backend`, `shadow_page_tile_size`, and `shadow_page_budget_mb` to the terrain shadow settings surface (`terrain_params.py`, `native_lighting.rs`). Expose paged-shadow stats. Create `terrain_tv11_paged_shadow_demo.py` using real DEM assets. Document the feature. | Example renders with both backends and saves comparison PNGs; stats are queryable from Python; documentation covers what TV11 achieves, how to enable it, and budget tuning. |

---

## 4. Architecture

### 4.1 Per-Cascade Page Domain Model

Each cascade `i` (0..3) gets:

- A **proper light-space frustum fit** from the existing split generation in `src/core/cascade_split.rs`. The shared-matrix bug in `shadow.rs` is fixed so each cascade computes its own orthographic light-space projection covering only its assigned depth range.
- A **virtual page grid**: the cascade's light-space AABB divided into `pages_x × pages_y` pages. Non-square cascade AABBs produce non-square grids.
- **Texel-grid snapping**: page boundaries align to a fixed grid in light space. Pages do not shift with the camera — this is the core shimmer fix.

### 4.2 Page Representation

```rust
struct ShadowPage {
    cascade: u32,
    page_x: u32,
    page_y: u32,
    light_space_aabb: [f32; 4],  // light-space coverage of this page
}
```

**V1 page size:** 256×256 texels of `Depth32Float` = 256 KiB per page. At the default 64 MiB budget minus fixed overhead, the resident pool holds approximately 250 pages.

### 4.3 Shadow Page Cache

Modeled on the VT `TileCache` LRU pattern in `src/core/tile_cache/cache.rs` — used as a design pattern, not a drop-in dependency. The shadow page cache is a separate, shadow-specific implementation.

- **Page table** (GPU storage buffer): maps `(cascade, page_x, page_y)` → atlas slot index, or a not-resident sentinel.
- **Resident page pool**: a single `Depth32Float` atlas texture. Atlas dimensions sized to fit `(budget - fixed_overhead) / page_tile_bytes` pages. Slots assigned by LRU eviction.
- **Budget partitioning**: `shadow_page_budget_mb` is total paged-shadow backend memory. Fixed bytes are reserved first for page table buffer, upload staging, and bookkeeping. The remainder goes to the resident page pool. A 128 MiB budget is a user override, not the default.
- **Stats** (queryable from Python):
  - `shadow_pages_resident: u32`
  - `shadow_pages_rendered: u32`
  - `shadow_cache_hits: u32`
  - `shadow_cache_misses: u32`
  - `shadow_cache_evictions: u32`
  - `shadow_fixed_overhead_bytes: u64`
  - `shadow_resident_pool_bytes: u64`

### 4.4 Request → Render → Sample Flow

1. **Request** (CPU, v1): For each cascade, identify pages whose light-space coverage contains visible receivers (camera-frustum terrain) plus a caster margin (off-screen terrain that can cast shadows into the visible region). Receiver-driven, not view-only.
2. **Allocate**: Requested pages that are not already resident get atlas slots via LRU eviction. All requested pages must be resident before the beauty pass. A missing page at render time is an error/debug condition, not normal fallback behavior.
3. **Render**: Only newly allocated or invalidated pages get their shadow depth rendered. Each page renders terrain shadow depth for its light-space sub-frustum into its atlas slot.
4. **Sample** (WGSL): The paged terrain shader variant looks up `(cascade, page_x, page_y)` in the page table, computes the atlas UV with guard band clamping, and performs a depth comparison sample.

### 4.5 `debug_all_pages_resident=true`

Forces all pages in all cascades to be requested every frame. Still goes through the full paged pipeline: page table lookup, atlas allocation, paged shadow sampling. Validates addressing and domain math against the repaired per-cascade CSM path within tolerance. Does not bypass the paged backend. Available as a manual debug/validation mode, not a CI test.

---

## 5. Shader Strategy

### 5.1 Two Shader Variants from Shared Source

A single WGSL module cannot declare `@group(3)` with CSM resource types for one pipeline and paged resource types for another — binding types are part of the shader interface and must match the pipeline layout at compile time. TV11 produces two shader variants through the existing preprocessing path in `pipeline_cache.rs:55`:

- **`terrain_pbr_pom_csm`**: Group 3 declares CSM bindings (current shader, unchanged). Contains `sample_shadow_pcf_terrain` and cascade blend logic.
- **`terrain_pbr_pom_paged`**: Group 3 declares paged bindings. Contains `sample_shadow_paged` and the same cascade blend logic.

All shared code — PBR, POM, triplanar, material layers, lighting composition, cascade selection, cascade blending, debug modes — lives in common source fragments included by both variants. Only the Group 3 declarations and the shadow sampling function differ. No runtime backend branching in the shader.

### 5.2 Bind Group Layouts

**CSM layout (Group 3) — unchanged:**

| Binding | Type | Resource |
|---|---|---|
| 0 | Storage buffer | `CsmUniforms` |
| 1 | `texture_depth_2d_array` | Shadow depth cascades |
| 2 | `sampler_comparison` | Depth comparison sampler |
| 3 | `texture_2d_array<f32>` | Moment maps |
| 4 | `sampler` | Moment sampler |

**Paged layout (Group 3) — new, 4 bindings:**

| Binding | Type | Resource |
|---|---|---|
| 0 | Storage buffer | `PagedShadowUniforms` (cascade matrices, split distances, bias params, page grid dims per cascade, atlas size, page tile size, guard band size) |
| 1 | Storage buffer | `array<PageTableEntry>` (page table) |
| 2 | `texture_depth_2d` | Shadow atlas (single flat depth texture) |
| 3 | `sampler_comparison` | Depth comparison sampler |

### 5.3 Pipeline Cache Key

`TerrainPipelineCacheKey` extends to include `shadow_backend`:

```rust
struct TerrainPipelineCacheKey {
    sample_count: u32,
    shadow_backend: ShadowBackend, // Csm | Paged
    // ... existing fields ...
}
```

The pipeline cache creates the correct shader variant and bind group layout at init time. Two separate `BindGroupLayout` instances are created; the terrain pass selects the correct one based on the active backend.

### 5.4 Paged Shadow Sampling

```wgsl
fn sample_shadow_paged(world_pos: vec3f, normal: vec3f, cascade_idx: u32) -> f32 {
    // 1. Transform to cascade's light space
    let light_pos = paged_uniforms.cascade_light_vp[cascade_idx] * vec4f(world_pos, 1.0);
    let light_uv = light_pos.xy * 0.5 + 0.5;

    // 2. Early-out if outside cascade coverage
    if (any(light_uv < vec2f(0.0)) || any(light_uv > vec2f(1.0))) {
        return 1.0; // unshadowed
    }

    // 3. Compute and clamp page coordinates
    let pages = vec2f(f32(paged_uniforms.pages_x[cascade_idx]),
                      f32(paged_uniforms.pages_y[cascade_idx]));
    let page_x = clamp(u32(light_uv.x * pages.x), 0u, paged_uniforms.pages_x[cascade_idx] - 1u);
    let page_y = clamp(u32(light_uv.y * pages.y), 0u, paged_uniforms.pages_y[cascade_idx] - 1u);

    // 4. Page table lookup
    let entry = page_table[page_index(cascade_idx, page_x, page_y)];
    if (entry.atlas_offset_x == PAGE_NOT_RESIDENT) {
        return 0.5; // deterministic fallback — should not occur in normal frames
    }

    // 5. Compute atlas UV with guard band clamping
    let local_uv = fract(light_uv * pages);
    let guard = paged_uniforms.guard_band_texels / f32(paged_uniforms.page_tile_size);
    let clamped_uv = clamp(local_uv, vec2f(guard), vec2f(1.0 - guard));
    let tile_size = f32(paged_uniforms.page_tile_size);
    let atlas_origin = vec2f(f32(entry.atlas_offset_x), f32(entry.atlas_offset_y));
    let atlas_uv = (atlas_origin * tile_size + clamped_uv * tile_size) / paged_uniforms.atlas_size;

    // 6. Depth comparison with bias (same bias logic as current path)
    let bias = compute_shadow_bias(normal, cascade_idx);
    return textureSampleCompare(shadow_atlas, shadow_cmp_sampler, atlas_uv, light_pos.z - bias);
}
```

### 5.5 Guard Band Strategy

Each page is rendered with a guard band of `G` texels on each edge (e.g. G=4 for a 5×5 PCF kernel). The page's light-space sub-frustum is expanded by `G` texels worth of world-space coverage to capture the border region. The sampling function clamps `local_uv` to `[G/tile_size, 1 - G/tile_size]` so PCF/PCSS taps never read across page boundaries in the atlas. Guard band texels are included in page memory accounting.

### 5.6 Cascade Transition Integration

Paged sampling plugs into the existing cascade blend path. The current `cascade_blend_range` logic (`terrain_pbr_pom.wgsl:1178`) interpolates shadow factors from two adjacent cascades in the transition zone. The paged variant replaces only the per-cascade sample function — cascade selection and blending are shared code:

```wgsl
fn evaluate_terrain_shadow_paged(world_pos: vec3f, normal: vec3f, view_depth: f32) -> f32 {
    let cascade = select_cascade_terrain(view_depth);
    let shadow_a = sample_shadow_paged(world_pos, normal, cascade);
    let shadow_b = sample_shadow_paged(world_pos, normal, min(cascade + 1u, MAX_CASCADES - 1u));
    let blend = cascade_blend_factor(view_depth, cascade);
    return mix(shadow_a, shadow_b, blend);
}
```

### 5.7 Debug Modes

- `SHADOW_DEBUG_CASCADES`: unchanged, colors by cascade index.
- `SHADOW_DEBUG_PAGES` (new): shadow debug mode alongside `SHADOW_DEBUG_CASCADES`. Visualizes page grid boundaries and residency status (green = resident, red = not resident, blue = guard band region).
- `DBG_SHADOW_FACTOR`: unchanged, used by stability tests for reprojection comparison.

---

## 6. Python API

### 6.1 Terrain Shadow Settings (`terrain_params.py`)

```python
@dataclass
class ShadowSettings:
    enabled: bool = True
    technique: str = "PCF"               # filtering: "NONE" | "HARD" | "PCF" | "PCSS"
    shadow_backend: str = "csm"          # allocation: "csm" | "paged"
    resolution: int = 2048               # CSM: per-cascade map size
    shadow_page_tile_size: int = 256     # Paged: tile size in texels
    shadow_page_budget_mb: int = 64      # Paged: total backend memory, cap 256
    cascades: int = 4
    max_distance: float = 3000.0
    # ... existing fields unchanged ...

    PAGED_SUPPORTED_TECHNIQUES = {"NONE", "HARD", "PCF", "PCSS"}

    def validate_for_terrain(self):
        # existing technique validation ...
        if self.shadow_backend not in ("csm", "paged"):
            raise ValueError(f"shadow_backend must be 'csm' or 'paged', got '{self.shadow_backend}'")
        if self.shadow_backend == "paged":
            if self.shadow_page_budget_mb > 256:
                raise ValueError("shadow_page_budget_mb exceeds 256 MiB hard cap")
            if self.technique not in self.PAGED_SUPPORTED_TECHNIQUES:
                raise ValueError(
                    f"shadow_backend='paged' does not support technique '{self.technique}'; "
                    f"supported: {sorted(self.PAGED_SUPPORTED_TECHNIQUES)}"
                )
```

`shadow_backend` is terrain-renderer policy, living on `ShadowSettings` as consumed by the terrain path (`terrain_params.py` → `native_lighting.rs`), not on the generic low-level shadow structs.

### 6.2 Rust Decode (`native_lighting.rs`)

Extends the existing terrain shadow param decode to read `shadow_backend`, `shadow_page_tile_size`, and `shadow_page_budget_mb`. Routes to either the existing CSM setup or the new paged setup path based on backend value. Performs Rust-side validation matching the Python validation before GPU resource creation.

### 6.3 Queryable Stats

When `shadow_backend = "paged"`, stats are exposed through the existing memory/stats reporting surface:

| Stat | Type | Meaning |
|---|---|---|
| `shadow_pages_resident` | `u32` | Currently resident pages in the atlas |
| `shadow_pages_rendered` | `u32` | Pages rendered this frame |
| `shadow_cache_hits` | `u32` | Page requests satisfied from cache |
| `shadow_cache_misses` | `u32` | Page requests requiring allocation |
| `shadow_cache_evictions` | `u32` | Pages evicted to make room |
| `shadow_fixed_overhead_bytes` | `u64` | Page table + staging + bookkeeping |
| `shadow_resident_pool_bytes` | `u64` | Resident page atlas memory |

---

## 7. Memory Budget

### 7.1 Budget Layers

The repo has three existing budget layers:

| Layer | Cap | Source |
|---|---|---|
| Single shadow map | 32 MiB | `src/lighting/shadow_map.rs:110` |
| Shadow manager | 256 MiB | `src/shadows/manager/types.rs:4` |
| Global terrain/system | 512 MiB | `terrain_params.py:78`, `config.py:548` |

TV11 adds `shadow_page_budget_mb` as a self-contained budget for the paged shadow backend:

- **Default:** 64 MiB
- **Hard validation cap:** 256 MiB
- **Meaning:** Total paged-shadow backend memory, not just resident pages.

### 7.2 Budget Partitioning

```
shadow_page_budget_mb (e.g. 64 MiB)
├── Fixed overhead (page table buffer, upload staging, bookkeeping)
│   └── Typically < 1 MiB for v1 page counts
└── Resident page pool (remainder)
    └── At 256 KiB per page (256×256 Depth32Float): ~250 pages in 64 MiB
```

The allocator reserves fixed overhead first, then sizes the resident pool from the remainder. Budget is enforced: page requests that would exceed the pool trigger LRU eviction.

---

## 8. Testing Strategy

### 8.1 Core DoD Tests (TV11.4)

**Test 1 — World-space translation stability (blocking):**
Render frame A, translate camera by a small amount, render frame B. Use terrain depth AOV (per `terrain-tv2-aovs.md`) to reproject visible terrain points from A into B. Compare `DBG_SHADOW_FACTOR` only on matched receivers — same world-space point, not same screen pixel. Assert max shadow-factor delta < threshold (e.g. 0.02) on matched points. Run on Fuji and Rainier DEMs.

**Test 2 — Cascade transition continuity (blocking):**
Use `SHADOW_DEBUG_CASCADES` to locate cascade split regions. Render `DBG_SHADOW_FACTOR` across those regions. Compute the spatial derivative of shadow factor perpendicular to the split boundary. Assert no derivative spike exceeding threshold. The existing `cascade_blend_range` (`csm_types.rs`) should produce smooth transitions; TV11 must prove the boundary band no longer produces a visible jump.

**Test 3 — Residency churn stability (blocking):**
Render 10 frames of a repeatable micro-pan. After frame 3 (warm-up), assert that `shadow_cache_misses` and `shadow_cache_evictions` per frame are below a bound (e.g. < 5% of resident pages). Uses the queryable stats from the Python API.

### 8.2 Secondary Test

**Test 4 — Rotation stability (non-blocking, must ship):**
Same world-space reprojection method as Test 1, but with camera rotation instead of translation. Looser threshold (e.g. 0.05). Rotation changes visibility more than translation, so some churn is expected, but the test must exist in v1.

### 8.3 Validation Harness

`debug_all_pages_resident=true` — available as a manual debug mode to verify page addressing matches repaired CSM output within tolerance. Exercises the full paged pipeline. Not a CI test.

---

## 9. Paged v1 Technique Scope

| Technique | CSM backend | Paged backend v1 |
|---|---|---|
| `NONE` | Supported | Supported |
| `HARD` | Supported | Supported |
| `PCF` | Supported | Supported |
| `PCSS` | Supported | Supported |
| `VSM` | Supported | **Rejected** (requires moment maps) |
| `EVSM` | Supported | **Rejected** (requires moment maps) |
| `MSM` | Supported | **Rejected** (requires moment maps) |

Validation enforced in both Python (`validate_for_terrain`) and Rust (`native_lighting.rs`) before GPU resource creation.

---

## 10. Key Files

| Component | Location |
|---|---|
| Shadow depth pass (shared matrix bug) | `src/viewer/terrain/render/shadow.rs:56, :89, :143` |
| Cascade split generation | `src/core/cascade_split.rs:74, :123` |
| VT cache/LRU pattern reference | `src/core/tile_cache/cache.rs` |
| VT feedback buffer pattern reference | `src/core/feedback_buffer.rs:78` |
| Terrain PBR shader (shadow sampling) | `src/shaders/terrain_pbr_pom.wgsl` |
| Shadow bind group layout | `src/terrain/renderer/bind_groups/layouts.rs:4` |
| Pipeline cache | `src/terrain/renderer/pipeline_cache.rs:55, :142` |
| Python terrain shadow settings | `python/forge3d/terrain_params.py:50` |
| Rust terrain shadow decode | `src/terrain/render_params/native_lighting.rs:61` |
| CSM types (cascade_blend_range) | CSM types module |
| Shadow debug modes | `src/shaders/terrain_pbr_pom.wgsl:91, :3576` |
