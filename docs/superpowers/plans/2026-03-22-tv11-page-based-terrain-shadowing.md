# TV11 Page-Based Terrain Shadowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate terrain shadow shimmer/swim and cascade-transition artifacts through page-based shadow allocation on stable per-cascade light-space domains.

**Architecture:** Fix the shared-matrix bug so each cascade gets its own light-space projection, then build a shadow-specific page cache with LRU eviction on stable per-cascade virtual page grids. CPU-driven page requests identify visible pages, which are rendered into a flat depth atlas. A second shader variant (`terrain_pbr_pom_paged`) samples the atlas via a GPU page table. Two separate bind group layouts and pipeline cache keys avoid runtime type conflicts. Default backend remains `"csm"`; paged is opt-in.

**Tech Stack:** Rust (wgpu, bytemuck, glam), WGSL shaders, Python (dataclass + PyO3 bridge), pytest

**Spec:** `docs/superpowers/specs/2026-03-22-tv11-page-based-terrain-shadowing-design.md`

---

## File Structure

### New Files (Rust)

| File | Responsibility |
|------|---------------|
| `src/terrain/renderer/shadows/paged/mod.rs` | Module root, re-exports, `ShadowBackend` enum |
| `src/terrain/renderer/shadows/paged/domain.rs` | Per-cascade page domain computation, texel-grid snapping |
| `src/terrain/renderer/shadows/paged/cache.rs` | `ShadowPageCache` — LRU cache modeled on `src/core/tile_cache/cache.rs` |
| `src/terrain/renderer/shadows/paged/page_table.rs` | `PageTableGpu` — GPU storage buffer for page-to-atlas mapping |
| `src/terrain/renderer/shadows/paged/request.rs` | CPU page request generation (receiver-driven + caster margin) |
| `src/terrain/renderer/shadows/paged/atlas.rs` | Shadow atlas texture allocation and slot management |
| `src/terrain/renderer/shadows/paged/render.rs` | Per-page shadow depth rendering into atlas slots |
| `src/terrain/renderer/shadows/paged/uniforms.rs` | `PagedShadowUniforms` struct (bytemuck Pod) |
| `src/terrain/renderer/shadows/paged/stats.rs` | `PagedShadowStats` — queryable residency/churn stats |
| `src/terrain/renderer/shadows/paged/bind_group.rs` | `create_paged_shadow_bind_group()` — bind group creation for the paged backend |

### New Files (Shader)

| File | Responsibility |
|------|---------------|
| `src/shaders/terrain_shadow_csm.wgsl` | Extracted CSM-specific code: Group 3 CSM bindings, `CsmUniforms`/`ShadowCascade` structs, `sample_shadow_pcf_terrain()`, `calculate_shadow_terrain()` |
| `src/shaders/terrain_shadow_paged.wgsl` | Paged shadow Group 3 bindings, `PagedShadowUniforms` struct, `PageTableEntry` struct, `sample_shadow_paged()`, `evaluate_terrain_shadow_paged()`, `SHADOW_DEBUG_PAGES` mode |

### New Files (Python / Tests / Examples)

| File | Responsibility |
|------|---------------|
| `tests/test_terrain_tv11_paged_shadows.py` | Unit tests (validation, config), integration tests (stability, cascade continuity, residency churn, rotation) |
| `examples/terrain_tv11_paged_shadow_demo.py` | Offscreen demo with real DEM, CSM vs paged comparison, debug viz, stats output |
| `docs/terrain/tv11-page-based-terrain-shadowing.md` | Feature documentation |

### Modified Files

| File | Change |
|------|--------|
| `src/terrain/renderer/shadows/render.rs:96-117` | Fix per-cascade light-space projections using `cascade_split.rs` |
| `src/terrain/renderer/shadows/setup.rs:59-80` | Route to paged backend when `shadow_backend == "paged"` |
| `src/terrain/renderer/shadows.rs` | Add `pub mod paged;` (note: module root is `shadows.rs`, not `shadows/mod.rs`) |
| `src/terrain/renderer/bind_groups/layouts.rs:6-56` | Add `create_paged_shadow_bind_group_layout()` (4 bindings) |
| `src/terrain/renderer/bind_groups/terrain_pass.rs` | Group 3 bind group is NOT created here (groups 0, 4, 6 only); no change needed |
| `src/terrain/renderer/shadows/main_bind_group.rs:4` | Current CSM Group 3 bind group creation — paged backend adds parallel `create_paged_shadow_bind_group()` path |
| `src/terrain/renderer/pipeline_cache.rs:55-142` | Add `shadow_backend` parameter to `preprocess_terrain_shader()` and `create_render_pipeline()`, generate paged shader variant |
| `src/terrain/renderer/draw/setup/pipeline.rs:70-94` | `ensure_pipeline_sample_count()` passes `shadow_bind_group_layout` — must select correct layout based on backend |
| `src/terrain/renderer/core.rs:179` | `TerrainScene` stores `shadow_bind_group_layout` — must store both CSM and paged layouts, select at render time |
| `src/terrain/renderer/aov.rs:21` | AOV render path also uses shadow bind group — must use backend-appropriate bind group |
| `src/core/cascade_split.rs:164,220,283` | Make `extract_frustum_corners()`, `calculate_light_projection()`, and `calculate_texel_size()` public (`pub fn`) |
| `src/terrain/renderer/uniforms.rs` | Add `PagedShadowPassUniforms` for per-page depth rendering |
| `python/forge3d/terrain_params.py:50-165` | Add `shadow_backend`, `shadow_page_tile_size`, `shadow_page_budget_mb`, `debug_all_pages_resident` to `ShadowSettings` |
| `src/terrain/render_params/native_lighting.rs:74-110` | Add paged shadow fields to `ShadowSettingsNative` |
| `src/terrain/render_params/decode_lighting.rs:146-175` | Decode paged shadow fields from Python config |
| `src/shaders/terrain_pbr_pom.wgsl:313-1195` | Extract CSM shadow code to `terrain_shadow_csm.wgsl`, add `// SHADOW_BACKEND_INCLUDE` marker |

---

## Task 1: Fix Per-Cascade Light-Space Domains (TV11.1)

**Files:**
- Modify: `src/terrain/renderer/shadows/render.rs:96-123`
- Modify: `src/terrain/renderer/shadows/setup.rs:59-80`
- Test: `tests/test_terrain_tv11_paged_shadows.py`

- [ ] **Step 1: Write Python test for per-cascade domain separation**

Create `tests/test_terrain_tv11_paged_shadows.py` with a test that renders with `SHADOW_DEBUG_CASCADES` and verifies four distinct color regions:

```python
"""TV11 — Page-Based Terrain Shadowing tests."""
import os
import pytest
import numpy as np

try:
    import forge3d as f3d
    from forge3d.terrain_params import (
        ShadowSettings,
        MaterialLayerSettings,
        make_terrain_params_config,
    )
    FORGE3D_AVAILABLE = True
except ImportError:
    FORGE3D_AVAILABLE = False

try:
    from _terrain_runtime import terrain_rendering_available
    GPU_AVAILABLE = terrain_rendering_available()
except Exception:
    GPU_AVAILABLE = False


def _create_test_hdr(path):
    """Write a minimal 8x4 RADIANCE HDR for IBL testing."""
    import struct
    w, h = 8, 4
    with open(path, "wb") as fp:
        fp.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        fp.write(f"-Y {h} +X {w}\n".encode())
        for _ in range(h * w):
            fp.write(struct.pack("BBBB", 128, 128, 128, 128))


def _build_ridge_heightmap(size=256):
    """Synthetic ridge terrain with enough depth variation for cascade splits."""
    y = np.linspace(0, 1, size, dtype=np.float32)
    x = np.linspace(0, 1, size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    ridge = 0.3 * np.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / 0.02)
    slope = 0.2 * yy
    return (ridge + slope).astype(np.float32)


def _build_overlay(domain):
    cmap = f3d.Colormap1D.from_stops(
        stops=[(domain[0], "#2b5329"), (domain[1], "#f5f5f0")],
        domain=domain,
    )
    return f3d.OverlayLayer.from_colormap1d(cmap, strength=1.0)


def _render_with_shadow_debug(renderer, material_set, ibl, heightmap, domain, shadow_debug_mode=1):
    """Render with shadow debug mode enabled (1 = SHADOW_DEBUG_CASCADES)."""
    params_cfg = make_terrain_params_config(
        size_px=(256, 256),
        render_scale=1.0,
        terrain_span=3.2,
        msaa_samples=1,
        z_scale=1.45,
        exposure=1.0,
        domain=domain,
        albedo_mode="colormap",
        colormap_strength=1.0,
        light_azimuth_deg=220.0,
        light_elevation_deg=35.0,
        sun_intensity=2.5,
        cam_radius=2.6,
        cam_phi_deg=200.0,
        cam_theta_deg=55.0,
        fov_y_deg=50.0,
        camera_mode="screen",
        overlays=[_build_overlay(domain)],
        shadows=ShadowSettings(
            enabled=True,
            technique="PCF",
            resolution=2048,
            cascades=4,
            max_distance=4000.0,
            softness=1.0,
            intensity=0.8,
            slope_scale_bias=0.001,
            depth_bias=0.0005,
            normal_bias=0.0002,
            min_variance=1e-4,
            light_bleed_reduction=0.5,
            evsm_exponent=40.0,
            fade_start=1.0,
        ),
    )
    # Set shadow debug mode via the config dict
    params_cfg["shadow_debug_mode"] = shadow_debug_mode
    params = f3d.TerrainRenderParams(params_cfg)
    frame = renderer.render_terrain_pbr_pom(
        material_set=material_set,
        env_maps=ibl,
        params=params,
        heightmap=heightmap,
    )
    return frame.to_numpy()


@pytest.fixture(scope="module")
def tv11_render_env(tmp_path_factory):
    if not GPU_AVAILABLE:
        pytest.skip("TV11 tests require GPU-backed forge3d module")
    tmp = tmp_path_factory.mktemp("tv11")
    hdr_path = tmp / "test.hdr"
    _create_test_hdr(hdr_path)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    heightmap = _build_ridge_heightmap()
    domain = (float(heightmap.min()), float(heightmap.max()))
    return renderer, material_set, ibl, heightmap, domain, tmp


class TestPerCascadeDomains:
    """TV11.1: Verify each cascade renders a distinct depth range."""

    def test_cascade_debug_shows_four_regions(self, tv11_render_env):
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env
        pixels = _render_with_shadow_debug(renderer, material_set, ibl, heightmap, domain,
                                           shadow_debug_mode=1)
        # SHADOW_DEBUG_CASCADES colors: R=cascade0, G=cascade1, B=cascade2, Y=cascade3
        r_dominant = np.sum((pixels[:,:,0] > 128) & (pixels[:,:,1] < 64) & (pixels[:,:,2] < 64))
        g_dominant = np.sum((pixels[:,:,1] > 128) & (pixels[:,:,0] < 64) & (pixels[:,:,2] < 64))
        b_dominant = np.sum((pixels[:,:,0] < 64) & (pixels[:,:,1] < 64) & (pixels[:,:,2] > 128))
        # At least 3 distinct cascade regions should be visible
        visible_regions = sum(1 for count in [r_dominant, g_dominant, b_dominant] if count > 100)
        assert visible_regions >= 3, (
            f"Expected >= 3 distinct cascade regions, got {visible_regions}. "
            f"R={r_dominant}, G={g_dominant}, B={b_dominant}"
        )
```

- [ ] **Step 2: Run test to verify it fails or establishes baseline**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestPerCascadeDomains -v`
Expected: Either FAIL (if cascade regions are identical due to shared matrix) or PASS (if existing system partially works).

- [ ] **Step 3: Make cascade_split helper functions public**

The functions `extract_frustum_corners()` (line 164), `calculate_light_projection()` (line 220), and `calculate_texel_size()` (line 283) in `src/core/cascade_split.rs` are currently private (`fn`). Change them to `pub fn` so `render.rs` can call them directly:

```rust
// cascade_split.rs:164 — change fn to pub fn
pub fn extract_frustum_corners(...) -> Vec<Vec3> { ... }
// cascade_split.rs:220
pub fn calculate_light_projection(...) -> Mat4 { ... }
// cascade_split.rs:283
pub fn calculate_texel_size(...) -> f32 { ... }
```

- [ ] **Step 4: Fix per-cascade light-space projections in render.rs**

Modify `src/terrain/renderer/shadows/render.rs` to compute separate light-space projections per cascade. The current code at line 96 computes one `light_view_proj` and reuses it for all cascades at line 117. Fix by computing per-cascade frustum fits using the already-available near/far distances from the splits array.

**Critical contract notes:**
- `calculate_light_projection()` already returns `light_projection * light_view` (see `cascade_split.rs:279`). Do NOT multiply by `light_view` again — the result IS the combined light_view_proj.
- `calculate_texel_size()` takes `&[Vec3; 8]` frustum corners, NOT a matrix (see `cascade_split.rs:283`).

```rust
// BEFORE (line 96, shared matrix):
// let light_view_proj = light_proj * light_view;

// AFTER: remove shared light_view_proj, compute per-cascade inside the loop
for cascade_idx in 0..cascade_count as usize {
    let near_d = splits.get(cascade_idx).copied().unwrap_or(near_plane);
    let far_d = splits.get(cascade_idx + 1).copied().unwrap_or(far_plane);

    // Extract frustum corners for this cascade's depth range
    let corners = crate::core::cascade_split::extract_frustum_corners(
        view_matrix, proj_matrix, near_d, far_d,
    );

    // calculate_light_projection() returns (light_projection * light_view) directly —
    // the result IS the combined light_view_proj, do NOT multiply by light_view again.
    let cascade_light_view_proj = crate::core::cascade_split::calculate_light_projection(
        &corners,
        light_dir,
        self.csm_renderer.config.shadow_map_size as f32,
    );

    // calculate_texel_size() takes frustum corners, not a matrix
    let cascade_texel_size = crate::core::cascade_split::calculate_texel_size(
        &corners, self.csm_renderer.config.shadow_map_size as f32,
    );

    // Store the combined light_view_proj directly (no separate light_projection needed
    // since the shader only uses light_view_proj for shadow coordinate transform)
    self.csm_renderer.uniforms.cascades[cascade_idx].light_projection =
        cascade_light_view_proj.to_cols_array();
    self.csm_renderer.uniforms.cascades[cascade_idx].light_view_proj =
        cascade_light_view_proj.to_cols_array_2d();
    self.csm_renderer.uniforms.cascades[cascade_idx].near_distance = near_d;
    self.csm_renderer.uniforms.cascades[cascade_idx].far_distance = far_d;
    self.csm_renderer.uniforms.cascades[cascade_idx].texel_size = cascade_texel_size;
    self.csm_renderer.uniforms.cascades[cascade_idx]._padding = 0.0;
}
```

- [ ] **Step 5: Run test to verify cascade regions are now distinct**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestPerCascadeDomains -v`
Expected: PASS — cascade debug shows >= 3 distinct color regions.

- [ ] **Step 6: Run existing shadow tests to verify no regression**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/ -k shadow -v`
Expected: All existing shadow tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/cascade_split.rs src/terrain/renderer/shadows/render.rs tests/test_terrain_tv11_paged_shadows.py
git commit -m "feat(tv11.1): fix per-cascade light-space projections in terrain shadow pass"
```

---

## Task 2: Add Python and Rust API Fields (TV11.5 partial)

**Files:**
- Modify: `python/forge3d/terrain_params.py:50-165`
- Modify: `src/terrain/render_params/native_lighting.rs:74-110`
- Modify: `src/terrain/render_params/decode_lighting.rs:146-175`
- Test: `tests/test_terrain_tv11_paged_shadows.py`

- [ ] **Step 1: Write validation tests for new ShadowSettings fields**

Add to `tests/test_terrain_tv11_paged_shadows.py`:

```python
class TestShadowSettingsValidation:
    """TV11.5: Validate new paged shadow settings fields."""

    def test_default_shadow_backend_is_csm(self):
        s = ShadowSettings(
            enabled=True, technique="PCF", resolution=2048, cascades=4,
            max_distance=3000.0, softness=1.0, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0,
            fade_start=1.0,
        )
        assert s.shadow_backend == "csm"
        assert s.shadow_page_tile_size == 256
        assert s.shadow_page_budget_mb == 64
        assert s.debug_all_pages_resident is False

    def test_paged_backend_accepts_pcf(self):
        s = ShadowSettings(
            enabled=True, technique="PCF", shadow_backend="paged",
            resolution=2048, cascades=4, max_distance=3000.0,
            softness=1.0, intensity=0.8, slope_scale_bias=0.001,
            depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
            light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        s.validate_for_terrain()  # Should not raise

    def test_paged_backend_rejects_vsm(self):
        s = ShadowSettings(
            enabled=True, technique="VSM", shadow_backend="paged",
            resolution=2048, cascades=4, max_distance=3000.0,
            softness=1.0, intensity=0.8, slope_scale_bias=0.001,
            depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
            light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        with pytest.raises(ValueError, match="does not support technique"):
            s.validate_for_terrain()

    def test_paged_budget_cap_256(self):
        s = ShadowSettings(
            enabled=True, technique="PCF", shadow_backend="paged",
            shadow_page_budget_mb=300, resolution=2048, cascades=4,
            max_distance=3000.0, softness=1.0, intensity=0.8,
            slope_scale_bias=0.001, depth_bias=0.0005, normal_bias=0.0002,
            min_variance=1e-4, light_bleed_reduction=0.5, evsm_exponent=40.0,
            fade_start=1.0,
        )
        with pytest.raises(ValueError, match="256"):
            s.validate_for_terrain()

    def test_invalid_shadow_backend(self):
        s = ShadowSettings(
            enabled=True, technique="PCF", shadow_backend="invalid",
            resolution=2048, cascades=4, max_distance=3000.0,
            softness=1.0, intensity=0.8, slope_scale_bias=0.001,
            depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
            light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
        )
        with pytest.raises(ValueError, match="shadow_backend"):
            s.validate_for_terrain()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestShadowSettingsValidation -v`
Expected: FAIL — `shadow_backend` field does not exist yet.

- [ ] **Step 3: Add fields to Python ShadowSettings**

Modify `python/forge3d/terrain_params.py` to add the new fields to `ShadowSettings`:

```python
# New fields with defaults (add after existing fields):
shadow_backend: str = "csm"            # "csm" | "paged"
shadow_page_tile_size: int = 256       # Paged: tile size in texels
shadow_page_budget_mb: int = 64        # Paged: total backend memory, cap 256
debug_all_pages_resident: bool = False # Paged: force all pages resident

PAGED_SUPPORTED_TECHNIQUES: ClassVar[set] = {"NONE", "HARD", "PCF", "PCSS"}
```

Add to `validate_for_terrain()`:

```python
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

- [ ] **Step 4: Add fields to Rust ShadowSettingsNative**

Modify `src/terrain/render_params/native_lighting.rs` — add to `ShadowSettingsNative`:

```rust
pub shadow_backend: String,            // "csm" or "paged"
pub shadow_page_tile_size: u32,        // default 256
pub shadow_page_budget_mb: u32,        // default 64, cap 256
pub debug_all_pages_resident: bool,    // default false
```

Update `Default` impl with the new defaults.

- [ ] **Step 5: Decode new fields from Python config**

Modify `src/terrain/render_params/decode_lighting.rs` `parse_shadow_settings()`:

```rust
shadow_backend: shadows
    .getattr("shadow_backend")
    .ok()
    .and_then(|v| v.extract::<String>().ok())
    .unwrap_or_else(|| "csm".to_string()),
shadow_page_tile_size: shadows
    .getattr("shadow_page_tile_size")
    .ok()
    .and_then(|v| v.extract::<i64>().ok())
    .unwrap_or(256) as u32,
shadow_page_budget_mb: shadows
    .getattr("shadow_page_budget_mb")
    .ok()
    .and_then(|v| v.extract::<i64>().ok())
    .unwrap_or(64) as u32,
debug_all_pages_resident: shadows
    .getattr("debug_all_pages_resident")
    .ok()
    .and_then(|v| v.extract::<bool>().ok())
    .unwrap_or(false),
```

- [ ] **Step 6: Run validation tests**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestShadowSettingsValidation -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/forge3d/terrain_params.py src/terrain/render_params/native_lighting.rs src/terrain/render_params/decode_lighting.rs tests/test_terrain_tv11_paged_shadows.py
git commit -m "feat(tv11.5): add shadow_backend and paged shadow settings to Python and Rust API"
```

---

## Task 3: Build Shadow Page Cache and Domain Model (TV11.2)

**Files:**
- Create: `src/terrain/renderer/shadows/paged/mod.rs`
- Create: `src/terrain/renderer/shadows/paged/domain.rs`
- Create: `src/terrain/renderer/shadows/paged/cache.rs`
- Create: `src/terrain/renderer/shadows/paged/stats.rs`
- Modify: `src/terrain/renderer/shadows.rs` (module root — not `shadows/mod.rs`)

- [ ] **Step 1: Create module root and ShadowBackend enum**

Create `src/terrain/renderer/shadows/paged/mod.rs`:

```rust
pub mod domain;
pub mod cache;
pub mod stats;

/// Shadow allocation backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShadowBackend {
    Csm,
    Paged,
}

impl ShadowBackend {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "csm" => Ok(Self::Csm),
            "paged" => Ok(Self::Paged),
            other => Err(format!("invalid shadow_backend '{}', expected 'csm' or 'paged'", other)),
        }
    }
}
```

Add `pub mod paged;` to `src/terrain/renderer/shadows.rs` (the module root file — the shadows module uses the single-file-plus-directory pattern).

- [ ] **Step 2: Create page domain model**

Create `src/terrain/renderer/shadows/paged/domain.rs`:

```rust
use glam::{Mat4, Vec3};

/// A single shadow page in a cascade's virtual grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShadowPageId {
    pub cascade: u32,
    pub page_x: u32,
    pub page_y: u32,
}

/// Per-cascade virtual page grid definition.
#[derive(Debug, Clone)]
pub struct CascadePageDomain {
    pub cascade_index: u32,
    pub pages_x: u32,
    pub pages_y: u32,
    pub light_view_proj: Mat4,
    /// Light-space AABB: [min_x, min_y, max_x, max_y]
    pub light_aabb: [f32; 4],
    /// Page table offset for prefix-sum indexing
    pub page_table_offset: u32,
}

impl CascadePageDomain {
    /// Compute the light-space AABB of a specific page.
    pub fn page_light_aabb(&self, page_x: u32, page_y: u32) -> [f32; 4] {
        let w = self.light_aabb[2] - self.light_aabb[0];
        let h = self.light_aabb[3] - self.light_aabb[1];
        let px = w / self.pages_x as f32;
        let py = h / self.pages_y as f32;
        [
            self.light_aabb[0] + page_x as f32 * px,
            self.light_aabb[1] + page_y as f32 * py,
            self.light_aabb[0] + (page_x + 1) as f32 * px,
            self.light_aabb[1] + (page_y + 1) as f32 * py,
        ]
    }

    /// Total number of pages in this cascade.
    pub fn page_count(&self) -> u32 {
        self.pages_x * self.pages_y
    }
}

/// Compute page domains for all cascades.
///
/// `tile_size` is in texels (e.g. 256).
/// `cascade_projs` is the per-cascade light projections from TV11.1.
pub fn compute_page_domains(
    cascade_light_view_projs: &[(Mat4, [f32; 4])], // (light_view_proj, light_aabb) per cascade
    tile_size: u32,
    shadow_map_size: u32,
) -> Vec<CascadePageDomain> {
    let mut domains = Vec::with_capacity(cascade_light_view_projs.len());
    let mut offset = 0u32;

    for (i, (lvp, aabb)) in cascade_light_view_projs.iter().enumerate() {
        // Compute page grid dimensions from cascade AABB extents — non-square AABBs
        // produce non-square grids as required by the spec.
        let extent_x = aabb[2] - aabb[0];
        let extent_y = aabb[3] - aabb[1];
        let texel_size_x = extent_x / shadow_map_size as f32;
        let texel_size_y = extent_y / shadow_map_size as f32;
        let coverage_texels_x = (extent_x / texel_size_x).ceil() as u32;
        let coverage_texels_y = (extent_y / texel_size_y).ceil() as u32;
        let pages_x = (coverage_texels_x / tile_size).max(1);
        let pages_y = (coverage_texels_y / tile_size).max(1);

        domains.push(CascadePageDomain {
            cascade_index: i as u32,
            pages_x,
            pages_y,
            light_view_proj: *lvp,
            light_aabb: *aabb,
            page_table_offset: offset,
        });
        offset += pages_x * pages_y;
    }
    domains
}

/// Total page table entries across all cascades (for GPU buffer sizing).
pub fn total_page_table_entries(domains: &[CascadePageDomain]) -> u32 {
    domains.iter().map(|d| d.page_count()).sum()
}
```

- [ ] **Step 3: Create shadow page cache**

Create `src/terrain/renderer/shadows/paged/cache.rs`:

```rust
use std::collections::{HashMap, VecDeque};
use super::domain::ShadowPageId;
use super::stats::PagedShadowStats;

/// Atlas slot index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtlasSlot {
    /// Tile origin in atlas (column, row in tile units).
    pub col: u32,
    pub row: u32,
}

/// Shadow page cache with LRU eviction.
/// Modeled on src/core/tile_cache/cache.rs pattern.
pub struct ShadowPageCache {
    capacity: usize,
    atlas_cols: u32,
    atlas_rows: u32,
    resident: HashMap<ShadowPageId, AtlasSlot>,
    lru_queue: VecDeque<ShadowPageId>,
    free_slots: Vec<AtlasSlot>,
    stats: PagedShadowStats,
}

impl ShadowPageCache {
    pub fn new(budget_bytes: u64, tile_size: u32, fixed_overhead_bytes: u64) -> Self {
        let page_bytes = (tile_size * tile_size * 4) as u64; // Depth32Float
        let pool_bytes = budget_bytes.saturating_sub(fixed_overhead_bytes);
        let capacity = (pool_bytes / page_bytes) as usize;

        // Square-ish atlas layout
        let atlas_cols = (capacity as f64).sqrt().ceil() as u32;
        let atlas_rows = ((capacity as u32) + atlas_cols - 1) / atlas_cols;

        let mut free_slots = Vec::with_capacity(capacity);
        for row in (0..atlas_rows).rev() {
            for col in (0..atlas_cols).rev() {
                if (row * atlas_cols + col) < capacity as u32 {
                    free_slots.push(AtlasSlot { col, row });
                }
            }
        }

        Self {
            capacity,
            atlas_cols,
            atlas_rows,
            resident: HashMap::with_capacity(capacity),
            lru_queue: VecDeque::with_capacity(capacity),
            free_slots,
            stats: PagedShadowStats::default(),
        }
    }

    /// Look up a page. Returns slot if resident, updates LRU.
    pub fn access(&mut self, page: &ShadowPageId) -> Option<AtlasSlot> {
        if let Some(&slot) = self.resident.get(page) {
            self.stats.hits += 1;
            // Move to front of LRU
            if let Some(pos) = self.lru_queue.iter().position(|p| p == page) {
                self.lru_queue.remove(pos);
            }
            self.lru_queue.push_front(*page);
            Some(slot)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Allocate a page. Evicts LRU if full. Returns (slot, newly_allocated).
    pub fn allocate(&mut self, page: ShadowPageId) -> Option<(AtlasSlot, bool)> {
        if let Some(slot) = self.access(&page) {
            return Some((slot, false));
        }

        // Evict if at capacity
        while self.free_slots.is_empty() {
            if !self.evict_lru() {
                return None; // Cannot evict
            }
        }

        let slot = self.free_slots.pop()?;
        self.resident.insert(page, slot);
        self.lru_queue.push_front(page);
        // Note: miss already counted by the access() call above
        self.stats.pages_rendered += 1;
        Some((slot, true))
    }

    fn evict_lru(&mut self) -> bool {
        if let Some(evicted) = self.lru_queue.pop_back() {
            if let Some(slot) = self.resident.remove(&evicted) {
                self.free_slots.push(slot);
                self.stats.evictions += 1;
                return true;
            }
        }
        false
    }

    pub fn is_resident(&self, page: &ShadowPageId) -> bool {
        self.resident.contains_key(page)
    }

    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn atlas_dims(&self) -> (u32, u32) {
        (self.atlas_cols, self.atlas_rows)
    }

    pub fn stats(&self) -> &PagedShadowStats {
        &self.stats
    }

    pub fn reset_frame_stats(&mut self) {
        self.stats.pages_rendered = 0;
        self.stats.hits = 0;
        self.stats.misses = 0;
    }

    /// Returns all currently resident pages and their slots.
    pub fn resident_pages(&self) -> impl Iterator<Item = (&ShadowPageId, &AtlasSlot)> {
        self.resident.iter()
    }
}
```

- [ ] **Step 4: Create stats struct**

Create `src/terrain/renderer/shadows/paged/stats.rs`:

```rust
/// Queryable page cache statistics.
#[derive(Debug, Clone, Default)]
pub struct PagedShadowStats {
    pub pages_resident: u32,
    pub pages_rendered: u32,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub fixed_overhead_bytes: u64,
    pub resident_pool_bytes: u64,
}
```

- [ ] **Step 5: Add Rust unit tests for cache**

Add `#[cfg(test)]` module in `cache.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn page(cascade: u32, x: u32, y: u32) -> ShadowPageId {
        ShadowPageId { cascade, page_x: x, page_y: y }
    }

    #[test]
    fn test_cache_allocate_and_access() {
        let mut cache = ShadowPageCache::new(1024 * 1024, 256, 0); // ~4 pages
        let (slot, new) = cache.allocate(page(0, 0, 0)).unwrap();
        assert!(new);
        let (slot2, new2) = cache.allocate(page(0, 0, 0)).unwrap();
        assert!(!new2);
        assert_eq!(slot, slot2);
    }

    #[test]
    fn test_cache_eviction_at_capacity() {
        // Budget for exactly 2 pages (256*256*4 = 256KB each, budget = 512KB)
        let mut cache = ShadowPageCache::new(512 * 1024, 256, 0);
        cache.allocate(page(0, 0, 0)).unwrap();
        cache.allocate(page(0, 1, 0)).unwrap();
        assert_eq!(cache.resident_count(), 2);
        // Third allocation should evict LRU
        cache.allocate(page(0, 2, 0)).unwrap();
        assert_eq!(cache.resident_count(), 2);
        assert!(cache.stats().evictions >= 1);
    }

    #[test]
    fn test_cache_lru_order() {
        let mut cache = ShadowPageCache::new(768 * 1024, 256, 0); // ~3 pages
        cache.allocate(page(0, 0, 0));
        cache.allocate(page(0, 1, 0));
        cache.allocate(page(0, 2, 0));
        // Access page(0,0,0) to move it to front
        cache.access(&page(0, 0, 0));
        // Allocate a 4th page — should evict page(0,1,0) (least recently used)
        cache.allocate(page(0, 3, 0));
        assert!(!cache.is_resident(&page(0, 1, 0)));
        assert!(cache.is_resident(&page(0, 0, 0)));
    }
}
```

- [ ] **Step 6: Run Rust tests**

Run: `cd C:/Users/milos/forge3d && cargo test shadows::paged -- --nocapture`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/terrain/renderer/shadows/paged/ src/terrain/renderer/shadows/mod.rs
git commit -m "feat(tv11.2): add shadow page cache, domain model, and stats"
```

---

## Task 4: Build Page Table, Atlas, and Paged Bind Group Layout (TV11.2 continued)

**Files:**
- Create: `src/terrain/renderer/shadows/paged/page_table.rs`
- Create: `src/terrain/renderer/shadows/paged/atlas.rs`
- Create: `src/terrain/renderer/shadows/paged/uniforms.rs`
- Create: `src/terrain/renderer/shadows/paged/bind_group.rs`
- Modify: `src/terrain/renderer/bind_groups/layouts.rs`
- Modify: `src/terrain/renderer/pipeline_cache.rs`

- [ ] **Step 1: Create page table GPU buffer**

Create `src/terrain/renderer/shadows/paged/page_table.rs`:

```rust
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const PAGE_NOT_RESIDENT: u32 = 0xFFFFFFFF;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PageTableEntry {
    pub atlas_offset_x: u32,  // PAGE_NOT_RESIDENT if not loaded
    pub atlas_offset_y: u32,
}

pub struct PageTableGpu {
    buffer: wgpu::Buffer,
    entries: Vec<PageTableEntry>,
    entry_count: u32,
}

impl PageTableGpu {
    pub fn new(device: &wgpu::Device, total_entries: u32) -> Self {
        let entries = vec![
            PageTableEntry {
                atlas_offset_x: PAGE_NOT_RESIDENT,
                atlas_offset_y: PAGE_NOT_RESIDENT,
            };
            total_entries as usize
        ];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tv11.paged_shadow.page_table"),
            contents: bytemuck::cast_slice(&entries),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        Self { buffer, entries, entry_count: total_entries }
    }

    pub fn update_entry(&mut self, index: u32, slot_col: u32, slot_row: u32) {
        self.entries[index as usize] = PageTableEntry {
            atlas_offset_x: slot_col,
            atlas_offset_y: slot_row,
        };
    }

    pub fn clear_entry(&mut self, index: u32) {
        self.entries[index as usize] = PageTableEntry {
            atlas_offset_x: PAGE_NOT_RESIDENT,
            atlas_offset_y: PAGE_NOT_RESIDENT,
        };
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.entries));
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn byte_size(&self) -> u64 {
        (self.entry_count as u64) * std::mem::size_of::<PageTableEntry>() as u64
    }
}
```

- [ ] **Step 2: Create shadow atlas**

Create `src/terrain/renderer/shadows/paged/atlas.rs`:

```rust
pub struct ShadowAtlas {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    tile_size: u32,
    atlas_cols: u32,
    atlas_rows: u32,
}

impl ShadowAtlas {
    pub fn new(device: &wgpu::Device, tile_size: u32, atlas_cols: u32, atlas_rows: u32) -> Self {
        let width = tile_size * atlas_cols;
        let height = tile_size * atlas_rows;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tv11.paged_shadow.atlas"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, tile_size, atlas_cols, atlas_rows }
    }

    pub fn view(&self) -> &wgpu::TextureView { &self.view }
    pub fn tile_size(&self) -> u32 { self.tile_size }
    pub fn width(&self) -> u32 { self.tile_size * self.atlas_cols }
    pub fn height(&self) -> u32 { self.tile_size * self.atlas_rows }
    pub fn byte_size(&self) -> u64 { (self.width() as u64) * (self.height() as u64) * 4 }

    /// Get the viewport rect for a given atlas slot.
    pub fn slot_viewport(&self, col: u32, row: u32) -> (u32, u32, u32, u32) {
        (col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size)
    }
}
```

- [ ] **Step 3: Create PagedShadowUniforms**

Create `src/terrain/renderer/shadows/paged/uniforms.rs`:

```rust
use bytemuck::{Pod, Zeroable};

/// Maximum cascades supported.
pub const MAX_CASCADES: usize = 4;

#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PagedShadowUniforms {
    /// Per-cascade light view-projection matrices.
    pub cascade_light_vp: [[[f32; 4]; 4]; MAX_CASCADES],  // 4 * 64 = 256 bytes

    /// Per-cascade far distances for blend computation.
    pub cascade_far: [f32; MAX_CASCADES],                   // 16 bytes

    /// Per-cascade page grid dimensions.
    pub pages_x: [u32; MAX_CASCADES],                       // 16 bytes
    pub pages_y: [u32; MAX_CASCADES],                       // 16 bytes

    /// Per-cascade page table offsets (prefix-sum).
    pub page_table_offset: [u32; MAX_CASCADES],             // 16 bytes

    /// Shadow config.
    pub cascade_count: u32,                                  // 4 bytes
    pub page_tile_size: u32,                                 // 4 bytes
    pub guard_band_texels: f32,                              // 4 bytes
    pub cascade_blend_range: f32,                            // 4 bytes

    /// Atlas dimensions in texels.
    pub atlas_size: [f32; 2],                                // 8 bytes

    /// Bias params.
    pub depth_bias: f32,                                     // 4 bytes
    pub slope_bias: f32,                                     // 4 bytes

    /// Debug mode (0=none, 1=cascades, 2=raw, 3=pages).
    pub debug_mode: u32,                                     // 4 bytes
    pub technique: u32,                                      // 4 bytes
    pub pcf_kernel_size: u32,                                // 4 bytes
    pub _pad: u32,                                           // 4 bytes
}
```

- [ ] **Step 4: Create paged shadow bind group layout**

Modify `src/terrain/renderer/bind_groups/layouts.rs` — add:

```rust
pub fn create_paged_shadow_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("tv11.paged_shadow.bind_group_layout"),
        entries: &[
            // Binding 0: PagedShadowUniforms (storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: Page table (storage buffer)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: Shadow atlas (depth 2D texture)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 3: Comparison sampler
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
        ],
    })
}
```

- [ ] **Step 5: Add shadow_backend parameter to pipeline construction**

The pipeline cache does not use a formal cache key struct — it uses direct function calls (`preprocess_terrain_shader()`, `create_render_pipeline()`). Add `shadow_backend: ShadowBackend` as a parameter to both functions:

```rust
use crate::terrain::renderer::shadows::paged::ShadowBackend;

// Modify preprocess_terrain_shader() signature:
pub fn preprocess_terrain_shader(
    // ... existing params ...
    shadow_backend: ShadowBackend,
) -> String {
    // When shadow_backend == Paged: concatenate terrain_shadow_paged.wgsl
    //   instead of the CSM Group 3 bindings and shadow sampling functions
    // When shadow_backend == Csm: existing concatenation (unchanged)
}

// Modify create_render_pipeline() to accept and forward shadow_backend,
// selecting the correct bind group layout (CSM 5-binding or paged 4-binding)
```

See Task 5 for the shader factoring strategy.

- [ ] **Step 6: Update paged/mod.rs with new submodules**

Add to `src/terrain/renderer/shadows/paged/mod.rs`:

```rust
pub mod page_table;
pub mod atlas;
pub mod uniforms;
pub mod bind_group;
```

- [ ] **Step 7: Compile and run Rust tests**

Run: `cd C:/Users/milos/forge3d && cargo test shadows::paged -- --nocapture && cargo check`
Expected: PASS — all new code compiles, Rust tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/terrain/renderer/shadows/paged/ src/terrain/renderer/bind_groups/layouts.rs src/terrain/renderer/pipeline_cache.rs
git commit -m "feat(tv11.2): add page table, atlas, uniforms, and paged bind group layout"
```

---

## Task 5: Create Paged Shadow Shader Variant (TV11.2/TV11.3)

**Files:**
- Create: `src/shaders/terrain_shadow_paged.wgsl`
- Modify: `src/terrain/renderer/pipeline_cache.rs`

- [ ] **Step 1: Write paged shadow WGSL fragment**

Create `src/shaders/terrain_shadow_paged.wgsl` containing:
- `PagedShadowUniforms` struct (matching the Rust Pod struct)
- `PageTableEntry` struct
- Group 3 bindings (storage buffer, storage buffer, texture_depth_2d, sampler_comparison)
- `sample_shadow_paged()` function with guard band clamping
- `evaluate_terrain_shadow_paged()` with conditional cascade blending
- `SHADOW_DEBUG_PAGES` mode (page grid boundaries and residency visualization)
- `PAGE_NOT_RESIDENT` constant

Key shader code as specified in the design spec Section 5.4 and 5.6.

- [ ] **Step 2: Factor CSM shadow code out of terrain_pbr_pom.wgsl**

The current `terrain_pbr_pom.wgsl` is monolithic — CSM Group 3 declarations, `CsmUniforms` struct, `ShadowCascade` struct, `sample_shadow_pcf_terrain()`, and `calculate_shadow_terrain()` are embedded inline. To support two shader variants:

1. Extract the CSM-specific code (Group 3 bindings at lines 357-370, `CsmUniforms`/`ShadowCascade` structs at lines 313-355, and all shadow sampling functions at lines 846-1195) into a new file `src/shaders/terrain_shadow_csm.wgsl`.
2. The main `terrain_pbr_pom.wgsl` includes a marker comment (e.g. `// SHADOW_BACKEND_INCLUDE`) at the extraction point.
3. `preprocess_terrain_shader()` in `pipeline_cache.rs` concatenates either `terrain_shadow_csm.wgsl` or `terrain_shadow_paged.wgsl` at the marker position, depending on `shadow_backend`.
4. Shared functions used by both variants (e.g. `compute_shadow_bias()`, `select_cascade_terrain()`) stay in the main file above the marker.

- [ ] **Step 3: Wire shader variant into pipeline cache**

In `pipeline_cache.rs`, modify `preprocess_terrain_shader()`:
- Read the marker comment in `terrain_pbr_pom.wgsl`
- When `shadow_backend == Paged`: substitute `terrain_shadow_paged.wgsl` content
- When `shadow_backend == Csm`: substitute `terrain_shadow_csm.wgsl` content
- Use the corresponding bind group layout in `create_render_pipeline()`

- [ ] **Step 3: Verify shader compilation**

Run: `cd C:/Users/milos/forge3d && cargo check`
Expected: Compiles. The WGSL module should be validated by naga at pipeline creation time.

- [ ] **Step 4: Commit**

```bash
git add src/shaders/terrain_shadow_paged.wgsl src/terrain/renderer/pipeline_cache.rs
git commit -m "feat(tv11.3): add paged shadow shader variant and pipeline cache integration"
```

---

## Task 6: CPU Page Request Generation and Paged Rendering (TV11.3)

**Files:**
- Create: `src/terrain/renderer/shadows/paged/request.rs`
- Create: `src/terrain/renderer/shadows/paged/render.rs`
- Modify: `src/terrain/renderer/shadows/setup.rs`
- Modify: `src/terrain/renderer/shadows/paged/mod.rs`

- [ ] **Step 1: Add request and render submodules to paged/mod.rs**

Add to `src/terrain/renderer/shadows/paged/mod.rs`:
```rust
pub mod request;
pub mod render;
```

- [ ] **Step 2: Create CPU page request generator**

Create `src/terrain/renderer/shadows/paged/request.rs`:

```rust
use glam::{Mat4, Vec3, Vec4};
use super::domain::{CascadePageDomain, ShadowPageId};

/// Margin in page units added around the camera frustum to capture off-screen casters.
const CASTER_MARGIN_PAGES: i32 = 2;

/// Generate page requests for all cascades based on camera visibility.
pub fn generate_page_requests(
    domains: &[CascadePageDomain],
    view_proj: &Mat4,
    debug_all_resident: bool,
) -> Vec<ShadowPageId> {
    let mut requests = Vec::new();

    for domain in domains {
        if debug_all_resident {
            // Request ALL pages in all cascades
            for py in 0..domain.pages_y {
                for px in 0..domain.pages_x {
                    requests.push(ShadowPageId {
                        cascade: domain.cascade_index,
                        page_x: px,
                        page_y: py,
                    });
                }
            }
            continue;
        }

        // Project camera frustum corners into this cascade's light space
        let inv_vp = view_proj.inverse();
        let ndc_corners = [
            Vec4::new(-1.0, -1.0, 0.0, 1.0), Vec4::new(1.0, -1.0, 0.0, 1.0),
            Vec4::new(-1.0, 1.0, 0.0, 1.0),  Vec4::new(1.0, 1.0, 0.0, 1.0),
            Vec4::new(-1.0, -1.0, 1.0, 1.0), Vec4::new(1.0, -1.0, 1.0, 1.0),
            Vec4::new(-1.0, 1.0, 1.0, 1.0),  Vec4::new(1.0, 1.0, 1.0, 1.0),
        ];

        let mut min_px = i32::MAX;
        let mut min_py = i32::MAX;
        let mut max_px = i32::MIN;
        let mut max_py = i32::MIN;

        for ndc in &ndc_corners {
            let world = inv_vp * *ndc;
            let world = world / world.w;
            let light = domain.light_view_proj * world;
            let uv_x = (light.x * 0.5 + 0.5).clamp(0.0, 1.0);
            let uv_y = (light.y * 0.5 + 0.5).clamp(0.0, 1.0);
            let px = (uv_x * domain.pages_x as f32) as i32;
            let py = (uv_y * domain.pages_y as f32) as i32;
            min_px = min_px.min(px);
            min_py = min_py.min(py);
            max_px = max_px.max(px);
            max_py = max_py.max(py);
        }

        // Add caster margin
        min_px = (min_px - CASTER_MARGIN_PAGES).max(0);
        min_py = (min_py - CASTER_MARGIN_PAGES).max(0);
        max_px = (max_px + CASTER_MARGIN_PAGES).min(domain.pages_x as i32 - 1);
        max_py = (max_py + CASTER_MARGIN_PAGES).min(domain.pages_y as i32 - 1);

        for py in min_py..=max_py {
            for px in min_px..=max_px {
                requests.push(ShadowPageId {
                    cascade: domain.cascade_index,
                    page_x: px as u32,
                    page_y: py as u32,
                });
            }
        }
    }

    requests
}
```

- [ ] **Step 3: Create per-page shadow depth renderer**

Create `src/terrain/renderer/shadows/paged/render.rs`:

Renders shadow depth into each requested atlas slot. For each page:
1. Compute the page's light-space sub-frustum (with guard band expansion)
2. Set the viewport to the page's atlas slot
3. Draw the terrain shadow depth grid (reuse existing `terrain_shadow_depth.wgsl` with per-page `light_view_proj`)

Key function signature:
```rust
pub fn render_paged_shadow_passes(
    encoder: &mut wgpu::CommandEncoder,
    atlas: &ShadowAtlas,
    cache: &ShadowPageCache,
    domains: &[CascadePageDomain],
    newly_allocated: &[(ShadowPageId, AtlasSlot)],
    shadow_depth_pipeline: &wgpu::RenderPipeline,
    shadow_depth_bind_group_layout: &wgpu::BindGroupLayout,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    heightmap_view: &wgpu::TextureView,
    terrain_params: [f32; 4],
    height_curve: [f32; 4],
    guard_band_texels: u32,
)
```

- [ ] **Step 4: Wire paged backend into setup.rs**

Modify `src/terrain/renderer/shadows/setup.rs` to route to the paged backend when `shadow_backend == "paged"`:

```rust
if decoded.shadow.shadow_backend == "paged" {
    // 1. Compute page domains from per-cascade projections
    // 2. Generate CPU page requests
    // 3. Allocate pages in cache
    // 4. Render newly allocated pages
    // 5. Upload page table to GPU
    // 6. Create paged shadow bind group
    // Return ShadowSetup with paged bind group
} else {
    // Existing CSM path (unchanged)
}
```

- [ ] **Step 5: Write integration test for paged backend**

Add to `tests/test_terrain_tv11_paged_shadows.py`:

```python
class TestPagedBackendRendering:
    """TV11.3: Verify paged backend produces valid shadow output."""

    def test_paged_backend_renders_without_error(self, tv11_render_env):
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env
        params_cfg = make_terrain_params_config(
            size_px=(256, 256),
            render_scale=1.0,
            terrain_span=3.2,
            msaa_samples=1,
            z_scale=1.45,
            exposure=1.0,
            domain=domain,
            albedo_mode="colormap",
            colormap_strength=1.0,
            light_azimuth_deg=220.0,
            light_elevation_deg=35.0,
            sun_intensity=2.5,
            cam_radius=2.6,
            cam_phi_deg=200.0,
            cam_theta_deg=55.0,
            fov_y_deg=50.0,
            camera_mode="screen",
            overlays=[_build_overlay(domain)],
            shadows=ShadowSettings(
                enabled=True,
                technique="PCF",
                shadow_backend="paged",
                resolution=2048,
                cascades=4,
                max_distance=4000.0,
                softness=1.0,
                intensity=0.8,
                slope_scale_bias=0.001,
                depth_bias=0.0005,
                normal_bias=0.0002,
                min_variance=1e-4,
                light_bleed_reduction=0.5,
                evsm_exponent=40.0,
                fade_start=1.0,
            ),
        )
        params = f3d.TerrainRenderParams(params_cfg)
        frame = renderer.render_terrain_pbr_pom(
            material_set=material_set,
            env_maps=ibl,
            params=params,
            heightmap=heightmap,
        )
        pixels = frame.to_numpy()
        out_path = tmp / "paged_test.png"
        frame.save(str(out_path))
        assert out_path.exists()
        assert pixels.shape == (256, 256, 4)
        # Basic sanity: image is not all black or all white
        mean_luma = np.mean(pixels[:,:,:3].astype(np.float32)) / 255.0
        assert 0.05 < mean_luma < 0.95, f"Mean luminance {mean_luma} out of range"

    def test_debug_all_pages_equivalent_to_csm(self, tv11_render_env):
        """debug_all_pages_resident=True should produce output within tolerance of CSM."""
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env
        base_kwargs = dict(
            size_px=(256, 256), render_scale=1.0, terrain_span=3.2,
            msaa_samples=1, z_scale=1.45, exposure=1.0, domain=domain,
            albedo_mode="colormap", colormap_strength=1.0,
            light_azimuth_deg=220.0, light_elevation_deg=35.0, sun_intensity=2.5,
            cam_radius=2.6, cam_phi_deg=200.0, cam_theta_deg=55.0, fov_y_deg=50.0,
            camera_mode="screen", overlays=[_build_overlay(domain)],
        )
        # Render CSM
        csm_params = f3d.TerrainRenderParams(make_terrain_params_config(
            **base_kwargs,
            shadows=ShadowSettings(
                enabled=True, technique="PCF", shadow_backend="csm",
                resolution=2048, cascades=4, max_distance=4000.0,
                softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
            ),
        ))
        csm_frame = renderer.render_terrain_pbr_pom(
            material_set=material_set, env_maps=ibl, params=csm_params, heightmap=heightmap,
        )
        # Render paged with debug_all_pages_resident
        paged_params = f3d.TerrainRenderParams(make_terrain_params_config(
            **base_kwargs,
            shadows=ShadowSettings(
                enabled=True, technique="PCF", shadow_backend="paged",
                debug_all_pages_resident=True,
                resolution=2048, cascades=4, max_distance=4000.0,
                softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
            ),
        ))
        paged_frame = renderer.render_terrain_pbr_pom(
            material_set=material_set, env_maps=ibl, params=paged_params, heightmap=heightmap,
        )
        csm_px = csm_frame.to_numpy()[:,:,:3].astype(np.float32)
        paged_px = paged_frame.to_numpy()[:,:,:3].astype(np.float32)
        max_diff = np.max(np.abs(csm_px - paged_px))
        mean_diff = np.mean(np.abs(csm_px - paged_px))
        assert mean_diff < 5.0, f"Paged (all resident) vs CSM mean diff {mean_diff:.1f} > 5.0"
        assert max_diff < 30.0, f"Paged (all resident) vs CSM max diff {max_diff:.0f} > 30"
```

- [ ] **Step 6: Run integration tests**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestPagedBackendRendering -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/terrain/renderer/shadows/paged/ src/terrain/renderer/shadows/setup.rs tests/test_terrain_tv11_paged_shadows.py
git commit -m "feat(tv11.3): add CPU page requests, paged shadow rendering, and backend routing"
```

---

## Task 7: Expose Paged Shadow Stats via Python API (TV11.5 partial)

**Files:**
- Modify: `src/terrain/renderer/shadows/paged/mod.rs`
- Modify: `src/terrain/renderer/py_api.rs` (or equivalent Python-facing API)
- Modify: `python/forge3d/terrain_params.py`

Stats must be exposed before Task 8 (stability tests) so the residency churn test can query them.

- [ ] **Step 1: Wire stats from ShadowPageCache through to Python**

The `PagedShadowStats` struct (created in Task 3) needs to be accessible from Python. Add a `get_paged_shadow_stats()` method to the terrain renderer's Python API (following the pattern of `get_probe_memory_report()` from TV5). Returns a dict with keys matching the spec:

```python
# Returned dict shape:
{
    "shadow_pages_resident": int,
    "shadow_pages_rendered": int,
    "shadow_cache_hits": int,
    "shadow_cache_misses": int,
    "shadow_cache_evictions": int,
    "shadow_fixed_overhead_bytes": int,
    "shadow_resident_pool_bytes": int,
}
```

- [ ] **Step 2: Update Python type stub**

Add `get_paged_shadow_stats()` to the `TerrainRenderer` class in `python/forge3d/__init__.pyi:484`:

```python
def get_paged_shadow_stats(self) -> dict[str, int]: ...
```

- [ ] **Step 3: Commit**

```bash
git add src/terrain/renderer/shadows/paged/ src/terrain/renderer/py_api.rs python/forge3d/__init__.pyi
git commit -m "feat(tv11.5): expose paged shadow stats via Python API"
```

---

## Task 8: Stability and Residency Tests (TV11.4)

**Files:**
- Modify: `tests/test_terrain_tv11_paged_shadows.py`

- [ ] **Step 1: Write world-space translation stability test**

```python
class TestShadowStability:
    """TV11.4: Core stability DoD tests."""

    def test_translation_stability(self, tv11_render_env):
        """World-space shadow factor stability under camera translation.

        Uses render_with_aov() (TV2 AOV path via py_api.rs:96) to get both
        beauty frame and depth/normal AOVs. Reprojects terrain points from
        frame A into frame B using depth AOV, then compares shadow factor
        only on matched world-space receivers.
        """
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env

        def render_with_aov_at_phi(cam_phi_deg):
            """Render with AOV output for world-space reprojection."""
            params_cfg = make_terrain_params_config(
                size_px=(256, 256), render_scale=1.0, terrain_span=3.2,
                msaa_samples=1, z_scale=1.45, exposure=1.0, domain=domain,
                albedo_mode="colormap", colormap_strength=1.0,
                light_azimuth_deg=220.0, light_elevation_deg=35.0, sun_intensity=2.5,
                cam_radius=2.6, cam_phi_deg=cam_phi_deg,
                cam_theta_deg=55.0, fov_y_deg=50.0, camera_mode="screen",
                overlays=[_build_overlay(domain)],
                shadows=ShadowSettings(
                    enabled=True, technique="PCF", shadow_backend="paged",
                    resolution=2048, cascades=4, max_distance=4000.0,
                    softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                    depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                    light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
                ),
            )
            params = f3d.TerrainRenderParams(params_cfg)
            # render_with_aov returns (Frame, AovFrame) — the AovFrame carries
            # depth and normal channels for world-space reprojection.
            frame, aov = renderer.render_with_aov(
                material_set=material_set, env_maps=ibl, params=params,
                heightmap=heightmap,
            )
            return frame.to_numpy(), aov

        # Render two frames with small camera pan
        beauty_a, aov_a = render_with_aov_at_phi(200.0)
        beauty_b, aov_b = render_with_aov_at_phi(200.5)  # 0.5 degree pan

        # Use depth AOV to find matched receivers: pixels in A and B whose
        # world-space positions correspond to the same terrain point.
        # For micro-pan, depth channels overlap significantly in the center.
        depth_a = np.array(aov_a.depth_channel())  # float32 depth per pixel
        depth_b = np.array(aov_b.depth_channel())

        # Shadow factor proxy: extract from beauty luminance in shadowed regions.
        # A proper implementation would use a dedicated shadow-factor AOV channel;
        # for v1 we use luminance as the factor proxy on depth-matched receivers.
        luma_a = np.mean(beauty_a[:,:,:3].astype(np.float32), axis=2) / 255.0
        luma_b = np.mean(beauty_b[:,:,:3].astype(np.float32), axis=2) / 255.0

        # Match receivers: pixels where depth is valid and similar (same terrain point)
        valid_a = depth_a > 0
        valid_b = depth_b > 0
        # Center crop for conservative overlap after micro-pan
        crop = slice(80, 176)
        va = valid_a[crop, crop]
        vb = valid_b[crop, crop]
        matched = va & vb
        if np.sum(matched) < 100:
            pytest.skip("Insufficient matched receivers for stability test")

        diff = np.abs(luma_a[crop, crop] - luma_b[crop, crop])
        matched_diff = diff[matched]
        max_diff = float(np.max(matched_diff))
        mean_diff = float(np.mean(matched_diff))

        assert max_diff < 0.10, f"Max shadow factor delta {max_diff:.3f} > 0.10 on matched receivers"
        assert mean_diff < 0.02, f"Mean shadow factor delta {mean_diff:.4f} > 0.02 on matched receivers"
```

- [ ] **Step 2: Write cascade transition continuity test**

```python
    def test_cascade_transition_continuity(self, tv11_render_env):
        """No derivative spike at cascade boundaries.

        Uses two renders: one with SHADOW_DEBUG_CASCADES to locate boundary
        positions, and one with AOV to get per-pixel shadow-affected luminance
        on depth-valid receivers. Checks gradient smoothness across boundaries.
        """
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env

        # 1. Locate cascade boundaries via debug visualization
        cascade_viz = _render_with_shadow_debug(
            renderer, material_set, ibl, heightmap, domain, shadow_debug_mode=1)

        # 2. Render with AOV for shadow factor analysis
        params_cfg = make_terrain_params_config(
            size_px=(256, 256), render_scale=1.0, terrain_span=3.2,
            msaa_samples=1, z_scale=1.45, exposure=1.0, domain=domain,
            albedo_mode="colormap", colormap_strength=1.0,
            light_azimuth_deg=220.0, light_elevation_deg=35.0, sun_intensity=2.5,
            cam_radius=2.6, cam_phi_deg=200.0, cam_theta_deg=55.0, fov_y_deg=50.0,
            camera_mode="screen", overlays=[_build_overlay(domain)],
            shadows=ShadowSettings(
                enabled=True, technique="PCF", shadow_backend="paged",
                resolution=2048, cascades=4, max_distance=4000.0,
                softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
            ),
        )
        params = f3d.TerrainRenderParams(params_cfg)
        frame, aov = renderer.render_with_aov(
            material_set=material_set, env_maps=ibl, params=params, heightmap=heightmap,
        )
        beauty_px = frame.to_numpy()[:,:,:3].astype(np.float32) / 255.0
        depth = np.array(aov.depth_channel())

        # 3. Find cascade boundary rows from debug visualization
        cascade_color = cascade_viz[:,:,:3].astype(np.float32)
        row_diffs = np.max(np.abs(np.diff(cascade_color, axis=0)), axis=(1, 2))
        boundary_rows = np.where(row_diffs > 50)[0]

        if len(boundary_rows) == 0:
            pytest.skip("No cascade boundaries visible in test scene")

        # 4. Check shadow gradient is smooth across boundaries on valid terrain
        # Use per-pixel luminance as shadow-factor proxy on depth-valid receivers
        luma = np.mean(beauty_px, axis=2)
        valid = depth > 0
        luma_masked = np.where(valid, luma, np.nan)
        gradient = np.abs(np.diff(luma_masked, axis=0))

        for br in boundary_rows[:3]:
            if br < 5 or br >= luma.shape[0] - 5:
                continue
            # Gradient at boundary vs surrounding average
            local_grad = np.nanmean(gradient[br-1:br+2, :])
            surround_grad = np.nanmean(gradient[max(0,br-10):br-2, :])
            if surround_grad > 0.001:
                ratio = local_grad / surround_grad
                assert ratio < 5.0, (
                    f"Gradient spike at cascade boundary row {br}: "
                    f"local={local_grad:.4f}, surround={surround_grad:.4f}, ratio={ratio:.1f}"
                )
```

- [ ] **Step 3: Write residency churn stability test**

```python
    def test_residency_churn_after_warmup(self, tv11_render_env):
        """Page cache churn should be bounded after warm-up frames."""
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env
        warmup_frames = 3
        measure_frames = 7
        total_frames = warmup_frames + measure_frames

        for i in range(total_frames):
            phi = 200.0 + i * 0.3  # Slow micro-pan
            params_cfg = make_terrain_params_config(
                size_px=(256, 256), render_scale=1.0, terrain_span=3.2,
                msaa_samples=1, z_scale=1.45, exposure=1.0, domain=domain,
                albedo_mode="colormap", colormap_strength=1.0,
                light_azimuth_deg=220.0, light_elevation_deg=35.0, sun_intensity=2.5,
                cam_radius=2.6, cam_phi_deg=phi,
                cam_theta_deg=55.0, fov_y_deg=50.0, camera_mode="screen",
                overlays=[_build_overlay(domain)],
                shadows=ShadowSettings(
                    enabled=True, technique="PCF", shadow_backend="paged",
                    resolution=2048, cascades=4, max_distance=4000.0,
                    softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                    depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                    light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
                ),
            )
            params = f3d.TerrainRenderParams(params_cfg)
            renderer.render_terrain_pbr_pom(
                material_set=material_set, env_maps=ibl, params=params, heightmap=heightmap,
            )

        # Query stats after all frames (API wired in Task 7)
        stats = renderer.get_paged_shadow_stats()
        resident = stats["shadow_pages_resident"]
        evictions = stats["shadow_cache_evictions"]
        misses = stats["shadow_cache_misses"]

        # After warm-up, churn should be bounded: < 5% of resident pages per frame
        if resident > 0:
            eviction_rate = evictions / (measure_frames * resident)
            miss_rate = misses / (measure_frames * resident)
            assert eviction_rate < 0.05, (
                f"Eviction rate {eviction_rate:.3f} > 5% during micro-pan "
                f"({evictions} evictions over {measure_frames} frames, {resident} resident)"
            )
            assert miss_rate < 0.10, (
                f"Miss rate {miss_rate:.3f} > 10% during micro-pan "
                f"({misses} misses over {measure_frames} frames, {resident} resident)"
            )
```

- [ ] **Step 4: Write rotation stability test (secondary)**

```python
    def test_rotation_stability(self, tv11_render_env):
        """Shadow stability under camera rotation (looser thresholds)."""
        renderer, material_set, ibl, heightmap, domain, tmp = tv11_render_env

        def render_at_theta(theta):
            params_cfg = make_terrain_params_config(
                size_px=(256, 256), render_scale=1.0, terrain_span=3.2,
                msaa_samples=1, z_scale=1.45, exposure=1.0, domain=domain,
                albedo_mode="colormap", colormap_strength=1.0,
                light_azimuth_deg=220.0, light_elevation_deg=35.0, sun_intensity=2.5,
                cam_radius=2.6, cam_phi_deg=200.0,
                cam_theta_deg=theta, fov_y_deg=50.0, camera_mode="screen",
                overlays=[_build_overlay(domain)],
                shadows=ShadowSettings(
                    enabled=True, technique="PCF", shadow_backend="paged",
                    resolution=2048, cascades=4, max_distance=4000.0,
                    softness=1.0, intensity=0.8, slope_scale_bias=0.001,
                    depth_bias=0.0005, normal_bias=0.0002, min_variance=1e-4,
                    light_bleed_reduction=0.5, evsm_exponent=40.0, fade_start=1.0,
                ),
            )
            params = f3d.TerrainRenderParams(params_cfg)
            frame = renderer.render_terrain_pbr_pom(
                material_set=material_set, env_maps=ibl, params=params, heightmap=heightmap,
            )
            return frame.to_numpy()

        frame_a = render_at_theta(55.0)
        frame_b = render_at_theta(56.0)  # 1 degree rotation

        luma_a = np.mean(frame_a[64:192, 64:192, :3].astype(np.float32), axis=2) / 255.0
        luma_b = np.mean(frame_b[64:192, 64:192, :3].astype(np.float32), axis=2) / 255.0
        diff = np.abs(luma_a - luma_b)

        # Looser thresholds for rotation
        assert np.max(diff) < 0.25, f"Max diff {np.max(diff):.3f} > 0.25 under rotation"
        assert np.mean(diff) < 0.06, f"Mean diff {np.mean(diff):.4f} > 0.06 under rotation"
```

- [ ] **Step 5: Run all stability tests**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/test_terrain_tv11_paged_shadows.py::TestShadowStability -v`
Expected: PASS.

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/ -v --timeout=120`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_terrain_tv11_paged_shadows.py
git commit -m "test(tv11.4): add shadow stability, cascade continuity, and residency churn tests"
```

---

## Task 9: Create Example and Documentation (TV11.5)

**Files:**
- Create: `examples/terrain_tv11_paged_shadow_demo.py`
- Create: `docs/terrain/tv11-page-based-terrain-shadowing.md`

- [ ] **Step 1: Create example script**

Create `examples/terrain_tv11_paged_shadow_demo.py` following the TV5 demo pattern:

```python
"""TV11 — Page-Based Terrain Shadowing Demo.

Renders a terrain scene with both CSM and paged shadow backends,
producing comparison images and paged shadow statistics.

Usage:
    python terrain_tv11_paged_shadow_demo.py [--dem PATH] [--output DIR] [--width W] [--height H]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def _import_forge3d():
    try:
        import forge3d as f3d
        return f3d
    except ModuleNotFoundError:
        from _import_shim import ensure_repo_import
        ensure_repo_import()
        import forge3d as f3d
        return f3d

DEFAULT_DEM = Path(__file__).parent.parent / "assets" / "tif" / "Mount_Fuji_30m.tif"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "out" / "tv11_paged_shadows"

# ... (full demo implementation following TV5 pattern)
# Renders CSM, paged, debug_all_pages_resident, SHADOW_DEBUG_PAGES
# Saves comparison PNGs and stats to output directory
```

- [ ] **Step 2: Run example and verify image output**

Run: `cd C:/Users/milos/forge3d && python examples/terrain_tv11_paged_shadow_demo.py --width 512 --height 512`
Expected: Output PNGs saved to `examples/out/tv11_paged_shadows/`, no errors.

- [ ] **Step 3: Write feature documentation**

Create `docs/terrain/tv11-page-based-terrain-shadowing.md`:

Document:
- What TV11 achieves (shimmer/swim elimination, cascade transition smoothing)
- How to enable (`shadow_backend="paged"`)
- Configuration knobs (`shadow_page_tile_size`, `shadow_page_budget_mb`)
- Performance characteristics
- Debug modes (`SHADOW_DEBUG_PAGES`)
- Queryable stats
- Known limitations (v1: CPU requests only, no moment-based techniques)

- [ ] **Step 4: Commit**

```bash
git add examples/terrain_tv11_paged_shadow_demo.py docs/terrain/tv11-page-based-terrain-shadowing.md
git commit -m "docs(tv11.5): add paged shadow demo example and feature documentation"
```

---

## Task 10: Final Verification and Cleanup

- [ ] **Step 1: Run full test suite**

Run: `cd C:/Users/milos/forge3d && python -m pytest tests/ -v --timeout=120`
Expected: All tests PASS.

- [ ] **Step 2: Run Rust tests**

Run: `cd C:/Users/milos/forge3d && cargo test -- --nocapture`
Expected: All tests PASS.

- [ ] **Step 3: Run example with real DEM assets**

Run: `cd C:/Users/milos/forge3d && python examples/terrain_tv11_paged_shadow_demo.py --dem assets/tif/dem_rainier.tif`
Expected: Output PNGs show valid terrain shadows with both backends.

- [ ] **Step 4: Verify image output quality**

Manually inspect:
- CSM vs paged comparison: shadows should be equivalent
- Paged debug pages visualization: page grid visible
- No shadow shimmer in paged output

- [ ] **Step 5: Final commit with any cleanup**

```bash
git add src/terrain/ src/shaders/ src/core/cascade_split.rs python/forge3d/terrain_params.py tests/test_terrain_tv11_paged_shadows.py examples/terrain_tv11_paged_shadow_demo.py docs/terrain/tv11-page-based-terrain-shadowing.md
git commit -m "chore(tv11): final cleanup and verification"
```
