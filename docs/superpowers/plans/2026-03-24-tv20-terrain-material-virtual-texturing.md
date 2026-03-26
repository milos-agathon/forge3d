# TV20 Terrain Material Virtual Texturing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the monolithic resident material texture array with VT-backed terrain material layer sampling, so large terrains can stream material tiles on demand via LRU-managed atlas paging.

**Architecture:** The existing generic VT core (`src/core/virtual_texture/`) gets prerequisite fixes (mip convention, per-mip page table, border support, array layers, PageTableEntry shape). A new `TerrainMaterialVT` module wraps it for terrain-specific tile loading. Python API adds `TerrainVTSettings` on `TerrainRenderParams` and source registration on the renderer. The shader gains a VT sampling path in `@group(6)` gated by a uniform flag.

**Tech Stack:** Rust (wgpu), Python (pyo3/numpy), WGSL shaders

**Spec:** `docs/superpowers/specs/2026-03-23-tv20-terrain-material-virtual-texturing-design.md`

---

## File Structure

### New files
| File | Responsibility |
|---|---|
| `src/terrain/render_params/native_vt.rs` | `TerrainVTSettingsNative`, `VTLayerFamilyNative` structs |
| `src/terrain/render_params/decode_vt.rs` | `parse_vt_settings()` — Python→Rust VT config decode |
| `src/terrain/renderer/virtual_texture.rs` | `TerrainMaterialVT` — wraps VT core for terrain materials, tile extraction, source storage |
| `tests/test_tv20_virtual_texturing.py` | All 9+ regression tests |
| `examples/terrain_tv20_virtual_texturing_demo.py` | End-to-end demo with real DEM |

### Modified files
| File | Changes |
|---|---|
| `python/forge3d/terrain_params.py` | Add `VTLayerFamily`, `TerrainVTSettings`, `vt` field on `TerrainRenderParams` |
| `python/forge3d/__init__.py` | Re-export `VTLayerFamily`, `TerrainVTSettings` |
| `python/forge3d/__init__.pyi` | Type stubs for new classes |
| `src/core/tile_cache/types.rs` | Flip `TileId::parent()`/`children()` mip convention |
| `src/core/tile_cache/allocator.rs` | Accept `slot_size` for grid layout |
| `src/core/virtual_texture/types.rs` | `PageTableEntry` shape → `{atlas_u, atlas_v, scale_u, scale_v}`; `VirtualTextureConfig` gains `tile_border`, `material_count` |
| `src/core/virtual_texture/mod.rs` | `page_table_data: Vec<Vec<PageTableEntry>>`, new struct field |
| `src/core/virtual_texture/constructor.rs` | Array-layer atlas, per-mip page table init, border/slot_size |
| `src/core/virtual_texture/update.rs` | Mip convention flip, per-mip page table write, ancestor fallback |
| `src/core/virtual_texture/upload.rs` | Border fill, array layer upload, tile_id_to_page_index mip-aware |
| `src/terrain/render_params.rs` | Add `mod native_vt; mod decode_vt;` and `use` for new native types |
| `src/terrain/render_params/core.rs` | Add `vt` field to `DecodedTerrainSettings` |
| `src/terrain/render_params/private_impl.rs` | Call `parse_vt_settings()` |
| `src/terrain/renderer.rs` | Add `mod virtual_texture;` |
| `src/terrain/renderer/py_api.rs` | `register_material_vt_source()`, `clear_material_vt_sources()`, `get_material_vt_stats()` |
| `src/terrain/renderer/core.rs` | Add VT fallback resources + `TerrainMaterialVT` to `TerrainScene` |
| `src/terrain/renderer/bind_groups/layouts.rs` | Extend @group(6) layout with bindings 3-7 |
| `src/terrain/renderer/bind_groups/terrain_pass.rs` | Create VT bind entries + fallback resources |
| `src/terrain/renderer/constructor.rs` | Init VT fallback textures/buffers |
| `src/terrain/renderer/offline.rs` | Per-frame VT update guard |
| `src/shaders/terrain_pbr_pom.wgsl` | VT uniforms, `textureLoad` page lookup, `vt_sample_axis()`, dual-path `sample_triplanar()` |

**Note — `set_bind_group(6)` call sites audited:** The calls in `draw/execute.rs`, `aov.rs`, `offline.rs`, and `water_reflection/bind_group.rs` all pass `material_layer_bind_group` from `TerrainPassBindGroups.material_layer`, which is created centrally in `create_terrain_pass_bind_groups()`. These callers do NOT need code changes — the layout expansion and bind group creation changes happen in `layouts.rs` and `terrain_pass.rs` only.

---

## Task 1: Python API — VTLayerFamily and TerrainVTSettings

**Files:**
- Modify: `python/forge3d/terrain_params.py` (add after `OfflineQualitySettings` at ~line 928)
- Modify: `python/forge3d/__init__.py` (add to imports)
- Modify: `python/forge3d/__init__.pyi` (add type stubs)
- Test: `tests/test_tv20_virtual_texturing.py`

- [ ] **Step 1: Write pure-Python validation tests**

Create `tests/test_tv20_virtual_texturing.py` with tests for the dataclass contract:

```python
"""TV20: Terrain material virtual texturing."""
from __future__ import annotations
import pytest

# --- Pure-Python tests (no native module needed) ---

def test_vt_layer_family_defaults():
    from forge3d.terrain_params import VTLayerFamily
    f = VTLayerFamily(family="albedo")
    assert f.tile_size == 248
    assert f.tile_border == 4
    assert f.slot_size == 256
    assert f.pages_x0 == 17  # ceil(4096/248)
    assert f.pages_y0 == 17

def test_vt_layer_family_rejects_unknown():
    from forge3d.terrain_params import VTLayerFamily
    with pytest.raises(ValueError, match="family must be one of"):
        VTLayerFamily(family="diffuse")

def test_vt_layer_family_accepts_normal_mask():
    from forge3d.terrain_params import VTLayerFamily
    VTLayerFamily(family="normal")
    VTLayerFamily(family="mask")

def test_vt_settings_rejects_duplicate_family():
    from forge3d.terrain_params import VTLayerFamily, TerrainVTSettings
    with pytest.raises(ValueError, match="duplicate"):
        TerrainVTSettings(layers=[
            VTLayerFamily(family="albedo"),
            VTLayerFamily(family="albedo"),
        ])

def test_vt_settings_rejects_indivisible_atlas():
    from forge3d.terrain_params import VTLayerFamily, TerrainVTSettings
    with pytest.raises(ValueError, match="divisible"):
        TerrainVTSettings(atlas_size=1000, layers=[
            VTLayerFamily(family="albedo", tile_size=248, tile_border=4),
        ])

def test_vt_settings_actual_mip_count():
    from forge3d.terrain_params import TerrainVTSettings
    s = TerrainVTSettings(max_mip_levels=8)
    assert s.actual_mip_count("albedo") >= 1

def test_vt_layer_pages_at_mip_non_pot():
    from forge3d.terrain_params import VTLayerFamily
    f = VTLayerFamily(family="albedo", virtual_size_px=(3000, 2000), tile_size=248)
    px, py = f.pages_at_mip(0)
    assert px == 13  # ceil(3000/248)
    assert py == 9   # ceil(2000/248)
    px1, py1 = f.pages_at_mip(1)
    assert px1 == 7  # ceil(13/2)
    assert py1 == 5  # ceil(9/2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tv20_virtual_texturing.py -v -k "not runtime" 2>&1 | head -30`
Expected: FAIL — `VTLayerFamily` not defined

- [ ] **Step 3: Implement VTLayerFamily, TerrainVTSettings in terrain_params.py**

Add before `class TerrainRenderParams` (line 1406). Insert `VTLayerFamily` and `TerrainVTSettings` dataclasses at end of settings section, after the last settings class (e.g. after `SkySettings`). Then add `vt: Optional[TerrainVTSettings] = None` to `TerrainRenderParams`. Also add both classes to the `__all__` list at the bottom of the file. See spec Section 3.2 for exact code.

- [ ] **Step 4: Update __init__.py and __init__.pyi exports**

Add `VTLayerFamily` and `TerrainVTSettings` to the import/export lists.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tv20_virtual_texturing.py -v -k "not runtime" 2>&1 | head -30`
Expected: All 7 pure-Python tests PASS

- [ ] **Step 6: Commit**

```bash
git add python/forge3d/terrain_params.py python/forge3d/__init__.py python/forge3d/__init__.pyi tests/test_tv20_virtual_texturing.py
git commit -m "feat(tv20): add VTLayerFamily, TerrainVTSettings Python dataclasses"
```

---

## Task 2: VT Core — PageTableEntry Shape and Config Changes

**Files:**
- Modify: `src/core/virtual_texture/types.rs`
- Modify: `src/core/tile_cache/types.rs`
- Modify: `src/core/tile_cache/allocator.rs`
- Test: `src/core/virtual_texture/tests.rs`

- [ ] **Step 1: Update PageTableEntry in types.rs**

Replace the existing struct (spec Section 5.4):

```rust
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct PageTableEntry {
    pub atlas_u: f32,
    pub atlas_v: f32,
    pub scale_u: f32,  // 0.0 = nothing resident
    pub scale_v: f32,
}
```

Add `tile_border` and `material_count` to `VirtualTextureConfig`:

```rust
pub struct VirtualTextureConfig {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,       // content pixels
    pub tile_border: u32,     // gutter pixels per edge
    pub max_mip_levels: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub format: wgpu::TextureFormat,
    pub use_feedback: bool,
    pub material_count: u32,  // array layer count
}
```

Update `Default` impl: `tile_border: 4`, `material_count: 1`.

- [ ] **Step 2: Flip TileId mip convention**

In `src/core/tile_cache/types.rs`, flip `parent()` to go mip+1 and `children()` to go mip-1 (spec Section 5.5):

```rust
pub fn parent(&self) -> Option<Self> {
    Some(Self {
        x: self.x / 2,
        y: self.y / 2,
        mip_level: self.mip_level + 1,
    })
    // Note: caller must check mip_level < max_mip_levels
}

pub fn children(&self) -> Option<[Self; 4]> {
    if self.mip_level == 0 { return None; }
    let child_mip = self.mip_level - 1;
    let child_x = self.x * 2;
    let child_y = self.y * 2;
    Some([
        Self { x: child_x,     y: child_y,     mip_level: child_mip },
        Self { x: child_x + 1, y: child_y,     mip_level: child_mip },
        Self { x: child_x,     y: child_y + 1, mip_level: child_mip },
        Self { x: child_x + 1, y: child_y + 1, mip_level: child_mip },
    ])
}
```

- [ ] **Step 3: Update AtlasSlot and AtlasAllocator**

In `src/core/tile_cache/types.rs`, update `AtlasSlot` to remove `mip_bias` (no longer needed — shader selects mip explicitly). The struct becomes:

```rust
pub struct AtlasSlot {
    pub atlas_x: u32,
    pub atlas_y: u32,
    pub atlas_u: f32,  // slot origin UV (caller offsets by border for content)
    pub atlas_v: f32,
}
```

In `src/core/tile_cache/allocator.rs`, change `new_with_dimensions` to accept `slot_size` (= `tile_size + 2 * tile_border`). The grid layout uses `slot_size`, not content `tile_size`. UV calculations use `slot_size` for slot origin. Update `allocate()` and `deallocate()` to use the new `AtlasSlot` shape (no `mip_bias`).

- [ ] **Step 4: Run existing VT core tests**

Run: `cargo test --lib virtual_texture -- --nocapture 2>&1 | tail -20`
Expected: Tests compile and pass (fix any broken tests from shape change)

- [ ] **Step 5: Commit**

```bash
git add src/core/virtual_texture/types.rs src/core/tile_cache/types.rs src/core/tile_cache/allocator.rs
git commit -m "feat(tv20): update PageTableEntry shape, VT config, mip convention"
```

---

## Task 3: VT Core — Per-Mip Page Table and Constructor

**Files:**
- Modify: `src/core/virtual_texture/mod.rs`
- Modify: `src/core/virtual_texture/constructor.rs`
- Modify: `src/core/virtual_texture/upload.rs`
- Modify: `src/core/virtual_texture/update.rs`
- Test: `src/core/virtual_texture/tests.rs`

- [ ] **Step 1: Update VirtualTexture struct for per-mip page table**

In `mod.rs`, change `page_table_data` from `Vec<PageTableEntry>` to `Vec<Vec<PageTableEntry>>`.

- [ ] **Step 2: Rewrite constructor for per-mip init, array atlas, and border support**

In `constructor.rs`:
- Atlas texture becomes `texture_2d_array` with `depth_or_array_layers = config.material_count`
- Compute `slot_size = config.tile_size + 2 * config.tile_border`
- Atlas allocator uses `slot_size`
- Page table data is `Vec<Vec<PageTableEntry>>` with one vec per mip level
- Each mip level L has `ceil_div(pages_x0, 1<<L).max(1) * ceil_div(pages_y0, 1<<L).max(1)` entries
- `actual_mip_count = min(config.max_mip_levels, floor(log2(max(pages_x0, pages_y0))) + 1)`

Add helper:
```rust
fn ceil_div(a: u32, b: u32) -> u32 { (a + b - 1) / b }
```

- [ ] **Step 3: Make tile_id_to_page_index mip-aware**

In `upload.rs`, update the function to index into the correct mip level's page table:

```rust
pub(super) fn page_index_at_mip(&self, tile_id: &TileId) -> (usize, usize) {
    let mip = tile_id.mip_level as usize;
    let divisor = 1u32 << tile_id.mip_level;
    let mip_pages_x = ceil_div(self.pages_x0(), divisor).max(1);
    let index = (tile_id.y * mip_pages_x + tile_id.x) as usize;
    (mip, index)
}
```

- [ ] **Step 4: Rewrite update_page_table for per-mip upload**

In `update.rs`, write each mip level to the GPU page table texture separately (spec Section 5.3).

- [ ] **Step 5: Invert distance→mip in calculate_visible_tiles**

Near tiles get mip 0 (finest), far tiles get high mip (coarsest):

```rust
let mip_level = ((distance / visible_radius as f32)
    * (self.actual_mip_count() - 1) as f32) as u32;
// mip 0 = near = finest, mip N = far = coarsest
```

- [ ] **Step 6: Add ancestor fallback population**

In `update.rs`, after all tile loads, populate non-resident entries with ancestor sub-rect transforms. Walk `TileId::parent()` chain until a resident ancestor is found, compute `atlas_u/v` and `scale_u/v` from the quadrant path (spec Section 5.4 entry population rules).

- [ ] **Step 7: Upload coarsest-level tiles at init**

In `constructor.rs` or a new `init_coarsest_level()` method: upload all tiles at the coarsest mip level at VT creation time (spec Section 5.6). For now, fill with procedural/fallback data.

- [ ] **Step 8: Run VT core tests**

Run: `cargo test --lib virtual_texture -- --nocapture 2>&1 | tail -20`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/core/virtual_texture/
git commit -m "feat(tv20): per-mip page table, array atlas, ancestor fallback, border support"
```

---

## Task 4: Rust Decode — Native VT Settings and Parse

**Files:**
- Create: `src/terrain/render_params/native_vt.rs`
- Create: `src/terrain/render_params/decode_vt.rs`
- Modify: `src/terrain/render_params/core.rs`
- Modify: `src/terrain/render_params/private_impl.rs`

- [ ] **Step 1: Create native_vt.rs**

```rust
#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct VTLayerFamilyNative {
    pub family: String,
    pub virtual_size: (u32, u32),
    pub tile_size: u32,
    pub tile_border: u32,
    pub fallback: [f32; 4],
}

#[cfg(feature = "extension-module")]
#[derive(Clone)]
pub struct TerrainVTSettingsNative {
    pub enabled: bool,
    pub layers: Vec<VTLayerFamilyNative>,
    pub atlas_size: u32,
    pub residency_budget_mb: f32,
    pub max_mip_levels: u32,
    pub use_feedback: bool,
}

#[cfg(feature = "extension-module")]
impl Default for TerrainVTSettingsNative {
    fn default() -> Self {
        Self { enabled: false, layers: vec![], atlas_size: 4096,
               residency_budget_mb: 256.0, max_mip_levels: 8, use_feedback: true }
    }
}
```

- [ ] **Step 2: Create decode_vt.rs**

Follow the pattern from `decode_effects.rs`. Extract VT settings from `params.vt`:

```rust
pub(super) fn parse_vt_settings(params: &Bound<'_, PyAny>) -> TerrainVTSettingsNative {
    if let Ok(vt) = params.getattr("vt") {
        if vt.is_none() { return TerrainVTSettingsNative::default(); }
        let enabled: bool = vt.getattr("enabled").and_then(|v| v.extract()).unwrap_or(false);
        if !enabled { return TerrainVTSettingsNative::default(); }
        // ... extract layers, atlas_size, etc.
        // Reject non-albedo families with NotImplementedError
    } else {
        TerrainVTSettingsNative::default()
    }
}
```

- [ ] **Step 3: Register new modules in `src/terrain/render_params.rs`**

Add module declarations alongside existing ones (after `mod decode_probes;`):

```rust
mod decode_vt;
mod native_vt;
```

Add `use` statement alongside existing native type imports:

```rust
use native_vt::{TerrainVTSettingsNative, VTLayerFamilyNative};
```

- [ ] **Step 4: Add `vt` field to DecodedTerrainSettings**

In `core.rs`, add `pub vt: TerrainVTSettingsNative` to `DecodedTerrainSettings`.

- [ ] **Step 5: Wire parse_vt_settings into private_impl.rs**

In `from_python_params()`, add `vt: parse_vt_settings(&params)` to the `DecodedTerrainSettings` struct literal.

- [ ] **Step 6: Verify compilation**

Run: `cargo build --features extension-module 2>&1 | tail -20`
Expected: Compiles

- [ ] **Step 7: Commit**

```bash
git add src/terrain/render_params.rs src/terrain/render_params/
git commit -m "feat(tv20): add native VT settings decode path"
```

---

## Task 5: Bind Group Layout Extension and VT Fallback Resources

**Files:**
- Modify: `src/terrain/renderer/bind_groups/layouts.rs`
- Modify: `src/terrain/renderer/core.rs`
- Modify: `src/terrain/renderer/constructor.rs`
- Modify: `src/terrain/renderer/bind_groups/terrain_pass.rs`

- [ ] **Step 1: Extend @group(6) layout with VT bindings 3-7**

In `create_material_layer_bind_group_layout()` in `layouts.rs`, add 5 new entries:
- binding 3: uniform buffer (VTUniforms)
- binding 4: uniform buffer (VTFallbackColors)
- binding 5: texture_2d_array (atlas)
- binding 6: sampler (atlas sampler)
- binding 7: texture_2d (page table — for textureLoad, `NonFiltering` sample type)

- [ ] **Step 2: Add VT fallback resources to TerrainScene**

In `core.rs`, add fields:

```rust
pub(super) vt_uniform_buffer: wgpu::Buffer,
pub(super) vt_fallback_uniform_buffer: wgpu::Buffer,
pub(super) vt_atlas_fallback_texture: wgpu::Texture,
pub(super) vt_atlas_fallback_view: wgpu::TextureView,
pub(super) vt_page_table_fallback_texture: wgpu::Texture,
pub(super) vt_page_table_fallback_view: wgpu::TextureView,
pub(super) vt_atlas_sampler: wgpu::Sampler,
```

- [ ] **Step 3: Initialize fallback resources in constructor**

In `constructor.rs`, create:
- 1×1×1 `texture_2d_array` (Rgba8UnormSrgb, single-layer, 1px) for VT atlas fallback
- 1×1 `texture_2d` (Rgba32Float, single texel, zeros) for page table fallback
- VT uniform buffer (32 bytes, `enabled = 0`)
- VT fallback colors buffer (16 × 16 bytes = 256 bytes, zeros)
- Linear filtering sampler for atlas

- [ ] **Step 4: Update bind group creation in terrain_pass.rs**

In `create_terrain_pass_bind_groups()`, extend the `material_layer` bind group to include VT bindings. When VT is disabled (no active `TerrainMaterialVT`), bind the fallback resources. When VT is active, bind the real atlas/page_table textures.

- [ ] **Step 5: Verify compilation**

Run: `cargo build --features extension-module 2>&1 | tail -20`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add src/terrain/renderer/
git commit -m "feat(tv20): extend group(6) layout with VT bindings and fallback resources"
```

---

## Task 6: Shader — VT Sampling Path

**Files:**
- Modify: `src/shaders/terrain_pbr_pom.wgsl`

- [ ] **Step 1: Add VT uniform structs and bindings**

After the existing `@group(6) @binding(2)` probe SSBO, add spec Section 6.1 declarations: `VTUniforms`, `VTFallbackColors`, atlas texture, atlas sampler, page table texture.

- [ ] **Step 2: Add VT helper functions**

Add `vt_ceil_div()`, `vt_pages_at_mip()`, `vt_page_lookup()` (spec Section 6.2).

- [ ] **Step 3: Add vt_sample_axis()**

Add the per-axis VT sampling function (spec Section 6.3).

- [ ] **Step 4: Update sample_triplanar() with VT dual path**

Add the `if (u_vt.enabled == 1u)` branch with negative-safe `fract()` UV wrapping, float-first mip clamping, and triplanar VT sampling. The existing non-VT path stays unchanged in the `else` branch.

- [ ] **Step 5: Verify shader compilation**

Run: `cargo build --features extension-module 2>&1 | tail -20`
Expected: Compiles (shader is validated at pipeline creation time, but build will catch syntax errors in wgsl preprocessing)

- [ ] **Step 6: Commit**

```bash
git add src/shaders/terrain_pbr_pom.wgsl
git commit -m "feat(tv20): add VT sampling path to terrain shader with textureLoad page lookup"
```

---

## Task 7: TerrainMaterialVT — Source Registration and Tile Loading

**Files:**
- Create: `src/terrain/renderer/virtual_texture.rs`
- Modify: `src/terrain/renderer/core.rs`
- Modify: `src/terrain/renderer/py_api.rs`

- [ ] **Step 1: Register new module in `src/terrain/renderer.rs`**

Add after the existing `mod` declarations (e.g. after `mod water_reflection;`):

```rust
mod virtual_texture;
```

- [ ] **Step 2: Create TerrainMaterialVT struct**

```rust
pub(super) struct VTSource {
    pub material_index: u32,
    pub virtual_size: (u32, u32),
    pub data: Vec<u8>,
    pub fallback_color: [f32; 4],
}

pub(super) struct TerrainMaterialVT {
    pub albedo_vt: Option<VirtualTexture>,
    pub sources: HashMap<(u32, String), VTSource>,  // (material_index, family)
    pub fallback_colors: Vec<[f32; 4]>,  // per material_index
}
```

Implement:
- `register_source()` — stores source data, validates virtual_size consistency
- `clear_sources()` — drops all registered sources and VT instances
- `ensure_initialized()` — creates VirtualTexture from settings + sources if not yet initialized
- `update()` — calls `VirtualTexture::update()`, called once per logical frame
- `stats()` — returns per-family stats dict
- `load_tile_for_sources()` — extracts tile region from source images for a given TileId, all materials

- [ ] **Step 3: Add TerrainMaterialVT to TerrainScene**

In `core.rs`, add `pub(super) material_vt: Mutex<TerrainMaterialVT>`.

- [ ] **Step 4: Add PyMethods for source registration**

In `py_api.rs`, add:
- `register_material_vt_source()` — validates family name, schema, virtual_size consistency; stores into `material_vt`
- `clear_material_vt_sources()` — clears stored sources
- `get_material_vt_stats()` — returns Python dict from `material_vt.stats()`

- [ ] **Step 5: Add render-time VT validation**

In the render path (before bind group creation), check: if decoded VT settings are enabled, validate all material indices have registered sources. If not → `PyRuntimeError`.

- [ ] **Step 6: Wire VT update into render path**

Before `create_terrain_pass_bind_groups()`, if VT is enabled:
1. Call `material_vt.ensure_initialized(device, queue, decoded.vt, material_count)`
2. Call `material_vt.update(device, queue, camera_info)`
3. Pass real atlas/page_table views to bind group creation instead of fallbacks

- [ ] **Step 7: Verify compilation**

Run: `cargo build --features extension-module 2>&1 | tail -20`
Expected: Compiles

- [ ] **Step 8: Commit**

```bash
git add src/terrain/renderer.rs src/terrain/renderer/virtual_texture.rs src/terrain/renderer/core.rs src/terrain/renderer/py_api.rs
git commit -m "feat(tv20): TerrainMaterialVT with source registration, tile loading, render-time validation"
```

---

## Task 8: Offline VT Update Guard

**Files:**
- Modify: `src/terrain/renderer/offline.rs`

- [ ] **Step 1: Add per-frame VT update guard**

In the offline accumulation render loop, call `material_vt.update()` only on the first sample of each logical frame, not on every accumulation sample. Check if VT is enabled in decoded settings; if not, skip entirely.

- [ ] **Step 2: Verify compilation**

Run: `cargo build --features extension-module 2>&1 | tail -20`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add src/terrain/renderer/offline.rs
git commit -m "feat(tv20): per-logical-frame VT update guard for offline accumulation"
```

---

## Task 9: Integration Tests — Full Render Pipeline

**Files:**
- Modify: `tests/test_tv20_virtual_texturing.py`

- [ ] **Step 1: Add runtime test fixtures and helpers**

```python
from _terrain_runtime import terrain_rendering_available

try:
    import forge3d as f3d
    from forge3d.terrain_params import (
        PomSettings, make_terrain_params_config,
        VTLayerFamily, TerrainVTSettings,
    )
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False

needs_runtime = pytest.mark.skipif(
    not HAS_NATIVE or not terrain_rendering_available(),
    reason="requires terrain-capable forge3d runtime",
)

@pytest.fixture()
def runtime(tmp_path):
    hdr_path = tmp_path / "env.hdr"
    _write_test_hdr(hdr_path)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(hdr_path), intensity=1.0)
    return renderer, material_set, ibl
```

- [ ] **Step 2: Write test_vt_disabled_matches_baseline**

Render with `vt=None` and with `vt=TerrainVTSettings(enabled=False)`. Assert outputs are identical.

- [ ] **Step 3: Write test_vt_enabled_renders_valid_image**

Register albedo sources for all material indices, render with VT enabled. Assert: image is not all-black, has nonzero pixel variance.

- [ ] **Step 4: Write test_vt_stats_report_residency**

After VT render, call `get_material_vt_stats()`. Assert keys exist, `resident_pages > 0`, `miss_rate` in [0,1].

- [ ] **Step 5: Write test_budget_enforcement_evicts_lru**

Use tiny `atlas_size` + `residency_budget_mb=0.1` with large virtual_size. Render. Assert no crash, stats show eviction (resident_pages < total_pages).

- [ ] **Step 6: Write test_normal_mask_family_reserved**

Create `TerrainVTSettings(layers=[VTLayerFamily("normal")])`, render. Assert `NotImplementedError` from native decode.

- [ ] **Step 7: Write test_clear_sources_resets_state**

Register sources, clear, render with VT enabled. Assert `PyRuntimeError` (missing sources).

- [ ] **Step 8: Write test_mixed_virtual_size_rejects**

Register source with `virtual_size_px=(4096,4096)`, then register another with `(2048,2048)` for same family. Assert `ValueError`.

- [ ] **Step 9: Write test_missing_material_index_rejects**

Enable VT, register only material_index=0, render with material set that has 2+ layers. Assert `PyRuntimeError` naming the missing index.

- [ ] **Step 10: Write test_non_pot_virtual_size_renders**

Use `virtual_size_px=(3000, 2000)`. Register sources, render. Assert valid image.

- [ ] **Step 11: Run all tests**

Run: `python -m pytest tests/test_tv20_virtual_texturing.py -v 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 12: Commit**

```bash
git add tests/test_tv20_virtual_texturing.py
git commit -m "test(tv20): add 9 regression tests for terrain material virtual texturing"
```

---

## Task 10: Example — terrain_tv20_virtual_texturing_demo.py

**Files:**
- Create: `examples/terrain_tv20_virtual_texturing_demo.py`

- [ ] **Step 1: Write the demo script**

Follow the TV12 example pattern. Load a real DEM (default: `assets/tif/dem_rainier.tif`), derive per-material albedo sources from elevation bands (color by height), register VT sources, render with VT enabled, report stats, save output.

Key structure:
```python
def main():
    args = parse_args()
    f3d = _import_forge3d()
    dem, heightmap = _load_dem(args.dem)
    session = f3d.Session(window=False)
    renderer = f3d.TerrainRenderer(session)
    material_set = f3d.MaterialSet.terrain_default()
    ibl = f3d.IBL.from_hdr(str(HDR_PATH), intensity=1.0)

    # Generate per-material albedo sources from heightmap
    for idx in range(material_set.layer_count):
        source = _generate_material_albedo(heightmap, idx, args.virtual_size)
        renderer.register_material_vt_source(
            material_index=idx, family="albedo",
            image_or_pyramid=source,
            virtual_size_px=(args.virtual_size, args.virtual_size),
        )

    vt = f3d.TerrainVTSettings(enabled=True, atlas_size=args.atlas_size)
    params = _make_params(size_px=(args.width, args.height), vt=vt)
    frame = renderer.render_terrain_pbr_pom(material_set, ibl, params, heightmap)

    stats = renderer.get_material_vt_stats()
    _print_stats(stats)
    _save_output(frame, args.output)
```

- [ ] **Step 2: Test the example runs**

Run: `python examples/terrain_tv20_virtual_texturing_demo.py --output examples/out/tv20_demo.png 2>&1 | tail -10`
Expected: Produces a PNG and prints stats

- [ ] **Step 3: Commit**

```bash
git add examples/terrain_tv20_virtual_texturing_demo.py
git commit -m "feat(tv20): add virtual texturing demo example with real DEM"
```

---

## Task 11: Documentation

**Files:**
- Create: `docs/tv20-terrain-material-virtual-texturing.md`

- [ ] **Step 1: Write feature documentation**

Document:
- What TV20 achieves (VT-backed terrain material paging)
- Python API usage (TerrainVTSettings, register_material_vt_source, stats)
- How to use VT in existing terrain pipelines
- Budget tuning guidance
- Limitations (v1 albedo only, normal/mask reserved)

- [ ] **Step 2: Commit**

```bash
git add docs/tv20-terrain-material-virtual-texturing.md
git commit -m "docs(tv20): document terrain material virtual texturing features"
```

---

## Task 12: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --timeout=120 2>&1 | tail -30`
Expected: All tests PASS, no regressions

- [ ] **Step 2: Run the example and verify image output**

Run: `python examples/terrain_tv20_virtual_texturing_demo.py --output examples/out/tv20_final.png`
Expected: Valid PNG output, stats printed

- [ ] **Step 3: Final commit with any fixups**

Only if needed — fix any test failures or compilation issues discovered in full-suite run.
