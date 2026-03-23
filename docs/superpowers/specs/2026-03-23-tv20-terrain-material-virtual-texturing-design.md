# TV20 — Terrain Material Virtual Texturing Design

**Date:** 2026-03-23
**Epic:** TV20 (from `docs/plans/2026-03-16-terrain-viz-epics.md`)
**Scope:** VT-backed terrain material layer sampling, replacing the monolithic resident material textures in the terrain path. Procedural snow/rock/wetness blending is unchanged. RVT-style runtime compositing is explicitly out of scope.

---

## 1. Problem Statement

Large terrains can stream geometry and elevation data well, but terrain material delivery still behaves like a smaller-scene texture workflow. The current path uploads all material albedo layers as a fully-resident `texture_2d_array` (`Rgba8UnormSrgb`, `gpu.rs:179-192`). When the combined material set exceeds GPU memory, the only option today is to reduce resolution or layer count.

Top engines solve this with terrain-oriented virtual texturing: only the material tiles near the camera are resident, the rest stream on demand. Forge3D already has generic VT foundations in `src/core/virtual_texture/` (atlas, page table, LRU tile cache, GPU feedback buffer, stats), but these are not connected to the terrain material path.

---

## 2. Scope

### v1 (this epic)

- VT-backed **albedo** material layer sampling.
- Layer descriptor contract that accepts **albedo + normal + optional mask** families from day one.
- `normal` and `mask` families accepted in the contract but raise `NotImplementedError` at decode time.

### Not in v1

- RVT-style runtime composited cache.
- Normal or mask layer paging (reserved for v2).
- Changes to procedural snow/rock/wetness blending.

---

## 3. Python API

### 3.1 Settings on `TerrainRenderParams`

```python
# terrain_params.py — added to TerrainRenderParams
vt: Optional[TerrainVTSettings] = None   # None = VT disabled, existing path unchanged
```

Follows the `Optional[...]` pattern used by `fog`, `reflection`, `detail`, `probes`, etc. (`terrain_params.py:1444-1456`). When `vt is None` or `vt.enabled is False`, the entire VT path is skipped and the existing monolithic `material_albedo_tex` array path runs unchanged.

### 3.2 `TerrainVTSettings` and `VTLayerFamily`

```python
@dataclass
class VTLayerFamily:
    """Describes one paged terrain material family."""
    family: str                        # "albedo" | "normal" | "mask"
    virtual_size_px: Tuple[int, int] = (4096, 4096)  # family-wide invariant
    tile_size: int = 256               # content pixels per tile edge
    tile_border: int = 4               # gutter pixels per tile edge
    fallback: Tuple[float, ...] = (0.5, 0.5, 0.5, 1.0)  # last-resort per-family fallback

    def __post_init__(self) -> None:
        VALID_FAMILIES = {"albedo", "normal", "mask"}
        if self.family not in VALID_FAMILIES:
            raise ValueError(f"family must be one of {VALID_FAMILIES}")
        if self.family != "albedo":
            raise NotImplementedError(f"family '{self.family}' reserved for v2")
        if self.tile_size < 16:
            raise ValueError("tile_size must be >= 16")
        if self.tile_border < 0:
            raise ValueError("tile_border must be >= 0")
        w, h = self.virtual_size_px
        if w < self.tile_size or h < self.tile_size:
            raise ValueError("virtual_size_px must be >= tile_size in both dimensions")

    @property
    def slot_size(self) -> int:
        """Physical atlas slot size = content + 2 * border."""
        return self.tile_size + 2 * self.tile_border

    @property
    def pages_x0(self) -> int:
        """Finest-level page count X."""
        import math
        return math.ceil(self.virtual_size_px[0] / self.tile_size)

    @property
    def pages_y0(self) -> int:
        """Finest-level page count Y."""
        import math
        return math.ceil(self.virtual_size_px[1] / self.tile_size)

    @property
    def full_pyramid_levels(self) -> int:
        """Maximum mip levels the virtual extent can support.
        Derived from finest page counts, not raw pixel ratio."""
        import math
        return int(math.floor(math.log2(max(self.pages_x0, self.pages_y0)))) + 1

    def pages_at_mip(self, mip: int) -> Tuple[int, int]:
        """Page count at a given mip level (ceil-div, min 1)."""
        import math
        return (
            max(1, math.ceil(self.pages_x0 / (2 ** mip))),
            max(1, math.ceil(self.pages_y0 / (2 ** mip))),
        )

@dataclass
class TerrainVTSettings:
    """Terrain material virtual texturing configuration."""
    enabled: bool = False
    layers: List[VTLayerFamily] = field(default_factory=lambda: [
        VTLayerFamily(family="albedo")
    ])
    atlas_size: int = 4096
    residency_budget_mb: float = 256.0
    max_mip_levels: int = 8
    use_feedback: bool = True

    def __post_init__(self) -> None:
        families = [l.family for l in self.layers]
        if len(families) != len(set(families)):
            raise ValueError("duplicate family in layers")
        if self.atlas_size < 256:
            raise ValueError("atlas_size must be >= 256")
        if self.residency_budget_mb <= 0:
            raise ValueError("residency_budget_mb must be > 0")
        if self.max_mip_levels < 1:
            raise ValueError("max_mip_levels must be >= 1")
        for layer in self.layers:
            if self.atlas_size % layer.slot_size != 0:
                raise ValueError(
                    f"atlas_size ({self.atlas_size}) must be divisible by "
                    f"slot_size ({layer.slot_size}) for family '{layer.family}'"
                )

    def actual_mip_count(self, family: str = "albedo") -> int:
        """Effective mip count: min(requested, full pyramid levels)."""
        layer = next(l for l in self.layers if l.family == family)
        return min(self.max_mip_levels, layer.full_pyramid_levels)
```

### 3.3 Renderer Methods (persistent source registration + stats)

```python
# On TerrainRenderer
def register_material_vt_source(
    self,
    material_index: int,       # maps to existing material array layer index
    family: str,               # "albedo"
    image_or_pyramid,          # np.ndarray (H×W×C) or list[np.ndarray] for mip pyramid
    virtual_size_px: Tuple[int, int],  # must match family's virtual_size_px
    fallback_color: Tuple[float, ...] = None,  # per-material fallback; None → use family default
) -> None: ...

def clear_material_vt_sources(self) -> None: ...

def get_material_vt_stats(self) -> dict: ...
```

**Validation at registration:**
- `virtual_size_px` must match the family's `virtual_size_px`. Mismatch → `ValueError`.
- `family` must be an active family in the current `TerrainVTSettings`. Unknown → `ValueError`.

**Missing material_index handling:**
When VT is enabled and `render_terrain_pbr_pom()` begins, validation checks that every `material_index` referenced by the material set has a registered source for each active family. Missing index → `PyRuntimeError` naming the missing index. No silent auto-fill.

---

## 4. Tile Size Semantics

`tile_size` means **content pixels**. The physical atlas allocation per tile is `slot_size = tile_size + 2 * tile_border`.

| Term | Definition |
|---|---|
| `tile_size` | Content pixels per tile edge (what the shader samples) |
| `tile_border` | Gutter pixels per edge (absorbs bilinear/aniso bleed) |
| `slot_size` | `tile_size + 2 * tile_border` — physical atlas allocation |

The atlas allocator uses `slot_size` for grid layout. Tile upload writes `slot_size × slot_size` pixels: the inner `tile_size × tile_size` region is content, the outer border ring is filled with neighboring tile edge data (clamp-to-edge at virtual texture boundaries).

Page table UV entries point to the **content interior**: offset inward by `tile_border / atlas_size` from the slot origin. Gradient scaling in the shader uses `tile_size / atlas_size` (content, not slot), so filtering stays within the content region.

---

## 5. Rust Architecture

### 5.1 One VirtualTexture per family

Each active family gets its own `VirtualTexture` instance. v1: only albedo. The `VirtualTextureConfig` for the albedo family:

```rust
VirtualTextureConfig {
    width: family.virtual_size_px.0,
    height: family.virtual_size_px.1,
    tile_size: family.tile_size,          // content size
    tile_border: family.tile_border,      // NEW field
    max_mip_levels: actual_mip_count,     // min(requested, full_pyramid)
    atlas_width: settings.atlas_size,
    atlas_height: settings.atlas_size,
    format: Rgba8UnormSrgb,               // matches existing material format
    use_feedback: settings.use_feedback,
    material_count: registered_material_count,  // array layer count
}
```

### 5.2 Atlas as `texture_2d_array`

The atlas texture is a `texture_2d_array` with `depth_or_array_layers = material_count`. This mirrors the existing `material_albedo_tex` structure but with VT-managed residency. All material layers share the same tile layout — when tile (x, y, mip) is loaded, data for all material layers is uploaded to the corresponding array layer of the atlas slot.

### 5.3 Per-mip page table

**Page grid derivation** (all mip arithmetic uses ceil-div, not right-shift):

```text
pages_x0 = ceil(virtual_w / tile_size)     -- finest-level page count
pages_y0 = ceil(virtual_h / tile_size)
actual_mip_count = min(max_mip_levels, floor(log2(max(pages_x0, pages_y0))) + 1)
pages_at_mip(L) = (max(1, ceil(pages_x0 / 2^L)), max(1, ceil(pages_y0 / 2^L)))
```

Non-power-of-two virtual extents are first-class. Right-shift (`>>`) must not be used for mip page counts — it truncates and breaks the page table for non-POT sizes.

**Storage**: `page_table_data: Vec<Vec<PageTableEntry>>` — indexed `[mip_level][y * mip_pages_x + x]`.

Each mip level L stores `pages_at_mip(L).0 × pages_at_mip(L).1` entries.

**GPU texture**: `texture_2d` with `mip_level_count = actual_mip_count`. Finest mip (level 0) has `pages_x0 × pages_y0` texels. Each mip level is uploaded separately:

```rust
fn ceil_div(a: u32, b: u32) -> u32 { (a + b - 1) / b }

for mip in 0..actual_mip_count {
    let divisor = 1u32 << mip;
    let mip_pages_x = ceil_div(pages_x0, divisor).max(1);
    let mip_pages_y = ceil_div(pages_y0, divisor).max(1);
    queue.write_texture(
        ImageCopyTexture { texture: &page_table, mip_level: mip, .. },
        bytemuck::cast_slice(&page_table_data[mip as usize]),
        ImageDataLayout { bytes_per_row: Some(mip_pages_x * 16), .. },
        Extent3d { width: mip_pages_x, height: mip_pages_y, .. },
    );
}
```

### 5.4 PageTableEntry with sub-rect transform

```rust
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
#[repr(C)]
pub struct PageTableEntry {
    pub atlas_u: f32,    // content region start in atlas UV space
    pub atlas_v: f32,
    pub scale_u: f32,    // UV extent this tile maps to; 0.0 = nothing resident
    pub scale_v: f32,
}
```

**Entry population rules:**

| Case | `atlas_u/v` | `scale_u/v` |
|---|---|---|
| Native tile resident | `(slot_origin + border) / atlas_size` | `tile_size / atlas_size` |
| Parent fallback (d=1) | parent base + quadrant_offset × parent_scale × 0.5 | native_scale × 0.5 |
| Ancestor fallback (d levels) | ancestor base + accumulated sub-rect offset | native_scale × 2^(-d) |
| Nothing resident | 0.0 | 0.0 |

The CPU update walks `TileId::parent()` up the mip chain until a resident ancestor is found, then computes the sub-rect transform by accumulating the quadrant path at each level.

### 5.5 Mip convention

**Standard GPU convention:**
- **mip 0** = finest resolution (most tiles, near camera, largest page table level)
- **mip N** = coarsest resolution (fewest tiles, far camera, smallest page table level)

The existing `TileId::parent()`/`children()` and `calculate_visible_tiles()` use the opposite convention. **Prerequisite fix**: flip `TileId::parent()` to go mip+1 (coarser) and `children()` to go mip-1 (finer), and invert the distance→mip mapping in `calculate_visible_tiles()`.

### 5.6 Coarsest-level init guarantee

At VT initialization, **all tiles at the coarsest managed level** (mip = `actual_mip_count - 1`) are uploaded. The tile count at that level is derived from `pages_at_mip()`:

```text
pages_x0 = ceil(virtual_w / tile_size)
pages_y0 = ceil(virtual_h / tile_size)
actual_mip_count = min(max_mip_levels, floor(log2(max(pages_x0, pages_y0))) + 1)
coarsest_mip = actual_mip_count - 1
coarsest_pages_x = max(1, ceil(pages_x0 / 2^coarsest_mip))
coarsest_pages_y = max(1, ceil(pages_y0 / 2^coarsest_mip))
total_coarsest = coarsest_pages_x * coarsest_pages_y
```

This is often 1 tile but not always. The spec does **not** assume 1 tile. These tiles are populated from the lowest-resolution source data, or from per-material `fallback_color` if no source image covers that region. This ensures every page table entry always has a valid ancestor — the `scale == 0` / nothing-resident case only occurs before init completes.

`actual_mip_count` is exposed in stats so Python can reason about it.

### 5.7 Decode integration

`DecodedTerrainSettings` gains `vt: TerrainVTSettingsNative` parsed via `parse_vt_settings(&params)` in `private_impl.rs`, following the existing pattern for fog/detail/probes.

### 5.8 Tile data source

The existing `load_tile_data()` generates procedural tiles. For terrain materials, tile loading is replaced with a pluggable source that extracts tile regions from registered `VTSource` data:

```rust
struct VTSource {
    material_index: u32,
    virtual_size: (u32, u32),
    data: Vec<u8>,          // full RGBA image data
    fallback_color: [f32; 4],
}
```

When tile (x, y, mip) is requested, the loader extracts the corresponding pixel region from each registered material's source image, fills tile borders from neighboring edge data, and uploads to the atlas.

---

## 6. Shader Integration

### 6.1 @group(6) extension

Current @group(6) bindings (material layers + probes):
- binding 0: `material_layer_uniforms` (uniform buffer)
- binding 1: `probe_grid_uniforms` (uniform buffer)
- binding 2: `probe_ssbo` (storage buffer, read-only)

Added VT bindings:

```wgsl
struct VTUniforms {
    enabled: u32,
    pages_x0: u32,          // finest-level page count X (ceil-div)
    pages_y0: u32,          // finest-level page count Y (ceil-div)
    actual_mip_count: u32,
    content_scale: vec2<f32>,  // tile_size / atlas_size (gradient transform)
    material_count: u32,
    _pad: u32,
};

@group(6) @binding(3) var<uniform> u_vt : VTUniforms;
@group(6) @binding(4) var vt_atlas_tex : texture_2d_array<f32>;
@group(6) @binding(5) var vt_atlas_samp : sampler;   // filtering sampler for atlas only
@group(6) @binding(6) var vt_page_table_tex : texture_2d<f32>;  // NO sampler — textureLoad only
```

Pipeline layout stays at 7 groups (0-6). No group 7.

### 6.2 Page table lookup — `textureLoad`, not filtered sampling

The page table contains atlas sub-rect metadata. Filtered sampling would interpolate neighboring entries and corrupt transforms. The shader uses `textureLoad` with integer coordinates:

```wgsl
fn vt_ceil_div(a: u32, b: u32) -> u32 { return (a + b - 1u) / b; }

fn vt_pages_at_mip(mip: u32) -> vec2<u32> {
    let divisor = 1u << mip;
    return vec2<u32>(
        max(1u, vt_ceil_div(u_vt.pages_x0, divisor)),
        max(1u, vt_ceil_div(u_vt.pages_y0, divisor)),
    );
}

fn vt_page_lookup(virtual_uv: vec2<f32>, mip: u32) -> vec4<f32> {
    let pages_at_mip = vt_pages_at_mip(mip);
    let page_xy = min(
        vec2<u32>(virtual_uv * vec2<f32>(pages_at_mip)),
        pages_at_mip - vec2<u32>(1u)
    );
    return textureLoad(vt_page_table_tex, page_xy, i32(mip));
}
```

### 6.3 VT-aware triplanar sampling

`sample_triplanar()` gains a VT path gated by `u_vt.enabled`:

```wgsl
fn vt_sample_axis(
    uv: vec2<f32>,
    ddx_uv: vec2<f32>,
    ddy_uv: vec2<f32>,
    layer: i32,
    mip: u32,
) -> vec3<f32> {
    let entry = vt_page_lookup(uv, mip);
    if (entry.z <= 0.0) {
        // Nothing resident — use per-material fallback (from uniform array)
        return vt_fallback_colors[layer].rgb;
    }
    let pages_at_mip = vec2<f32>(vt_pages_at_mip(mip));  // ceil-div, min 1
    let tile_uv = fract(uv * pages_at_mip);
    let atlas_uv = entry.xy + tile_uv * entry.zw;
    // Gradient transform: virtual → atlas space
    let atlas_ddx = ddx_uv * entry.zw * pages_at_mip;
    let atlas_ddy = ddy_uv * entry.zw * pages_at_mip;
    return textureSampleGrad(vt_atlas_tex, vt_atlas_samp, atlas_uv, layer, atlas_ddx, atlas_ddy).rgb;
}

fn sample_triplanar(
    world_pos: vec3<f32>, normal: vec3<f32>,
    scale: f32, blend_sharpness: f32,
    layer: f32, _lod_bias: f32,
) -> vec3<f32> {
    let weights = compute_triplanar_weights(normal, blend_sharpness);
    let uv_x = world_pos.yz * scale;
    let uv_y = world_pos.xz * scale;
    let uv_z = world_pos.xy * scale;
    let dpdx_w = dpdx(world_pos) * scale;
    let dpdy_w = dpdy(world_pos) * scale;
    let layer_i = i32(layer);

    if (u_vt.enabled == 1u) {
        // Compute mip from UV derivatives
        let max_deriv = max(length(dpdx_w), length(dpdy_w));
        let mip = clamp(u32(log2(max_deriv * f32(u_vt.pages_x0))), 0u, u_vt.actual_mip_count - 1u);
        let cx = vt_sample_axis(uv_x, dpdx_w.yz, dpdy_w.yz, layer_i, mip);
        let cy = vt_sample_axis(uv_y, dpdx_w.xz, dpdy_w.xz, layer_i, mip);
        let cz = vt_sample_axis(uv_z, dpdx_w.xy, dpdy_w.xy, layer_i, mip);
        return cx * weights.x + cy * weights.y + cz * weights.z;
    }

    // Existing non-VT path unchanged
    let color_x = textureSampleGrad(material_albedo_tex, material_samp, uv_x, layer_i, dpdx_w.yz, dpdy_w.yz).rgb;
    let color_y = textureSampleGrad(material_albedo_tex, material_samp, uv_y, layer_i, dpdx_w.xz, dpdy_w.xz).rgb;
    let color_z = textureSampleGrad(material_albedo_tex, material_samp, uv_z, layer_i, dpdx_w.xy, dpdy_w.xy).rgb;
    return color_x * weights.x + color_y * weights.y + color_z * weights.z;
}
```

---

## 7. Stats and Budget Surface

```python
stats = renderer.get_material_vt_stats()
# {
#     "albedo": {
#         "total_pages": int,
#         "resident_pages": int,
#         "cache_hits": int,
#         "cache_misses": int,
#         "tiles_streamed": int,
#         "memory_usage_bytes": int,   # includes material_count multiplier
#         "miss_rate": float,
#         "budget_mb": float,
#         "budget_utilization": float,
#         "actual_mip_count": int,
#         "material_count": int,
#     }
# }
```

**Budget enforcement:**
- Atlas creation is capped by `residency_budget_mb`.
- Memory accounting: `resident_tiles × slot_size² × bytes_per_pixel × material_count`.
- The `material_count` multiplier is required because the atlas is `texture_2d_array`, not single-layer.
- If the atlas can hold fewer tiles than the scene needs, LRU eviction handles overflow.
- Budget violation logs a warning but does not crash.

---

## 8. VT Core Prerequisite Fixes

The following fixes to `src/core/virtual_texture/` are required before terrain integration:

### 8.1 Mip convention flip

`TileId::parent()` must go to `mip_level + 1` (coarser), `children()` to `mip_level - 1` (finer). `calculate_visible_tiles()` must assign low mip to near tiles and high mip to far tiles.

### 8.2 Per-mip page table storage

Replace flat `page_table_data: Vec<PageTableEntry>` with `Vec<Vec<PageTableEntry>>`. Update `tile_id_to_page_index()` to be mip-aware. Update `update_page_table()` to write each mip level separately. All mip page counts must use ceil-div arithmetic (`ceil(pages_x0 / 2^L)`, min 1), not right-shift, to support non-power-of-two virtual extents.

### 8.3 Ancestor fallback population

After tile load/evict, the page table update loop must populate non-resident entries with their nearest resident ancestor's sub-rect transform.

### 8.4 Tile border support

`VirtualTextureConfig` gains `tile_border: u32`. `AtlasAllocator` uses `slot_size` for grid layout. Upload path fills border pixels. Page table UVs point to content interior.

### 8.5 Array layer support

Atlas texture gains `depth_or_array_layers = material_count`. Upload path writes to the correct array layer per material index.

---

## 9. Test Plan

| # | Test | Validates |
|---|---|---|
| 1 | `test_vt_disabled_matches_baseline` | `vt=None` produces bit-identical output to current path |
| 2 | `test_vt_enabled_renders_valid_image` | VT on + registered albedo sources → valid image, no black tiles |
| 3 | `test_vt_stats_report_residency` | Stats dict has correct keys, `resident_pages > 0`, `miss_rate` in [0,1] |
| 4 | `test_budget_enforcement_evicts_lru` | Small budget + large virtual size → eviction happens, no crash |
| 5 | `test_normal_mask_family_reserved` | Declaring `normal`/`mask` family → `NotImplementedError` |
| 6 | `test_clear_sources_resets_state` | Clearing sources + rendering falls back cleanly |
| 7 | `test_mixed_virtual_size_rejects` | Registering source with mismatched `virtual_size_px` → `ValueError` |
| 8 | `test_missing_material_index_rejects` | VT enabled but material_index unregistered → `PyRuntimeError` naming the index |

---

## 10. Example

`examples/terrain_tv20_virtual_texturing_demo.py` — loads a real DEM, registers per-material albedo sources derived from the DEM's elevation bands, renders with VT enabled, reports stats and saves output. Uses the same DEM loading pattern as existing TV12/TV2 examples.

---

## 11. Files to Create or Modify

### New files
- `src/terrain/renderer/virtual_texture.rs` — `TerrainMaterialVT` struct and update logic
- `src/terrain/render_params/decode_vt.rs` — `parse_vt_settings()`
- `src/terrain/render_params/native_vt.rs` — `TerrainVTSettingsNative`, `VTLayerFamilyNative`
- `tests/test_tv20_virtual_texturing.py` — all 8 tests
- `examples/terrain_tv20_virtual_texturing_demo.py` — demo script

### Modified files
- `python/forge3d/terrain_params.py` — add `VTLayerFamily`, `TerrainVTSettings`, `vt` field on `TerrainRenderParams`
- `python/forge3d/__init__.py` — re-export new types
- `python/forge3d/__init__.pyi` — type stubs
- `src/terrain/render_params/private_impl.rs` — add `vt` to `DecodedTerrainSettings`
- `src/terrain/render_params/mod.rs` — module declarations
- `src/terrain/renderer/py_api.rs` — `register_material_vt_source()`, `clear_material_vt_sources()`, `get_material_vt_stats()`
- `src/terrain/renderer/bind_groups/layouts.rs` — extend @group(6) layout
- `src/terrain/renderer/bind_groups/terrain_pass.rs` — create VT bind entries
- `src/shaders/terrain_pbr_pom.wgsl` — VT uniforms, `textureLoad` page lookup, `vt_sample_axis()`, dual-path `sample_triplanar()`
- `src/core/virtual_texture/types.rs` — `PageTableEntry` shape change, `tile_border` in config
- `src/core/virtual_texture/constructor.rs` — array layer atlas, per-mip page table init, border support
- `src/core/virtual_texture/update.rs` — mip convention flip, per-mip page table write, ancestor fallback
- `src/core/virtual_texture/upload.rs` — border fill, array layer upload
- `src/core/tile_cache/types.rs` — `TileId::parent()`/`children()` convention flip
- `src/core/tile_cache/allocator.rs` — slot_size-based grid layout
