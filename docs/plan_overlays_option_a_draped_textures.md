# Option A — Draped Overlays as Textures Sampled in Terrain Shading

> **Goal**: Implement an overlay layer stack where rasters/images and rasterized vectors become overlay textures in terrain UV/DEM space, sampled in the terrain shader and blended into material inputs so overlays are **lit and shadowed**.

## 1. Scope and Non-Goals

### In Scope

- Raster/image overlays draped onto terrain surface (RGBA textures)
- Vector geometry (points/lines/polygons) rasterized to overlay textures on CPU/GPU
- Overlay texture(s) sampled in terrain fragment shader, blended into albedo before lighting
- Multiple overlay layers with deterministic stacking order
- Overlays affected by sun lighting, shadows (sun visibility), and AO
- CRS/extent alignment validation
- Opt-in feature flag (default off, preserves existing behavior)

### Non-Goals

- Real-time vector editing/animation
- Screen-space decals (these are unlit by design)
- 3D geometry overlays (e.g., building extrusions) — separate feature
- Overlay anti-aliasing beyond texture filtering (AA handled at rasterization time)

---

## 2. Architecture Overview (Data Flow)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER API (Python/Rust)                            │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Overlay Layer Stack (CPU)                            │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│   │  Layer 0    │  │  Layer 1    │  │  Layer N    │  ← OverlayLayer struct  │
│   │ (raster/    │  │ (raster/    │  │ (vector     │                         │
│   │  image)     │  │  image)     │  │  rasterized)│                         │
│   └─────────────┘  └─────────────┘  └─────────────┘                         │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │ Flatten to single RGBA texture (GPU blit passes)
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Composite Overlay Texture (GPU)                         │
│                      Format: Rgba8UnormSrgb or Rgba16Float                   │
│                      Size: terrain_dims × overlay_resolution_scale           │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │ Bound to terrain shader as @binding(5)
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Terrain PBR Fragment Shader (WGSL)                        │
│   1. Sample overlay texture at in.uv                                         │
│   2. Blend overlay.rgb with terrain albedo (multiply/alpha-blend)            │
│   3. Continue with normal lighting: sun, shadows, AO, IBL                    │
│   4. Overlays are LIT because they modify albedo, not final color            │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Integration Points (File-Path + Symbol Specific)

| Component | File | Symbols | Purpose |
|-----------|------|---------|---------|
| **Overlay layer struct** | `src/viewer/terrain/overlay.rs` (NEW) | `struct OverlayLayer`, `struct OverlayStack` | Hold overlay data, extents, blend modes |
| **Scene integration** | `src/viewer/terrain/scene.rs` | `ViewerTerrainScene::overlay_stack`, `ViewerTerrainScene::load_overlay()` | Store overlay stack, manage GPU textures |
| **Bind group layout** | `src/viewer/terrain/scene.rs` | `ViewerTerrainScene::init_pbr_pipeline()` lines 265-317 | Add `@binding(5)` for overlay texture, `@binding(6)` for overlay sampler |
| **PBR shader** | `src/viewer/terrain/shader_pbr.rs` | `TERRAIN_PBR_SHADER` lines 24-28 | Add overlay_tex and overlay_sampler bindings, sample and blend |
| **Composite pass** | `src/viewer/terrain/overlay_composite.rs` (NEW) | `fn composite_overlays()` | Flatten layer stack to single texture |
| **Render integration** | `src/viewer/terrain/render.rs` | `ViewerTerrainScene::render()` lines 93-501 | Call composite before terrain pass |
| **IPC commands** | `src/viewer/viewer_enums.rs` | `ViewerCommand` enum | Add `LoadOverlay`, `RemoveOverlay`, `SetOverlayOrder` |
| **Python API** | `python/forge3d/terrain_params.py` | `TerrainParams` class | Add `overlays: List[OverlayConfig]` |

---

## 4. GPU Resources and Formats

### Textures

| Resource | Format | Size | Usage |
|----------|--------|------|-------|
| Per-layer texture | `Rgba8UnormSrgb` | Source dimensions | Upload from CPU, read by composite |
| Composite overlay | `Rgba8UnormSrgb` | `terrain_dims × overlay_scale` | Written by composite, read by terrain shader |
| Overlay sampler | Filtering: Linear, AddressMode: ClampToEdge | — | Sample composite in shader |

### Memory Estimates

| Overlay Resolution | Texel Count | Memory (RGBA8) | Memory (RGBA16F) |
|-------------------|-------------|----------------|------------------|
| 1k × 1k | 1M | 4 MB | 8 MB |
| 2k × 2k | 4M | 16 MB | 32 MB |
| 4k × 4k | 16M | 64 MB | 128 MB |

**Conservative estimate**: With 4 layers at 2k resolution each, budget ~64 MB for layer textures + 16 MB for composite = ~80 MB. Well within 512 MiB budget.

### Buffer Layout

```rust
// src/viewer/terrain/overlay.rs
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    /// Overlay extent in terrain UV space: [u_min, v_min, u_max, v_max]
    pub extent_uv: [f32; 4],
    /// Blend params: [opacity, blend_mode, tint_r, tint_g]
    pub blend_params: [f32; 4],
    /// Tint blue + padding: [tint_b, _, _, _]
    pub tint_pad: [f32; 4],
}
```

---

## 5. Shader Changes (WGSL)

### New Bindings in `shader_pbr.rs`

```wgsl
// After existing bindings (0-4), add:
@group(0) @binding(5) var overlay_tex: texture_2d<f32>;
@group(0) @binding(6) var overlay_sampler: sampler;

// In Uniforms struct, add:
    overlay_params: vec4<f32>,  // enabled, opacity, blend_mode, _
```

### Sampling and Blending in Fragment Shader

```wgsl
// In fs_main(), after computing albedo from terrain materials:

// Sample overlay texture (in terrain UV space)
let overlay_enabled = u.overlay_params.x > 0.5;
let overlay_opacity = u.overlay_params.y;

var final_albedo = albedo;
if overlay_enabled {
    let overlay = textureSample(overlay_tex, overlay_sampler, in.uv);
    // Alpha-premultiplied blend into albedo
    let blend_alpha = overlay.a * overlay_opacity;
    final_albedo = mix(albedo, overlay.rgb, blend_alpha);
}

// Continue with lighting using final_albedo instead of albedo
// The overlay is now LIT because it's part of the albedo term
let diffuse = final_albedo * sun_color * ndotl * sun_intensity * shadow_term;
```

### Lighting/Shadow Application Point

The overlay affects **albedo** (material color), which then participates in:
- Diffuse term: `final_albedo * sun_color * ndotl * sun_intensity * shadow_term`
- Ambient term: `sky_ambient(normal) * final_albedo * ambient_strength * ibl_intensity * height_ao`
- Specular is unaffected (comes from roughness/material properties)

This ensures overlays are **fully lit and shadowed** just like the terrain itself.

---

## 6. Overlay Mapping Model

### CRS/Extent → Terrain UV Mapping

```
Terrain coordinate space: [0, terrain_width] × [0, terrain_width]
Terrain UV space: [0, 1] × [0, 1]

overlay_uv = (world_coord - terrain_origin) / terrain_width

For an overlay with extent [x_min, y_min, x_max, y_max] in world coords:
  overlay_uv_min = [x_min / terrain_width, y_min / terrain_width]
  overlay_uv_max = [x_max / terrain_width, y_max / terrain_width]
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| **Overlay exceeds terrain bounds** | Clamp to [0,1] UV range; pixels outside terrain silently ignored |
| **Missing CRS** | Assume overlay is in terrain-local coordinates (pixels = terrain UV × overlay_dims) |
| **Mismatched projection** | Error at load time with clear message; require reprojection by user |
| **Overlay smaller than terrain** | Sample only within overlay UV range; outside region shows terrain albedo |

### Implementation in Shader

```wgsl
fn sample_overlay_in_extent(terrain_uv: vec2<f32>) -> vec4<f32> {
    let extent = u.overlay_extent;  // [u_min, v_min, u_max, v_max]
    
    // Check if terrain_uv is within overlay extent
    if terrain_uv.x < extent.x || terrain_uv.x > extent.z ||
       terrain_uv.y < extent.y || terrain_uv.y > extent.w {
        return vec4<f32>(0.0);  // Transparent outside overlay
    }
    
    // Remap terrain_uv to overlay texture [0,1]
    let overlay_uv = (terrain_uv - extent.xy) / (extent.zw - extent.xy);
    return textureSample(overlay_tex, overlay_sampler, overlay_uv);
}
```

---

## 7. User-Facing API

### Rust API (in `src/viewer/terrain/overlay.rs`)

```rust
/// Single overlay layer configuration
#[derive(Clone)]
pub struct OverlayLayer {
    pub name: String,
    pub data: OverlayData,
    pub extent: Option<[f32; 4]>,  // [x_min, y_min, x_max, y_max] in terrain coords
    pub opacity: f32,              // 0.0 - 1.0, default 1.0
    pub blend_mode: BlendMode,     // Normal, Multiply, Overlay, default Normal
    pub visible: bool,             // default true
    pub z_order: i32,              // Stacking order, lower = behind
}

pub enum OverlayData {
    Raster { rgba: Vec<u8>, width: u32, height: u32 },
    Image { path: PathBuf },
    #[cfg(feature = "vector-overlay")]
    Vector { features: Vec<VectorFeature>, style: VectorStyle },
}

pub enum BlendMode {
    Normal,   // Standard alpha blend
    Multiply, // overlay.rgb * albedo.rgb
    Overlay,  // Photoshop-style overlay
}

impl ViewerTerrainScene {
    /// Add an overlay layer. Returns layer ID.
    pub fn add_overlay(&mut self, layer: OverlayLayer) -> u32;
    
    /// Remove an overlay by ID.
    pub fn remove_overlay(&mut self, id: u32) -> bool;
    
    /// Set overlay visibility.
    pub fn set_overlay_visible(&mut self, id: u32, visible: bool);
    
    /// Set overlay opacity (0.0 - 1.0).
    pub fn set_overlay_opacity(&mut self, id: u32, opacity: f32);
    
    /// Reorder overlays. Overwrites z_order for all layers.
    pub fn reorder_overlays(&mut self, order: &[u32]);
    
    /// Get list of all overlay IDs in z-order.
    pub fn list_overlays(&self) -> Vec<u32>;
}
```

### Python API (in `python/forge3d/terrain_params.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum
import numpy as np

class BlendMode(Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    OVERLAY = "overlay"

@dataclass
class OverlayConfig:
    """Configuration for a terrain overlay layer."""
    name: str
    source: Union[str, np.ndarray]  # Path to image or RGBA numpy array
    extent: Optional[tuple[float, float, float, float]] = None  # x_min, y_min, x_max, y_max
    opacity: float = 1.0
    blend_mode: BlendMode = BlendMode.NORMAL
    visible: bool = True
    z_order: int = 0

# Example usage:
from forge3d import open_terrain_viewer

view = open_terrain_viewer("terrain.tif", width=1920, height=1080)

# Add satellite imagery as base overlay
view.add_overlay(OverlayConfig(
    name="satellite",
    source="satellite.png",
    extent=(0, 0, 1000, 1000),  # World coordinates
    opacity=0.8,
    blend_mode=BlendMode.MULTIPLY,
))

# Add contour lines (pre-rendered to PNG)
view.add_overlay(OverlayConfig(
    name="contours",
    source="contours.png",
    opacity=0.5,
    z_order=1,  # On top of satellite
))

# Add dynamic heatmap from numpy array
heatmap = np.zeros((512, 512, 4), dtype=np.uint8)
# ... fill heatmap ...
view.add_overlay(OverlayConfig(
    name="heatmap",
    source=heatmap,
    opacity=0.7,
    z_order=2,
))
```

### IPC Commands (for interactive viewer)

```json
// Load overlay from file
{"cmd": "load_overlay", "path": "overlay.png", "extent": [0, 0, 1000, 1000], "opacity": 0.8}

// Load overlay from base64 RGBA data
{"cmd": "load_overlay_data", "data": "base64...", "width": 512, "height": 512, "opacity": 1.0}

// Remove overlay
{"cmd": "remove_overlay", "id": 0}

// Set overlay visibility
{"cmd": "set_overlay_visible", "id": 0, "visible": false}

// Set overlay opacity
{"cmd": "set_overlay_opacity", "id": 0, "opacity": 0.5}

// Reorder overlays
{"cmd": "reorder_overlays", "order": [2, 0, 1]}
```

### Defaults

- **Opacity**: 1.0 (fully opaque)
- **Blend mode**: Normal (alpha blend)
- **Visibility**: true
- **Z-order**: 0 (layers at same z-order sorted by insertion order)
- **Overlay feature**: Off by default (existing behavior preserved)

---

## 8. Determinism and Quality

### Z-Fighting Prevention

Not applicable for Option A — overlays are texture-space, not geometry. No depth testing issues.

### Aliasing and Shimmering

| Issue | Mitigation |
|-------|------------|
| Overlay texture aliasing | Use trilinear filtering with mipmaps on overlay texture |
| Edge artifacts | Apply 1-pixel padding around overlay edges with transparent border |
| Temporal shimmering | N/A for static overlays; for animated overlays, use temporal AA in post-process |

### Lighting Integration

Overlays modify **albedo** before lighting, so they receive:

1. **Diffuse lighting**: `final_albedo * sun_color * ndotl * sun_intensity`
2. **Shadow term**: `* shadow_term` (from sun visibility compute pass)
3. **Ambient/IBL**: `sky_ambient(normal) * final_albedo * ambient_strength * height_ao`

This means overlays:
- ✅ Are lit by sun (diffuse term includes overlay color)
- ✅ Are shadowed (shadow_term multiplies diffuse result)
- ✅ Receive ambient occlusion (height_ao multiplies ambient term)
- ❌ Do not affect specular (specular depends on roughness, not albedo)

### Anti-Aliasing Approach

Consistent with repo design (no MSAA on terrain currently):
- Overlay textures should be pre-AA'd at rasterization time (for vector layers)
- GPU texture sampling uses bilinear filtering
- Post-process AA (if enabled) applies to final frame

---

## 9. Validation and Tests

### Automated Tests

#### Unit Test: Overlay Stack Logic

```python
# tests/test_terrain_overlay_stack.py

def test_overlay_layer_ordering():
    """Verify layers are stacked in z-order."""
    # Create 3 layers with different z-orders
    # Composite and check pixel values at known positions
    pass

def test_overlay_extent_mapping():
    """Verify overlay UV mapping respects extent."""
    # Create overlay with non-full extent
    # Check that overlay only appears in specified region
    pass

def test_overlay_blend_modes():
    """Verify blend modes produce expected results."""
    # Test Normal, Multiply, Overlay modes with known inputs
    pass
```

#### Integration Test: Overlay Rendering

```python
# tests/test_terrain_overlay_rendering.py

def test_overlay_lit_by_sun(viewer_context):
    """Verify overlay receives sun lighting."""
    sock, proc = viewer_context
    
    # Load terrain
    send_ipc(sock, {"cmd": "terrain.load", "path": DEM_PATH})
    
    # Add solid color overlay
    send_ipc(sock, {"cmd": "load_overlay", "path": "red_overlay.png", "opacity": 1.0})
    
    # Capture with sun at two different angles
    send_ipc(sock, {"cmd": "terrain.sun", "azimuth": 90, "elevation": 45})
    snap1 = capture_and_hash(sock, "overlay_sun_east")
    
    send_ipc(sock, {"cmd": "terrain.sun", "azimuth": 270, "elevation": 45})
    snap2 = capture_and_hash(sock, "overlay_sun_west")
    
    # Overlay should look different due to lighting change
    assert snap1 != snap2, "Overlay must be affected by sun direction"

def test_overlay_default_off():
    """Verify overlays don't change output when disabled (regression)."""
    # Render with overlays=None
    # Compare to baseline hash
    pass
```

### Commands to Run

```bash
# Run all overlay tests
python -m pytest tests/test_terrain_overlay*.py -v

# Run with coverage
python -m pytest tests/test_terrain_overlay*.py --cov=python/forge3d --cov-report=term-missing

# Cargo check for Rust compilation
cargo check --lib
```

### Viewer Sanity Run Procedure

1. Build: `cargo build --release`
2. Install: `pip install -e .`
3. Run: `python -m forge3d.terrain_demo --dem assets/sample_dem.tif`
4. Commands in viewer:
   ```
   pbr.enabled 1
   load_overlay assets/test_overlay.png
   set_overlay_opacity 0 0.7
   snap overlay_test.png
   ```
5. Expected: Overlay visible on terrain, lit by sun, receives shadows

---

## 10. Milestones and Deliverables

### Milestone 1: Overlay Data Structures

| Item | Details |
|------|---------|
| **Name** | Overlay Data Structures and CPU Stack |
| **Files** | `src/viewer/terrain/overlay.rs` (NEW), `src/viewer/terrain/mod.rs` |
| **Deliverables** | `OverlayLayer`, `OverlayStack`, `BlendMode` structs; unit tests for stacking logic |
| **Acceptance** | `cargo check` passes; `cargo test overlay` passes |
| **Risks** | None — pure Rust data structures |

### Milestone 2: GPU Composite Pass

| Item | Details |
|------|---------|
| **Name** | GPU Overlay Composite Pass |
| **Files** | `src/viewer/terrain/overlay_composite.rs` (NEW), `src/shaders/overlay_composite.wgsl` (NEW) |
| **Deliverables** | Compute/render pass that flattens layer stack to single texture |
| **Acceptance** | Can create composite texture from 2+ layers; visual inspection shows correct blending |
| **Risks** | Texture format compatibility (mitigate with `Rgba8UnormSrgb` fall-back) |

### Milestone 3: Shader Integration

| Item | Details |
|------|---------|
| **Name** | PBR Shader Overlay Sampling |
| **Files** | `src/viewer/terrain/shader_pbr.rs`, `src/viewer/terrain/scene.rs` |
| **Deliverables** | Overlay texture bound at @binding(5); sampled and blended into albedo |
| **Acceptance** | Overlay visible in PBR render; affected by changing sun direction |
| **Risks** | Bind group layout changes may require pipeline rebuild; mitigate with versioned layouts |

### Milestone 4: IPC and Python API

| Item | Details |
|------|---------|
| **Name** | User-Facing API |
| **Files** | `src/viewer/viewer_enums.rs`, `python/forge3d/terrain_params.py` |
| **Deliverables** | IPC commands for load/remove/toggle overlays; Python API |
| **Acceptance** | Can load overlay via Python; viewer IPC accepts commands |
| **Risks** | API surface decisions; mitigate with minimal initial API |

### Milestone 5: Vector Rasterization (Optional)

| Item | Details |
|------|---------|
| **Name** | Vector-to-Raster Overlay |
| **Files** | `src/viewer/terrain/vector_raster.rs` (NEW) |
| **Deliverables** | CPU rasterizer for points/lines/polygons to overlay texture |
| **Acceptance** | Can rasterize GeoJSON to overlay and display on terrain |
| **Risks** | Scope creep; gate behind `vector-overlay` feature flag |

### Milestone 6: Automated Tests

| Item | Details |
|------|---------|
| **Name** | Test Suite |
| **Files** | `tests/test_terrain_overlay*.py` |
| **Deliverables** | Integration tests for overlay lit/shadow behavior; regression test for default-off |
| **Acceptance** | `pytest tests/test_terrain_overlay*.py` all green |
| **Risks** | Flaky IPC tests; mitigate with robust timeout handling |

### Milestone 7: Documentation

| Item | Details |
|------|---------|
| **Name** | User Documentation |
| **Files** | `docs/terrain_overlays.rst` (NEW), `docs/api/overlay.rst` (NEW) |
| **Deliverables** | User guide, API reference, examples |
| **Acceptance** | `make html` builds; examples run |
| **Risks** | None |

---

## 11. Rollout Strategy

### Feature Flag

```rust
// src/viewer/terrain/pbr_renderer.rs
#[derive(Clone)]
pub struct ViewerTerrainPbrConfig {
    pub enabled: bool,
    // ... existing fields ...
    
    /// Enable overlay system (default: false)
    pub overlays_enabled: bool,
}

impl Default for ViewerTerrainPbrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // ...
            overlays_enabled: false,  // DEFAULT OFF
        }
    }
}
```

### Default Behavior

- `overlays_enabled = false` → no overlay texture binding, no composite pass, shader uses `final_albedo = albedo`
- Existing tests and renders are **byte-identical** to before
- User must explicitly enable via `pbr.overlays_enabled 1` or Python API

### Backend Fallback

| Backend | Issue | Fallback |
|---------|-------|----------|
| **Metal** | R32Float not filterable | Use `Rgba8UnormSrgb` (already required for overlay textures) |
| **WebGL2** | Limited texture formats | Error early if required format unavailable |
| **All** | Low VRAM | Skip overlay composite if VRAM < 128 MB free; log warning |

### Pinned Preset for Baseline

```json
// presets/baseline_no_overlays.json
{
    "pbr": {
        "enabled": true,
        "overlays_enabled": false
    }
}
```

Run: `python -m pytest tests/ --preset=baseline_no_overlays` to verify old behavior.

---

## Appendix: Evidence for Terrain Representation (T1 Answer)

**Answer: HYBRID**

The terrain is rendered from a **grid mesh displaced by a heightmap texture**:

1. **Grid mesh generation**: `src/viewer/terrain/scene.rs` → `create_grid_mesh(resolution)` (lines 750-778) generates a flat grid of vertices with UV coordinates
2. **Heightmap texture**: DEM loaded into `R32Float` texture at `load_terrain()` (lines 559-592)
3. **Vertex shader displacement**: `src/viewer/terrain/shader.rs` (lines 26-55) — `textureLoad(heightmap, texel, 0).r` samples height and computes `world_pos.y`
4. **UV mapping**: Vertex UVs match terrain grid position exactly (0-1 range)

This HYBRID approach means overlays can be mapped to terrain UV space and sampled in the fragment shader, making Option A (draped textures) the most natural fit.
