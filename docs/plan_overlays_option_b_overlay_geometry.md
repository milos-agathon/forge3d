# Option B — Overlay Geometry Rendered as Additional Lit Passes

> **Goal**: Render vector overlays as GPU geometry (points/lines/polygons) in world space, optionally draped onto the terrain heightfield, with proper lighting and shadowing integration (same sun/IBL/shadow inputs), depth-tested against terrain, with anti-aliasing strategy.

## 1. Scope and Non-Goals

### In Scope

- Vector geometry overlays (points, lines, polygons) rendered as GPU primitives
- World-space positioning with optional draping onto terrain heightfield
- Lighting integration: same sun direction, intensity, shadows as terrain
- Depth testing against terrain depth buffer
- Anti-aliasing via MSAA or line-width AA
- Multiple overlay layers with deterministic ordering via depth bias
- Opt-in feature flag (default off, preserves existing behavior)

### Non-Goals

- Raster/image overlays (better suited to Option A)
- Screen-space UI overlays (use existing screen-space system)
- Full 3D geometry (buildings, trees) — separate feature
- Per-vertex animation or skeletal animation
- Tesselation shaders for smooth curves (use pre-tessellated CPU geometry)

---

## 2. Architecture Overview (Data Flow)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            USER API (Python/Rust)                            │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │ VectorOverlay { vertices, indices, style, drape }
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Overlay Geometry Manager (CPU)                          │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│   │  Layer 0    │  │  Layer 1    │  │  Layer N    │  ← VectorOverlayLayer   │
│   │ (polygons)  │  │ (lines)     │  │ (points)    │                         │
│   └─────────────┘  └─────────────┘  └─────────────┘                         │
│         │                │                │                                  │
│         ▼                ▼                ▼                                  │
│   Drape Pass: Sample heightmap for each vertex, adjust Y                    │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │ Upload vertex/index buffers to GPU
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      GPU Vertex/Index Buffers                                │
│                      Per-layer: vertices, indices, instance data             │
└─────────────────┬────────────────────────────────────────────────────────────┘
                  │ 
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Overlay Render Pass (GPU)                                 │
│   For each layer (in z-order):                                               │
│     1. Bind overlay pipeline (point/line/fill variant)                       │
│     2. Bind terrain uniforms (sun_dir, lighting, shadow maps)                │
│     3. Apply depth bias to avoid z-fighting                                  │
│     4. Draw indexed primitives                                               │
│   Shader applies sun lighting + shadow lookup                                │
└──────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Frame Buffer (shared with terrain)                        │
│   Overlays write to same color + depth attachments                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Integration Points (File-Path + Symbol Specific)

| Component | File | Symbols | Purpose |
|-----------|------|---------|---------|
| **Overlay geometry struct** | `src/viewer/terrain/vector_overlay.rs` (NEW) | `struct VectorOverlayLayer`, `struct VectorVertex` | Hold geometry, style, drape config |
| **Scene integration** | `src/viewer/terrain/scene.rs` | `ViewerTerrainScene::vector_overlays`, `ViewerTerrainScene::add_vector_overlay()` | Store overlay layers, manage buffers |
| **Drape logic** | `src/viewer/terrain/vector_overlay.rs` (NEW) | `fn drape_vertices()` | Sample heightmap for vertex Y |
| **Overlay pipeline** | `src/viewer/terrain/overlay_pipeline.rs` (NEW) | `fn create_overlay_pipeline()` | Create render pipeline for overlays |
| **Overlay shader** | `src/shaders/vector_overlay.wgsl` (NEW) | `vs_main`, `fs_main` | Vertex transform, lit fragment |
| **Render integration** | `src/viewer/terrain/render.rs` | `ViewerTerrainScene::render()` | Call overlay render after terrain pass |
| **Shared uniforms** | `src/viewer/terrain/render.rs` | `TerrainPbrUniforms` | Pass sun_dir, lighting to overlay shader |
| **IPC commands** | `src/viewer/viewer_enums.rs` | `ViewerCommand` enum | Add `AddVectorOverlay`, `RemoveVectorOverlay` |
| **Python API** | `python/forge3d/terrain_params.py` | `VectorOverlayConfig` class | Python-side overlay configuration |

---

## 4. GPU Resources and Formats

### Vertex Format

```rust
// src/viewer/terrain/vector_overlay.rs
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VectorVertex {
    pub position: [f32; 3],  // World XYZ (Y may be offset for drape)
    pub color: [f32; 4],     // RGBA vertex color
    pub uv: [f32; 2],        // Texture coords (for textured overlays)
    pub normal: [f32; 3],    // For lit overlays (default: up vec)
}
// Stride: 48 bytes
```

### Buffers

| Resource | Usage | Size Estimate |
|----------|-------|---------------|
| Vertex buffer (per layer) | `VERTEX \| COPY_DST` | 48 bytes × vertex_count |
| Index buffer (per layer) | `INDEX \| COPY_DST` | 4 bytes × index_count (u32) |
| Overlay uniforms | `UNIFORM \| COPY_DST` | 128 bytes |

### Memory Estimates

| Overlay Complexity | Vertices | Indices | Memory |
|-------------------|----------|---------|--------|
| Simple contour lines (1k segments) | 2k | 2k | ~100 KB |
| Medium polygon set (10k triangles) | 30k | 30k | ~1.5 MB |
| Dense vector map (100k triangles) | 300k | 300k | ~15 MB |

Total for 4 layers at medium complexity: ~6 MB. Well within budget.

### Overlay Uniform Buffer

```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayUniforms {
    pub view_proj: [[f32; 4]; 4],    // 64 bytes
    pub sun_dir: [f32; 4],            // 16 bytes
    pub lighting: [f32; 4],           // sun_intensity, ambient, shadow_strength, _
    pub layer_params: [f32; 4],       // opacity, depth_bias, line_width, point_size
}
```

---

## 5. Shader Changes (WGSL)

### New Shader: `src/shaders/vector_overlay.wgsl`

```wgsl
// Vector Overlay Shader with Lighting
// Shared lighting model with terrain PBR shader

struct Uniforms {
    view_proj: mat4x4<f32>,
    sun_dir: vec4<f32>,
    lighting: vec4<f32>,       // sun_intensity, ambient, shadow_strength, _
    layer_params: vec4<f32>,   // opacity, depth_bias, line_width, point_size
};

@group(0) @binding(0) var<uniform> u: Uniforms;
// Optional: sun visibility texture for shadows
@group(0) @binding(1) var sun_vis_tex: texture_2d<f32>;
@group(0) @binding(2) var sun_vis_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) normal: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) terrain_uv: vec2<f32>,  // For shadow lookup
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Apply depth bias by slightly moving towards camera
    var pos = in.position;
    let depth_bias = u.layer_params.y;
    pos.y += depth_bias;  // Small offset to prevent z-fighting
    
    out.clip_pos = u.view_proj * vec4<f32>(pos, 1.0);
    out.color = in.color;
    out.world_pos = pos;
    out.normal = in.normal;
    
    // Compute terrain UV for shadow lookup (assumes terrain at origin, same width as coords)
    // This should match terrain UV mapping
    let terrain_width = 1000.0;  // Should be uniform
    out.terrain_uv = pos.xz / terrain_width;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_intensity = u.lighting.x;
    let ambient = u.lighting.y;
    let shadow_strength = u.lighting.z;
    let opacity = u.layer_params.x;
    
    // Base color from vertex
    let albedo = in.color.rgb;
    
    // Sun lighting
    let sun_dir = normalize(u.sun_dir.xyz);
    let normal = normalize(in.normal);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    
    // Sample sun visibility for shadows
    let sv_dims = textureDimensions(sun_vis_tex, 0);
    let sv_pixel = vec2<i32>(in.terrain_uv * vec2<f32>(sv_dims));
    let sv_clamped = clamp(sv_pixel, vec2<i32>(0), vec2<i32>(sv_dims) - vec2<i32>(1));
    let sun_vis = textureLoad(sun_vis_tex, sv_clamped, 0).r;
    
    // Shadow term
    let shadow_term = mix(1.0, sun_vis, shadow_strength);
    
    // Final lighting (same model as terrain)
    let sun_color = vec3<f32>(1.0, 0.95, 0.85);
    let diffuse = albedo * sun_color * ndotl * sun_intensity * shadow_term;
    let ambient_color = albedo * ambient;
    
    var color = diffuse + ambient_color;
    
    return vec4<f32>(color, in.color.a * opacity);
}
```

### Key Lighting Terms

The overlay shader uses the **same lighting model** as the terrain PBR shader:
- **Diffuse**: `albedo * sun_color * ndotl * sun_intensity * shadow_term`
- **Shadow lookup**: Sample `sun_vis_tex` at terrain UV to get shadow factor
- **Ambient**: `albedo * ambient`

This ensures overlays receive identical lighting and shadows as the terrain surface.

---

## 6. Overlay Mapping Model

### World-Space Positioning

Vector overlay vertices are specified in **terrain world coordinates**:
- X: horizontal position (matches terrain X)
- Y: height (absolute world height, or 0 if draping)
- Z: horizontal position (matches terrain Z)

### Draping Algorithm

For overlays that should conform to terrain surface:

```rust
// src/viewer/terrain/vector_overlay.rs
fn drape_vertices(
    vertices: &mut [VectorVertex],
    heightmap: &[f32],
    dims: (u32, u32),
    terrain_width: f32,
    height_offset: f32,  // Offset above terrain surface
) {
    for v in vertices.iter_mut() {
        // Convert world XZ to terrain UV
        let u = v.position[0] / terrain_width;
        let z = v.position[2] / terrain_width;
        
        // Sample heightmap (bilinear interpolation)
        let terrain_height = sample_heightmap_bilinear(
            heightmap, dims, u, z
        );
        
        // Set vertex Y to terrain height + offset
        v.position[1] = terrain_height + height_offset;
        
        // Compute normal from terrain gradient for proper lighting
        v.normal = compute_terrain_normal_at(heightmap, dims, u, z);
    }
}
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| **Vertex outside terrain bounds** | Clamp to terrain edge height; log warning |
| **Missing heightmap** | Error at drape time; require flat overlay or skip drape |
| **Extreme height offset** | Allow negative (underground) and large positive offsets |
| **Non-closed polygons** | Render as-is; no automatic closing |

---

## 7. User-Facing API

### Rust API (in `src/viewer/terrain/vector_overlay.rs`)

```rust
/// Primitive type for vector overlay
#[derive(Clone, Copy)]
pub enum OverlayPrimitive {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
}

/// Vector overlay layer configuration
#[derive(Clone)]
pub struct VectorOverlayLayer {
    pub name: String,
    pub vertices: Vec<VectorVertex>,
    pub indices: Vec<u32>,
    pub primitive: OverlayPrimitive,
    pub drape: bool,              // If true, drape onto terrain
    pub drape_offset: f32,        // Height above terrain when draped
    pub opacity: f32,             // 0.0 - 1.0
    pub depth_bias: f32,          // Z-fighting prevention (0.01 - 1.0)
    pub line_width: f32,          // For Lines/LineStrip (1.0 - 10.0)
    pub point_size: f32,          // For Points (1.0 - 20.0)
    pub visible: bool,
    pub z_order: i32,
}

impl ViewerTerrainScene {
    /// Add a vector overlay layer. Returns layer ID.
    pub fn add_vector_overlay(&mut self, layer: VectorOverlayLayer) -> u32;
    
    /// Update vertices for an existing layer (for animation).
    pub fn update_vector_overlay(&mut self, id: u32, vertices: Vec<VectorVertex>);
    
    /// Remove a vector overlay by ID.
    pub fn remove_vector_overlay(&mut self, id: u32) -> bool;
    
    /// Set vector overlay visibility.
    pub fn set_vector_overlay_visible(&mut self, id: u32, visible: bool);
    
    /// Set vector overlay opacity.
    pub fn set_vector_overlay_opacity(&mut self, id: u32, opacity: f32);
    
    /// List all vector overlay IDs.
    pub fn list_vector_overlays(&self) -> Vec<u32>;
}
```

### Python API (in `python/forge3d/terrain_params.py`)

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import numpy as np

class PrimitiveType(Enum):
    POINTS = "points"
    LINES = "lines"
    LINE_STRIP = "line_strip"
    TRIANGLES = "triangles"
    TRIANGLE_STRIP = "triangle_strip"

@dataclass
class VectorVertex:
    """Single vertex for vector overlay."""
    x: float
    y: float
    z: float
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

@dataclass
class VectorOverlayConfig:
    """Configuration for a vector overlay layer."""
    name: str
    vertices: List[VectorVertex]
    indices: List[int]
    primitive: PrimitiveType = PrimitiveType.TRIANGLES
    drape: bool = False
    drape_offset: float = 0.5       # Meters above terrain when draped
    opacity: float = 1.0
    depth_bias: float = 0.1
    line_width: float = 2.0
    point_size: float = 5.0
    visible: bool = True
    z_order: int = 0

# Example usage:
from forge3d import open_terrain_viewer

view = open_terrain_viewer("terrain.tif", width=1920, height=1080)

# Add a simple triangle
view.add_vector_overlay(VectorOverlayConfig(
    name="marker",
    vertices=[
        VectorVertex(100, 0, 100, r=1, g=0, b=0),
        VectorVertex(200, 0, 100, r=0, g=1, b=0),
        VectorVertex(150, 0, 200, r=0, g=0, b=1),
    ],
    indices=[0, 1, 2],
    primitive=PrimitiveType.TRIANGLES,
    drape=True,
    drape_offset=1.0,
))

# Add contour lines
view.add_vector_overlay(VectorOverlayConfig(
    name="contours",
    vertices=contour_vertices,  # From contour algorithm
    indices=contour_indices,
    primitive=PrimitiveType.LINES,
    drape=True,
    drape_offset=0.5,
    line_width=1.5,
    opacity=0.8,
    z_order=1,
))
```

### IPC Commands

```json
// Add vector overlay (triangle list)
{
    "cmd": "add_vector_overlay",
    "name": "marker",
    "primitive": "triangles",
    "vertices": [
        {"x": 100, "y": 0, "z": 100, "r": 1, "g": 0, "b": 0, "a": 1},
        {"x": 200, "y": 0, "z": 100, "r": 0, "g": 1, "b": 0, "a": 1},
        {"x": 150, "y": 0, "z": 200, "r": 0, "g": 0, "b": 1, "a": 1}
    ],
    "indices": [0, 1, 2],
    "drape": true,
    "drape_offset": 1.0,
    "opacity": 1.0
}

// Update overlay visibility
{"cmd": "set_vector_overlay_visible", "id": 0, "visible": false}

// Remove overlay
{"cmd": "remove_vector_overlay", "id": 0}
```

### Defaults

- **Drape**: false (use absolute Y coordinates)
- **Drape offset**: 0.5 meters
- **Opacity**: 1.0
- **Depth bias**: 0.1
- **Line width**: 2.0
- **Point size**: 5.0
- **Visibility**: true

---

## 8. Determinism and Quality

### Z-Fighting Prevention

Vector overlays share the depth buffer with terrain, which can cause z-fighting. Mitigations:

1. **Depth bias in vertex shader**: Offset vertex Y by small amount (configurable `depth_bias`)
2. **GPU depth bias state**: Use `wgpu::DepthBiasState { constant: 1, slope_scale: 1.0, clamp: 0.0 }`
3. **Per-layer ordering**: Higher z_order layers get larger depth bias

```rust
// In pipeline creation
depth_stencil: Some(wgpu::DepthStencilState {
    format: wgpu::TextureFormat::Depth32Float,
    depth_write_enabled: true,
    depth_compare: wgpu::CompareFunction::LessEqual,  // LessEqual for overlays
    stencil: wgpu::StencilState::default(),
    bias: wgpu::DepthBiasState {
        constant: 1,           // Constant offset
        slope_scale: 1.0,      // Slope-based offset
        clamp: 0.0,            // No clamping
    },
}),
```

### Aliasing and Shimmering

| Issue | Mitigation |
|-------|------------|
| **Line aliasing** | Use MSAA (if enabled globally) or line-width AA |
| **Polygon edge aliasing** | MSAA or conservative rasterization (if available) |
| **Temporal shimmering during navigation** | Stable vertex positions; avoid sub-pixel jitter |
| **Point sprites** | Use smooth point textures with alpha falloff |

### Anti-Aliasing Strategy

Consistent with repo design:
1. **Primary**: Rely on global MSAA if enabled (`multisample: wgpu::MultisampleState { count: 4, ... }`)
2. **Lines**: Use `line_width` > 1.0 for built-in AA on some backends
3. **Edges**: Alpha-blend polygon edges with soft transition in shader (optional)

### Lighting and Shadow Integration

Vector overlays use **identical lighting** to terrain:

| Lighting Term | Source | Application |
|---------------|--------|-------------|
| **Sun direction** | `u.sun_dir` (from terrain uniforms) | `ndotl = dot(normal, sun_dir)` |
| **Sun intensity** | `u.lighting.x` | Multiplied with diffuse |
| **Ambient** | `u.lighting.y` | Added as ambient term |
| **Shadow** | `sun_vis_tex` sampled at terrain UV | `shadow_term = mix(1.0, sun_vis, shadow_strength)` |

Overlays are **fully lit and shadowed** because:
- They sample the **same sun visibility texture** as terrain
- They use the **same diffuse + ambient model**
- Normals come from terrain gradient (when draped) or vertex data

---

## 9. Validation and Tests

### Automated Tests

#### Unit Test: Draping Logic

```python
# tests/test_vector_overlay_drape.py

def test_drape_vertices_flat_terrain():
    """Verify draping on flat terrain sets correct Y."""
    # Create flat heightmap (all zeros)
    # Drape vertices; verify Y = drape_offset
    pass

def test_drape_vertices_slope():
    """Verify draping on sloped terrain interpolates correctly."""
    # Create heightmap with known gradient
    # Drape vertex at known position
    # Verify Y matches expected height
    pass

def test_drape_outside_bounds():
    """Verify vertices outside terrain are clamped."""
    # Vertex at x=-100 should clamp to edge
    pass
```

#### Integration Test: Lit Overlay Rendering

```python
# tests/test_vector_overlay_rendering.py

def test_vector_overlay_lit_by_sun(viewer_context):
    """Verify vector overlay receives sun lighting."""
    sock, proc = viewer_context
    
    # Load terrain
    send_ipc(sock, {"cmd": "terrain.load", "path": DEM_PATH})
    
    # Add draped overlay
    send_ipc(sock, {
        "cmd": "add_vector_overlay",
        "name": "test_tri",
        "primitive": "triangles",
        "vertices": [...],
        "indices": [0, 1, 2],
        "drape": True,
    })
    
    # Capture with sun at two angles
    send_ipc(sock, {"cmd": "terrain.sun", "azimuth": 90, "elevation": 45})
    snap1 = capture_and_hash(sock, "overlay_sun_east")
    
    send_ipc(sock, {"cmd": "terrain.sun", "azimuth": 270, "elevation": 45})
    snap2 = capture_and_hash(sock, "overlay_sun_west")
    
    # Overlay should look different due to lighting
    assert snap1 != snap2, "Vector overlay must be affected by sun direction"

def test_vector_overlay_receives_shadows(viewer_context):
    """Verify vector overlay is shadowed by terrain."""
    # Place overlay in known shadow area
    # Compare to same overlay in lit area
    pass

def test_vector_overlay_default_off():
    """Verify no vector overlays by default (regression)."""
    # Render without adding overlays
    # Compare to baseline hash
    pass
```

### Commands to Run

```bash
# Run all vector overlay tests
python -m pytest tests/test_vector_overlay*.py -v

# Run with coverage
python -m pytest tests/test_vector_overlay*.py --cov=python/forge3d --cov-report=term-missing

# Cargo check for Rust compilation
cargo check --lib
```

### Viewer Sanity Run Procedure

1. Build: `cargo build --release`
2. Install: `pip install -e .`
3. Run: `python -m forge3d.terrain_demo --dem assets/sample_dem.tif`
4. Add overlay via Python:
   ```python
   import forge3d
   # ... add vector overlay programmatically
   ```
5. Commands in viewer:
   ```
   pbr.enabled 1
   snap vector_overlay_test.png
   ```
6. Expected: Vector overlay visible, conforming to terrain, lit by sun with shadows

---

## 10. Milestones and Deliverables

### Milestone 1: Vector Overlay Data Structures

| Item | Details |
|------|---------|
| **Name** | Vector Overlay Structs and Buffers |
| **Files** | `src/viewer/terrain/vector_overlay.rs` (NEW), `src/viewer/terrain/mod.rs` |
| **Deliverables** | `VectorVertex`, `VectorOverlayLayer` structs; GPU buffer management |
| **Acceptance** | `cargo check` passes; can create overlay struct |
| **Risks** | None — pure Rust data structures |

### Milestone 2: Draping Implementation

| Item | Details |
|------|---------|
| **Name** | Heightmap Draping |
| **Files** | `src/viewer/terrain/vector_overlay.rs` |
| **Deliverables** | `drape_vertices()` function with bilinear sampling; normal computation |
| **Acceptance** | Unit tests for draping pass; visual inspection shows overlay follows terrain |
| **Risks** | Edge case handling; mitigate with clamping and warnings |

### Milestone 3: Overlay Render Pipeline

| Item | Details |
|------|---------|
| **Name** | GPU Pipeline for Vector Overlays |
| **Files** | `src/viewer/terrain/overlay_pipeline.rs` (NEW), `src/shaders/vector_overlay.wgsl` (NEW) |
| **Deliverables** | Render pipeline with depth bias; shader with lighting |
| **Acceptance** | Can render simple triangle overlay; receives lighting |
| **Risks** | Depth bias tuning; mitigate with configurable bias |

### Milestone 4: Shadow Integration

| Item | Details |
|------|---------|
| **Name** | Sun Visibility Lookup in Overlay Shader |
| **Files** | `src/shaders/vector_overlay.wgsl`, `src/viewer/terrain/render.rs` |
| **Deliverables** | Bind sun_vis_tex to overlay shader; sample for shadow |
| **Acceptance** | Overlay in shadow area is visibly darker |
| **Risks** | UV mapping must match terrain; mitigate with shared uniform |

### Milestone 5: Render Integration

| Item | Details |
|------|---------|
| **Name** | Integrate Overlay Pass into Render Loop |
| **Files** | `src/viewer/terrain/render.rs` |
| **Deliverables** | Call overlay render after terrain; proper ordering |
| **Acceptance** | Overlays render on top of terrain with correct depth |
| **Risks** | Frame timing; mitigate by rendering in same command buffer |

### Milestone 6: IPC and Python API

| Item | Details |
|------|---------|
| **Name** | User-Facing API |
| **Files** | `src/viewer/viewer_enums.rs`, `python/forge3d/terrain_params.py` |
| **Deliverables** | IPC commands; Python `VectorOverlayConfig` class |
| **Acceptance** | Can add overlay via Python; IPC commands work |
| **Risks** | API surface decisions; mitigate with minimal initial API |

### Milestone 7: Tests and Documentation

| Item | Details |
|------|---------|
| **Name** | Test Suite and Docs |
| **Files** | `tests/test_vector_overlay*.py`, `docs/vector_overlays.rst` (NEW) |
| **Deliverables** | Integration tests; user documentation |
| **Acceptance** | All tests green; docs build |
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
    
    /// Enable vector overlay system (default: false)
    pub vector_overlays_enabled: bool,
}

impl Default for ViewerTerrainPbrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            // ...
            vector_overlays_enabled: false,  // DEFAULT OFF
        }
    }
}
```

### Default Behavior

- `vector_overlays_enabled = false` → no overlay pipeline, no render pass, no buffers
- Existing tests and renders are **byte-identical** to before
- User must explicitly enable via `pbr.vector_overlays_enabled 1` or add overlay

### Backend Fallback

| Backend | Issue | Fallback |
|---------|-------|----------|
| **Metal** | Line width > 1.0 not supported | Use width = 1.0; warn user |
| **WebGL2** | Point sprites limited | Fall back to small quads |
| **All** | MSAA not enabled | Warn that edges may alias |

### Pinned Preset for Baseline

```json
// presets/baseline_no_vector_overlays.json
{
    "pbr": {
        "enabled": true,
        "vector_overlays_enabled": false
    }
}
```

Run: `python -m pytest tests/ --preset=baseline_no_vector_overlays` to verify old behavior.

---

## Appendix: Comparison to Option A

| Aspect | Option A (Textures) | Option B (Geometry) |
|--------|---------------------|---------------------|
| **Best for** | Raster imagery, satellite photos | Vector data, contours, markers |
| **Memory** | Higher (full resolution textures) | Lower (vertices only) |
| **Lighting** | Blends into albedo | Direct lighting calculation |
| **Shadows** | Via terrain shadow term | Samples same shadow texture |
| **Anti-aliasing** | Texture filtering | MSAA or line AA |
| **Dynamic updates** | Re-upload texture | Update vertex buffer |
| **Z-fighting** | N/A (texture-space) | Requires depth bias |
| **Implementation complexity** | Lower | Higher |

**Recommendation**: Use Option A for raster/image overlays, Option B for vector/annotation overlays. Both can coexist in the same rendering pipeline.
