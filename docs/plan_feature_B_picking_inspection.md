# Feature B: Picking + Feature Inspection for Overlays and Terrain

## Investigation Summary

### Current Overlay Representation

**Vector overlays** (`src/viewer/terrain/vector_overlay.rs`):
- `VectorVertex`: position `[f32; 3]`, color `[f32; 4]`, uv `[f32; 2]`, normal `[f32; 3]`
- `VectorOverlayStack`: manages multiple `VectorOverlayLayer` instances
- Geometry stored in GPU vertex/index buffers
- **No feature ID encoded** — vertices have color but no identifier

**Terrain** (`src/viewer/terrain/scene.rs`):
- `ViewerTerrainData`: heightmap texture + grid mesh
- No ID buffer exists
- Elevation can be sampled from heightmap texture

### Current Readback Infrastructure

**`src/core/async_readback.rs`**:
- `AsyncReadbackHandle`: async texture/buffer readback
- `ReadbackBuffer`: double-buffered MAP_READ buffers
- Supports texture → buffer copy → CPU read
- **Latency:** 1-2 frames for async readback

**No picking system exists** — must be built from scratch.

### Subsystem Map (Picking-Relevant)

| Subsystem | File | Key Symbols | Picking Relevance |
|-----------|------|-------------|-------------------|
| Vector Overlay | `src/viewer/terrain/vector_overlay.rs` | `VectorVertex`, `VectorOverlayStack` | Source of pickable features |
| Terrain Scene | `src/viewer/terrain/scene.rs` | `ViewerTerrainData` | Terrain elevation queries |
| Async Readback | `src/core/async_readback.rs` | `AsyncReadbackHandle` | ID buffer readback |
| Input Handling | `src/viewer/input/mouse.rs` | Mouse events | Click/hover source |
| Depth Buffer | `src/viewer/terrain/render.rs` | `depth_texture`, `depth_view` | Depth for ray intersection |

---

## Plan 1: MVP — ID Buffer Rendering + Readback

### 1. Goal
Implement click-to-select via an offscreen ID buffer that encodes feature IDs. On click, readback the ID at cursor position and highlight the selected feature.

### 2. Scope and Non-Goals

**In scope:**
- Offscreen R32Uint ID buffer (same resolution as main)
- Render overlays with encoded feature ID instead of color
- Single-pixel readback on click
- Highlight selected feature (color tint or outline)
- Python callback with feature ID

**Non-goals:**
- Hover highlighting (requires per-frame readback)
- Terrain picking (elevation queries)
- Lasso/box selection
- BVH-based ray picking

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Click Event (mouse.rs)                      │
│                    (x, y) screen coords                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PickingManager                               │
│  - Request ID buffer readback at (x, y)                         │
│  - Map pixel → feature ID                                       │
│  - Invoke Python callback                                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
           ┌─────────┴─────────┐
           ▼                   ▼
┌──────────────────┐  ┌────────────────────────────────────────────┐
│  ID Render Pass  │  │           Async Readback                   │
│  (R32Uint FBO)   │  │  copy_texture_to_buffer → map → callback   │
└──────────────────┘  └────────────────────────────────────────────┘
```

**Reused:** `AsyncReadbackHandle`, mouse input handling
**New:** ID buffer, ID render pass, PickingManager

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/picking/mod.rs` | `PickingManager`, `PickResult` |
| New | `src/picking/id_buffer.rs` | `IdBufferPass`, ID render pipeline |
| Modify | `src/viewer/terrain/vector_overlay.rs` | Add `feature_id: u32` to vertex or instance |
| New | `src/shaders/id_buffer.wgsl` | Output feature ID as R32Uint |
| Modify | `src/viewer/input/mouse.rs` | Forward click to PickingManager |
| Modify | `src/viewer/render/main_loop.rs` | Render ID pass when picking enabled |
| PyO3 | `src/lib.rs` | `set_pick_callback()`, `get_selected_feature()` |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| ID Buffer | R32Uint | 1920×1080×4 = 8.3 MiB | 33 MiB | Per-pixel feature ID |
| ID Depth | Depth32Float | 8.3 MiB | 33 MiB | Depth test for ID pass |
| Readback Buffer | MAP_READ | 4 bytes | 4 bytes | Single pixel |

**Total VRAM impact:** ~17 MiB @ 1080p, ~66 MiB @ 4K

### 6. Shader Changes (WGSL)

**New file:** `src/shaders/id_buffer.wgsl`

```wgsl
struct IdVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) feature_id: u32,  // Packed into vertex or instance
}

struct IdVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) feature_id: u32,
}

@vertex
fn vs_id(in: IdVertexInput) -> IdVertexOutput {
    var out: IdVertexOutput;
    out.clip_position = u_view_proj * vec4(in.position, 1.0);
    out.feature_id = in.feature_id;
    return out;
}

@fragment
fn fs_id(in: IdVertexOutput) -> @location(0) u32 {
    return in.feature_id;
}
```

**Lighting integration:** None — ID buffer is unlit, purely for identification.

### 7. User-Facing API

**Rust:**
```rust
pub struct PickingManager { ... }
impl PickingManager {
    pub fn pick_at(&mut self, x: u32, y: u32) -> Option<PickResult>;
    pub fn set_highlight(&mut self, feature_id: Option<u32>);
}

pub struct PickResult {
    pub feature_id: u32,
    pub screen_pos: (u32, u32),
    pub world_pos: Option<Vec3>,  // If depth available
}
```

**Python:**
```python
def on_pick(result):
    print(f"Selected feature: {result.feature_id}")
    print(f"Attributes: {layer.get_attributes(result.feature_id)}")

viewer.set_pick_callback(on_pick)
viewer.set_picking_enabled(True)  # default: False (no ID buffer overhead)

# Query current selection
selected = viewer.get_selected_feature()
```

**Config keys:**
- `picking.enabled` (bool, default: `false`)
- `picking.highlight_color` (vec4, default: `[1.0, 0.8, 0.0, 0.5]`)

### 8. Quality & Determinism

- **Precision:** R32Uint supports 4 billion unique IDs
- **Edge cases:** ID=0 reserved for "no feature" (background/terrain)
- **Determinism:** Same click position always returns same ID
- **Failure modes:**
  - Readback timeout → return None, log warning
  - ID buffer resize lag → use cached size, update next frame

### 9. Validation & Tests

**Test:** `tests/test_picking_mvp.py`
```python
def test_pick_polygon():
    """Click on polygon returns correct feature ID."""
    layer = viewer.add_polygon_layer(gdf)  # gdf has 3 polygons
    viewer.set_picking_enabled(True)
    
    # Click center of polygon 1
    center = polygon_screen_center(gdf.iloc[1])
    result = viewer.pick_at(center.x, center.y)
    
    assert result is not None
    assert result.feature_id == 1

def test_pick_background():
    """Click on empty area returns None."""
    result = viewer.pick_at(0, 0)  # corner, likely empty
    assert result is None or result.feature_id == 0

def test_highlight_selection():
    """Selected feature is visually highlighted."""
    viewer.pick_at(center.x, center.y)
    img = viewer.snapshot()
    # Check that feature 1 has highlight tint
    assert feature_has_highlight(img, feature_id=1)
```

**Commands:**
```bash
python -m pytest tests/test_picking_mvp.py -v
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | ID Buffer Setup | `src/picking/id_buffer.rs` | R32Uint texture, depth | Buffer created at correct size | Format support |
| M2 | ID Vertex Data | `src/viewer/terrain/vector_overlay.rs` | Feature ID in vertex/instance | IDs encoded correctly | Vertex format change |
| M3 | ID Render Pass | `src/shaders/id_buffer.wgsl`, pipeline | Render IDs to buffer | Buffer contains correct IDs | Pipeline creation |
| M4 | Single-pixel Readback | `src/picking/mod.rs` | Async readback at click | ID returned in <2 frames | Readback timing |
| M5 | Highlight Rendering | Main shader modification | Tint selected feature | Visual feedback | Highlight visibility |
| M6 | Python Callback | `src/lib.rs` | `set_pick_callback()` | Callback invoked on pick | Thread safety |

---

## Plan 2: Standard — GPU Ray Picking + Hover Support

### 1. Goal
Implement GPU-accelerated ray picking without full-frame ID buffer rendering. Support hover highlighting with minimal latency using a small readback region.

### 2. Scope and Non-Goals

**In scope:**
- Ray-based picking using view-space geometry intersection
- Hover support via cursor tracking + small-region ID readback
- Terrain elevation query at pick point
- Attribute panel data retrieval
- Multi-select (shift+click)

**Non-goals:**
- BVH acceleration (use simple bounds check)
- Lasso selection
- Complex attribute editing

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Mouse Move / Click                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RayPickingEngine                              │
│  - Unproject cursor to ray (origin + direction)                 │
│  - Test ray against layer bounding boxes                        │
│  - For candidates: render small ID tile (64×64)                 │
│  - Readback center pixel                                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
           ┌─────────┴─────────┐
           ▼                   ▼
┌──────────────────┐  ┌────────────────────────────────────────────┐
│ Tile ID Render   │  │         Terrain Heightfield Query          │
│ (64×64 R32Uint)  │  │  Sample depth → reconstruct world pos      │
└──────────────────┘  └────────────────────────────────────────────┘
```

**Advantages over Plan 1:**
- No full-resolution ID buffer (much less VRAM)
- Hover support with acceptable latency (~16ms for 64×64)
- Terrain picking via depth reconstruction

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/picking/ray.rs` | `RayPickingEngine`, `unproject_cursor()` |
| New | `src/picking/bounds.rs` | `LayerBounds`, AABB intersection |
| New | `src/picking/tile_id.rs` | `TileIdPass`, small ID render |
| New | `src/picking/terrain_query.rs` | `query_terrain_at()`, depth→world |
| Modify | `src/viewer/input/mouse.rs` | Track cursor, throttle hover |
| New | `src/picking/selection.rs` | `SelectionSet`, multi-select |

### 5. GPU Resources and Formats

| Resource | Format | Size | Notes |
|----------|--------|------|-------|
| Tile ID Buffer | R32Uint | 64×64×4 = 16 KiB | Reused each query |
| Tile Depth | Depth32Float | 16 KiB | For tile render |
| Readback Buffer | MAP_READ | 4 bytes | Single pixel |
| Layer Bounds | CPU | ~1 KiB/layer | AABB per layer |

**Total VRAM impact:** ~32 KiB (constant, regardless of resolution)

### 6. Shader Changes (WGSL)

**Tile ID shader** — same as Plan 1, but rendered to smaller target.

**Terrain depth reconstruction:**
```wgsl
fn reconstruct_world_from_depth(screen_uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4(screen_uv * 2.0 - 1.0, depth, 1.0);
    let world_h = inverse_view_proj * ndc;
    return world_h.xyz / world_h.w;
}
```

### 7. User-Facing API

**Python:**
```python
# Hover callback (throttled to 60Hz)
def on_hover(result):
    if result:
        viewer.show_tooltip(f"Feature: {result.feature_id}")

viewer.set_hover_callback(on_hover)
viewer.set_hover_enabled(True)

# Terrain query
info = viewer.query_terrain(screen_x, screen_y)
print(f"Elevation: {info.elevation}m, Slope: {info.slope}°")

# Multi-select
viewer.set_multi_select(True)  # shift+click adds to selection
selected_ids = viewer.get_selection()  # returns list
```

**Config keys:**
- `picking.hover_enabled` (bool, default: `false`)
- `picking.hover_delay_ms` (int, default: `100`)
- `picking.tile_size` (int, default: `64`)

### 8. Quality & Determinism

- **Hover latency:** ~16-32ms for 64×64 tile render + readback
- **Precision:** Tile centered on cursor; sub-pixel accuracy not needed
- **Stability:** Throttle hover to avoid GPU thrashing
- **Failure modes:**
  - All layers outside frustum → skip picking
  - Depth buffer unavailable → fall back to Plan 1 approach

### 9. Validation & Tests

**Test:** `tests/test_picking_standard.py`
```python
def test_hover_callback():
    """Hover over feature triggers callback."""
    hover_results = []
    viewer.set_hover_callback(lambda r: hover_results.append(r))
    viewer.set_hover_enabled(True)
    
    # Simulate mouse move over feature
    viewer.simulate_mouse_move(feature_center.x, feature_center.y)
    time.sleep(0.2)  # Wait for hover delay
    
    assert len(hover_results) > 0
    assert hover_results[-1].feature_id == expected_id

def test_terrain_query():
    """Query terrain elevation at cursor."""
    info = viewer.query_terrain(512, 384)
    assert info.elevation > 0
    assert 0 <= info.slope <= 90

def test_multi_select():
    """Shift+click adds to selection."""
    viewer.set_multi_select(True)
    viewer.pick_at(pos1.x, pos1.y)
    viewer.pick_at(pos2.x, pos2.y, shift=True)
    
    selection = viewer.get_selection()
    assert len(selection) == 2
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Ray Unprojection | `src/picking/ray.rs` | Cursor → world ray | Ray origin/dir correct | Matrix inversion |
| M2 | Bounds Testing | `src/picking/bounds.rs` | AABB per layer | Correct layers identified | Bounds update on pan/zoom |
| M3 | Tile ID Render | `src/picking/tile_id.rs` | 64×64 ID buffer | IDs correct in tile | Viewport setup |
| M4 | Hover Throttle | `src/viewer/input/mouse.rs` | Rate-limited hover | No GPU thrashing | Timer accuracy |
| M5 | Terrain Query | `src/picking/terrain_query.rs` | Depth → elevation | Correct elevation | Depth precision |
| M6 | Multi-select | `src/picking/selection.rs` | Selection set | Add/remove works | UI feedback |
| M7 | Attribute Panel | Python side | Show feature attrs | Attrs displayed | Data binding |

---

## Plan 3: Premium — Unified Picking with BVH + Python Callbacks

### 1. Goal
Full-featured picking system with BVH-accelerated ray intersection, unified terrain+overlay picking, selection sets, lasso selection, and rich Python callback integration.

### 2. Scope and Non-Goals

**In scope:**
- GPU BVH for overlay geometry (reuse `src/accel/lbvh_gpu.rs`)
- Ray-heightfield intersection for terrain
- Lasso/box selection
- Selection sets with add/remove/toggle
- Rich Python callbacks with full feature attributes
- Highlight styles (outline, glow, color tint)

**Non-goals:**
- Editing feature geometry
- Attribute editing in viewer
- 3D picking through transparent surfaces

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Events                               │
│  Click / Drag (lasso) / Hover                                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               UnifiedPickingSystem                              │
│  - Ray generation from cursor                                   │
│  - BVH traversal for overlays (GPU compute)                     │
│  - Heightfield ray-march for terrain                            │
│  - Lasso polygon → frustum test                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
           ┌─────────┴─────────────┬─────────────────┐
           ▼                       ▼                 ▼
┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐
│   BVH Traverse   │  │  Heightfield March │  │  Lasso Frustum   │
│   (GPU compute)  │  │  (GPU compute)     │  │  (CPU/GPU)       │
└──────────────────┘  └────────────────────┘  └──────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               SelectionManager                                  │
│  - Selection sets (named groups)                                │
│  - Highlight style per set                                      │
│  - Python callback dispatch                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/picking/unified.rs` | `UnifiedPickingSystem` |
| Extend | `src/accel/lbvh_gpu.rs` | Add ray-BVH query shader |
| New | `src/picking/heightfield_ray.rs` | Ray-heightfield intersection |
| New | `src/picking/lasso.rs` | `LassoSelection`, frustum culling |
| New | `src/picking/selection_manager.rs` | Named selection sets |
| New | `src/picking/highlight.rs` | Outline, glow, tint rendering |
| New | `src/shaders/bvh_ray_query.wgsl` | BVH traversal compute |
| New | `src/shaders/heightfield_ray.wgsl` | Ray-march compute |

### 5. GPU Resources and Formats

| Resource | Format | Size | Notes |
|----------|--------|------|-------|
| BVH Nodes | Storage buffer | ~2 MiB / 100k tris | From lbvh_gpu |
| Ray Query Results | R32Uint×N | 4 KiB | Hit feature IDs |
| Lasso Mask | R8Unorm | 1080p: 2 MiB | Screen-space lasso |
| Highlight Mask | R8Unorm | 2 MiB | Selection outline |

**Total VRAM impact:** ~6-10 MiB depending on geometry complexity

### 6. Shader Changes (WGSL)

**BVH Ray Query (compute):**
```wgsl
@compute @workgroup_size(64)
fn cs_bvh_ray_query(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let ray_idx = gid.x;
    if (ray_idx >= ray_count) { return; }
    
    let ray = rays[ray_idx];
    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[stack_ptr] = 0u;  // root
    stack_ptr += 1u;
    
    var closest_t = 1e30;
    var closest_id = 0u;
    
    while (stack_ptr > 0u) {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];
        
        if (!ray_aabb_intersect(ray, node.aabb)) { continue; }
        
        if (node.is_leaf) {
            // Test triangles
            let hit = ray_triangle_intersect(ray, node.prim_idx);
            if (hit.t < closest_t) {
                closest_t = hit.t;
                closest_id = node.feature_id;
            }
        } else {
            stack[stack_ptr] = node.left; stack_ptr += 1u;
            stack[stack_ptr] = node.right; stack_ptr += 1u;
        }
    }
    
    results[ray_idx] = closest_id;
}
```

**Heightfield Ray March:**
```wgsl
fn ray_heightfield_intersect(ray: Ray, heightmap: texture_2d<f32>) -> f32 {
    // Binary search along ray for height crossing
    var t = 0.0;
    let max_t = 1000.0;
    let step = 1.0;
    
    while (t < max_t) {
        let p = ray.origin + ray.dir * t;
        let h = textureSampleLevel(heightmap, samp, world_to_uv(p), 0.0).r;
        if (p.z < h) {
            // Binary refinement
            return refine_intersection(ray, t - step, t);
        }
        t += step;
    }
    return -1.0;  // no hit
}
```

### 7. User-Facing API

**Python:**
```python
# Rich pick result
@dataclass
class PickResult:
    feature_id: int
    layer_name: str
    world_pos: tuple[float, float, float]
    attributes: dict  # Full feature attributes
    terrain_info: TerrainInfo | None  # If terrain hit

# Callbacks with full context
def on_pick(results: list[PickResult], event: PickEvent):
    for r in results:
        print(f"Picked {r.layer_name}:{r.feature_id} at {r.world_pos}")
        print(f"Attributes: {r.attributes}")

viewer.set_pick_callback(on_pick)

# Selection sets
viewer.create_selection_set("highlighted", color=(1, 0.8, 0, 0.5))
viewer.add_to_selection("highlighted", feature_ids=[1, 2, 3])
viewer.set_selection_style("highlighted", outline=True, glow=True)

# Lasso selection
viewer.set_lasso_mode(True)
# User draws lasso...
selected = viewer.get_lasso_selection()  # Returns IDs inside lasso

# Terrain query with slope/aspect
info = viewer.query_terrain_detailed(x, y)
print(f"Elevation: {info.elevation}, Slope: {info.slope}°, Aspect: {info.aspect}°")
```

### 8. Quality & Determinism

- **BVH precision:** Exact triangle intersection
- **Heightfield precision:** Binary refinement to <0.1m error
- **Lasso:** Screen-space polygon contains test
- **Determinism:** BVH traversal order is deterministic
- **Performance:** <5ms for single-ray BVH query, <1ms for heightfield

### 9. Validation & Tests

**Test:** `tests/test_picking_premium.py`
```python
def test_bvh_ray_query():
    """BVH returns correct feature for ray."""
    viewer.add_mesh_layer(complex_mesh)  # 100k triangles
    
    # Cast ray through known feature
    result = viewer.pick_at(feature_center.x, feature_center.y)
    assert result.feature_id == expected_id
    assert result.attributes["name"] == "Expected Feature"

def test_lasso_selection():
    """Lasso selects features inside polygon."""
    viewer.set_lasso_mode(True)
    viewer.simulate_lasso([p1, p2, p3, p4])  # Draw rectangle
    
    selected = viewer.get_lasso_selection()
    assert set(selected) == {1, 2, 5}  # Features inside

def test_heightfield_intersection():
    """Ray-heightfield returns correct elevation."""
    terrain_result = viewer.query_terrain_detailed(512, 384)
    
    # Compare with ground-truth elevation at that point
    expected_elev = dem.sample(terrain_result.world_pos[:2])
    assert abs(terrain_result.elevation - expected_elev) < 0.1

def test_selection_sets():
    """Multiple selection sets with different styles."""
    viewer.create_selection_set("primary", color="yellow")
    viewer.create_selection_set("secondary", color="blue")
    
    viewer.add_to_selection("primary", [1, 2])
    viewer.add_to_selection("secondary", [3, 4])
    
    img = viewer.snapshot()
    assert feature_has_color(img, 1, "yellow")
    assert feature_has_color(img, 3, "blue")
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | BVH Integration | `src/picking/unified.rs`, `src/accel/lbvh_gpu.rs` | Ray-BVH query | Correct hits | BVH rebuild on geometry change |
| M2 | BVH Query Shader | `src/shaders/bvh_ray_query.wgsl` | GPU BVH traversal | <5ms query | Stack overflow |
| M3 | Heightfield Ray | `src/picking/heightfield_ray.rs` | Ray-terrain intersection | <0.1m precision | Step size tuning |
| M4 | Lasso Selection | `src/picking/lasso.rs` | Screen-space lasso | Correct containment | Complex polygons |
| M5 | Selection Sets | `src/picking/selection_manager.rs` | Named sets + styles | Multiple active | Memory for large sets |
| M6 | Highlight Styles | `src/picking/highlight.rs` | Outline, glow | Visible highlights | Outline shader |
| M7 | Python Integration | `src/lib.rs` | Full attribute access | Callbacks work | GIL handling |

---

## ID-to-Attribute Mapping Strategy

**Current overlay representation lacks feature IDs.** Solutions:

1. **Instance buffer approach** (recommended):
   - Add `feature_id: u32` to `VectorVertex` or use instance buffer
   - ID assigned at layer creation, sequential per layer
   - Python maintains `layer.features[id]` → GeoDataFrame row

2. **Texture lookup approach** (for raster overlays):
   - Encode feature ID in overlay texture alpha/extra channel
   - Query texture at pick point

3. **CPU geometry approach** (fallback):
   - Store geometry on CPU, perform ray-polygon intersection
   - Slower but no GPU changes needed

---

## Recommendation

**Start with Plan 1 (MVP)** for basic click-to-select functionality (~1.5 weeks). This enables interactive feature inspection with minimal complexity. Upgrade to Plan 2 if hover highlighting is required. Plan 3 is for GIS-grade applications needing lasso selection and complex queries.
