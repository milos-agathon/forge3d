# Feature A: Interactive Classic Printed-Atlas Labels

## Investigation Summary

### Terrain Representation Answer: **HYBRID**

**Evidence:**
1. `src/terrain/mesh.rs::GridVertex` — 2D position `[f32; 2]` + UV, no Z stored
2. `src/terrain/mesh.rs::make_grid()` — generates vertex/index buffers for grid
3. `src/terrain/renderer.rs::ViewerTerrainData` — stores both `heightmap_texture` AND `vertex_buffer`/`index_buffer`
4. `src/shaders/terrain_pbr_pom.wgsl:1021-1044` — VS samples `height_tex` to compute world Z: `h_disp = textureSampleLevel(height_tex, ...)`
5. `src/shaders/terrain_pbr_pom.wgsl:959-1059` — Two camera modes: screen (fullscreen tri) and mesh (grid + view*proj)
6. `src/viewer/terrain/scene.rs::ViewerTerrainData` — confirms dual storage pattern

### Subsystem Map

| Subsystem | File Paths | Key Symbols | Notes |
|-----------|------------|-------------|-------|
| **Text/2D UI** | `src/core/text_overlay.rs` | `TextOverlayRenderer`, `TextInstance`, `TextOverlayUniforms` | MSDF placeholder; screen-space quads; no atlas yet |
| **Vector Overlay** | `src/viewer/terrain/vector_overlay.rs` | `VectorVertex`, `VectorOverlayStack`, `drape_vertices` | Lit, draped geometry; points/lines/polygons |
| **Terrain Renderer** | `src/terrain/renderer.rs`, `src/viewer/terrain/render.rs` | `TerrainScene`, `ViewerTerrainScene` | PBR+POM pipeline |
| **Lighting/Shadows** | `src/shadows/csm_renderer.rs`, `src/shaders/terrain_pbr_pom.wgsl` | `CsmRenderer`, `sample_shadow_terrain()` | CSM with PCF/PCSS |
| **Readback** | `src/core/async_readback.rs` | `AsyncReadbackHandle`, `ReadbackBuffer` | Double-buffered async |
| **Camera** | `src/camera/mod.rs`, `src/viewer/terrain/scene.rs` | `camera_look_at()`, `cam_phi_deg/theta_deg/radius/fov_deg` | Orbit camera state |

### Current Text Rendering State
- `TextOverlayRenderer` exists but is a **placeholder** (no glyph atlas, no MSDF font loading)
- Uses `TextInstance` with `rect_min/max`, `uv_min/max`, `color`
- Pipeline supports alpha blending over scene color
- **No font loading, no glyph metrics, no collision avoidance**

---

## Plan 1: MVP — SDF Text Atlas + Screen-Space Placement

### 1. Goal
Render static text labels in screen-space using pre-generated SDF/MSDF atlases with basic grid-based collision avoidance and depth occlusion testing against terrain.

### 2. Scope and Non-Goals
**In scope:**
- Load pre-generated MSDF font atlases (BMFont format or similar)
- Screen-space label placement with pixel coordinates
- Simple grid collision (reject overlapping labels)
- Depth-based occlusion (sample depth buffer, fade/hide if occluded)
- Basic styling: color, size, optional thin halo

**Non-goals:**
- Curved/line-following labels
- Complex decluttering algorithms
- Runtime font rasterization
- Multi-language shaping

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                               │
│  add_label(text, lat/lon, style) → label_id                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LabelManager (Rust)                          │
│  - Label storage (id → LabelData)                               │
│  - World-to-screen projection                                   │
│  - Grid collision detection                                     │
│  - Occlusion query (depth sample)                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               TextOverlayRenderer (existing)                    │
│  - MSDF atlas texture                                           │
│  - Instance buffer (TextInstance)                               │
│  - Fragment shader: MSDF sampling + halo                        │
└─────────────────────────────────────────────────────────────────┘
```

**Reused:** `TextOverlayRenderer` pipeline, `TextInstance` struct
**Extended:** Add glyph atlas loading, collision grid, depth sampling

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| Extend | `src/core/text_overlay.rs` | Add `load_msdf_atlas()`, `GlyphMetrics` |
| New | `src/labels/mod.rs` | `LabelManager`, `LabelData`, `LabelStyle` |
| New | `src/labels/collision.rs` | `CollisionGrid`, grid-based rejection |
| New | `src/labels/projection.rs` | World→screen projection, depth test |
| Modify | `src/viewer/render/main_loop.rs` | Call `label_manager.update()` + render |
| New | `src/shaders/msdf_label.wgsl` | MSDF sampling with halo support |
| PyO3 | `src/lib.rs` | Expose `add_label()`, `remove_label()`, `set_label_style()` |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| MSDF Atlas | Rgba8Unorm | 2048×2048 = 16 MiB | Same | Pre-generated, one per font |
| Instance Buffer | Dynamic | 64 bytes × 1000 = 64 KiB | Same | Grows with label count |
| Collision Grid | CPU only | ~100 KiB | ~400 KiB | 10×10 px cells |
| Depth readback | None | 0 | 0 | Sample in shader |

**Total VRAM impact:** ~17 MiB per font family

### 6. Shader Changes (WGSL)

**New file:** `src/shaders/msdf_label.wgsl`

```wgsl
// Bind group 0
@group(0) @binding(0) var<uniform> u_label: LabelUniforms; // resolution, alpha, smoothing
@group(0) @binding(1) var atlas_tex: texture_2d<f32>;
@group(0) @binding(2) var atlas_samp: sampler;
@group(0) @binding(3) var depth_tex: texture_2d<f32>; // scene depth for occlusion

struct LabelUniforms {
    resolution: vec2<f32>,
    smoothing: f32,
    halo_width: f32,
    halo_color: vec4<f32>,
}
```

**Lighting integration:** Labels are unlit (screen-space overlay), but fade based on terrain occlusion. No sun/shadow interaction.

### 7. User-Facing API

**Rust (internal):**
```rust
pub struct LabelManager { ... }
impl LabelManager {
    pub fn add_label(&mut self, text: &str, world_pos: Vec3, style: LabelStyle) -> LabelId;
    pub fn remove_label(&mut self, id: LabelId);
    pub fn set_label_style(&mut self, id: LabelId, style: LabelStyle);
    pub fn update(&mut self, view_proj: Mat4, depth_view: &TextureView);
}
```

**Python:**
```python
viewer.add_label("Mt. Rainier", lat=46.85, lon=-121.76, style={"size": 14, "color": "black"})
viewer.remove_label(label_id)
viewer.set_labels_enabled(True)  # default: True
```

**Config/Preset keys:**
- `labels.enabled` (bool, default: `true`)
- `labels.default_font` (string, default: `"atlas"`)
- `labels.collision_grid_size` (int, default: `10`)

### 8. Quality & Determinism

- **Anti-aliasing:** MSDF provides sub-pixel AA via signed distance
- **Camera stability:** Labels re-project each frame; no jitter if positions are stable
- **Determinism:** Grid collision order depends on insertion order (deterministic)
- **Failure modes:**
  - Missing atlas → error on load, no labels rendered
  - Too many labels → grid rejects overflow, log warning

### 9. Validation & Tests

**Test:** `tests/test_labels_mvp.py`
```python
def test_label_renders():
    """Single label renders at expected screen position."""
    viewer.add_label("Test", world_pos=(0, 0, 100))
    img = viewer.snapshot()
    # Assert non-empty pixels near expected screen coords
    assert label_visible_at(img, expected_x=512, expected_y=384, tolerance=20)

def test_label_occlusion():
    """Label behind terrain is faded/hidden."""
    # Place label behind a mountain peak
    viewer.add_label("Hidden", world_pos=(0, 0, -100))
    img = viewer.snapshot()
    assert label_alpha_below(img, threshold=0.3)
```

**Commands:**
```bash
python -m pytest tests/test_labels_mvp.py -v
```

### 10. Milestones & Deliverables

| # | Name | Files Touched | Deliverables | Acceptance Criteria | Risks |
|---|------|---------------|--------------|---------------------|-------|
| M1 | MSDF Atlas Loading | `src/core/text_overlay.rs`, `src/labels/atlas.rs` | Load BMFont `.fnt` + `.png` | Atlas texture created, glyph metrics parsed | Font format compatibility |
| M2 | LabelManager Core | `src/labels/mod.rs`, `src/labels/projection.rs` | Add/remove labels, world→screen | Labels project to correct screen coords | Coordinate system mismatch |
| M3 | Collision Grid | `src/labels/collision.rs` | Grid-based rejection | No overlapping labels in output | Grid cell size tuning |
| M4 | Depth Occlusion | `src/shaders/msdf_label.wgsl` | Sample depth, fade occluded | Labels behind terrain fade | Depth buffer access timing |
| M5 | Halo Rendering | `src/shaders/msdf_label.wgsl` | MSDF halo in shader | Thin white halo around dark text | Halo width tuning |
| M6 | Python API | `src/lib.rs`, `python/forge3d/` | `add_label()`, `remove_label()` | Python test passes | API ergonomics |

---

## Plan 2: Standard — Cartographic Rules Engine

### 1. Goal
Implement a cartographic labeling system with priority-based placement, scale-dependent visibility, line labeling, and terrain-aware occlusion with horizon fade.

### 2. Scope and Non-Goals

**In scope:**
- Priority system (higher priority labels placed first)
- Scale-dependent visibility (min/max zoom levels)
- Line labeling (labels along polylines, horizontal or angled)
- Improved occlusion (horizon-aware fade, not just depth)
- Halos, underlines, small-caps styling
- Leader lines for offset labels

**Non-goals:**
- Curved text along complex paths
- Simulated annealing decluttering
- Multi-language shaping (complex scripts)

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                               │
│  add_label(..., priority=, min_zoom=, max_zoom=)               │
│  add_line_label(geometry, text, placement="along")             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CartographicLabelEngine                         │
│  - Priority queue                                               │
│  - Scale filtering                                              │
│  - Placement candidates (point, line-center, line-along)        │
│  - Collision detection (R-tree or grid)                         │
│  - Occlusion + horizon fade                                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│               Extended TextOverlayRenderer                      │
│  - Multi-style instance buffer                                  │
│  - Leader line geometry                                         │
│  - Rotation support (angled labels)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Reused:** `TextOverlayRenderer`, `LabelManager` from Plan 1
**Extended:** Priority queue, R-tree collision, line labeling logic

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/labels/engine.rs` | `CartographicLabelEngine`, `PlacementCandidate` |
| New | `src/labels/priority.rs` | `PriorityQueue<LabelData>` |
| New | `src/labels/line_label.rs` | `compute_line_label_placement()` |
| New | `src/labels/rtree.rs` | `LabelRTree` for collision |
| Extend | `src/core/text_overlay.rs` | Add rotation to `TextInstance` |
| New | `src/labels/leader.rs` | `LeaderLineRenderer` |
| Modify | `src/shaders/msdf_label.wgsl` | Add rotation uniform, underline support |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| MSDF Atlas | Rgba8Unorm | 16 MiB | Same | Multiple fonts possible |
| Instance Buffer | Dynamic | 80 bytes × 2000 = 160 KiB | Same | +rotation, +flags |
| Leader Line VB | Dynamic | 32 bytes × 500 = 16 KiB | Same | Simple line segments |
| R-tree | CPU only | ~500 KiB | ~1 MiB | Label bounding boxes |

**Total VRAM impact:** ~17 MiB + minor buffers

### 6. Shader Changes (WGSL)

**Extended `TextInstance`:**
```wgsl
struct TextInstance {
    rect_min: vec2<f32>,
    rect_max: vec2<f32>,
    uv_min: vec2<f32>,
    uv_max: vec2<f32>,
    color: vec4<f32>,
    rotation: f32,      // radians
    flags: u32,         // bit 0: underline, bit 1: small-caps
    halo_color: vec4<f32>,
}
```

**Horizon fade:** Compute view angle to label; fade when angle approaches horizon threshold.

### 7. User-Facing API

**Python:**
```python
# Point label with priority
viewer.add_label("Capital City", lat=48.8, lon=2.3, 
                 priority=100, min_zoom=5, max_zoom=15,
                 style={"font": "serif", "size": 16, "small_caps": True})

# Line label along geometry
viewer.add_line_label(geometry=rail_line, text="Trans-Siberian Railway",
                      placement="along", repeat_distance=500)

# Leader line (offset label)
viewer.add_label("Peak: 4808m", world_pos=summit, 
                 offset=(20, -30), leader=True)
```

**Config keys:**
- `labels.max_visible` (int, default: `500`)
- `labels.horizon_fade_angle` (float, default: `5.0` degrees)
- `labels.line_label_spacing` (float, default: `200` pixels)

### 8. Quality & Determinism

- **Anti-aliasing:** MSDF + rotation-aware sampling
- **Stability:** Priority order is stable; same labels win each frame
- **Determinism:** R-tree order is deterministic given insertion order
- **Failure modes:**
  - All labels rejected → log warning, consider relaxing collision
  - Line too short → skip line label, place point label instead

### 9. Validation & Tests

**Test:** `tests/test_labels_cartographic.py`
```python
def test_priority_ordering():
    """Higher priority label wins collision."""
    viewer.add_label("Low", pos=A, priority=1)
    viewer.add_label("High", pos=A, priority=100)  # same position
    img = viewer.snapshot()
    assert "High" visible, "Low" not visible

def test_scale_visibility():
    """Label hidden outside zoom range."""
    viewer.add_label("City", pos=A, min_zoom=10, max_zoom=15)
    viewer.set_zoom(5)
    assert label_not_visible()
    viewer.set_zoom(12)
    assert label_visible()

def test_line_label():
    """Label placed along line geometry."""
    viewer.add_line_label(river_geom, "River Name")
    img = viewer.snapshot()
    assert text_follows_line(img, river_geom)
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Priority System | `src/labels/priority.rs` | Sort labels by priority | Higher priority placed first | None |
| M2 | Scale Filtering | `src/labels/engine.rs` | min/max zoom check | Labels appear/disappear at thresholds | Zoom metric definition |
| M3 | R-tree Collision | `src/labels/rtree.rs` | Replace grid with R-tree | Faster collision for many labels | R-tree crate selection |
| M4 | Line Labels | `src/labels/line_label.rs` | Place text along polylines | Text follows line direction | Curved line handling |
| M5 | Horizon Fade | `src/shaders/msdf_label.wgsl` | View-angle based fade | Labels near horizon fade smoothly | Threshold tuning |
| M6 | Leader Lines | `src/labels/leader.rs` | Offset labels with connectors | Leader rendered correctly | Z-fighting with terrain |
| M7 | Styling | Shader + API | Small-caps, underline, halos | Styles render correctly | Font metrics for small-caps |

---

## Plan 3: Premium — Atlas Typography with Curved Text

### 1. Goal
Full cartographic labeling system with curved text along complex polylines, annealing-based decluttering, callout boxes, and atlas-style typography controls (tracking, ligatures where supported).

### 2. Scope and Non-Goals

**In scope:**
- Curved text along Bézier/polyline paths
- Simulated annealing or greedy decluttering
- Callout boxes with background fill
- Typography controls: tracking (letter-spacing), kerning
- Multi-font support (serif for features, sans for labels)
- Batch labeling for vector layers

**Non-goals:**
- Complex script shaping (Arabic, Devanagari) — requires HarfBuzz
- Runtime font rasterization
- 3D extruded text

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python API                               │
│  LabelLayer(features, label_field, style_func)                 │
│  viewer.add_label_layer(layer)                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              AdvancedLabelEngine                                │
│  - Feature-to-label mapping                                     │
│  - Placement candidate generation (curved paths)                │
│  - Annealing/greedy declutter                                   │
│  - Multi-pass placement                                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CurvedTextRenderer                                 │
│  - Per-glyph transforms along path                              │
│  - Spline interpolation                                         │
│  - GPU instancing with per-glyph rotation                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              CalloutRenderer                                    │
│  - Background boxes with rounded corners                        │
│  - Pointer/arrow geometry                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Integration Points

| Action | File | Symbol |
|--------|------|--------|
| New | `src/labels/curved.rs` | `CurvedTextLayout`, `sample_path_at_offset()` |
| New | `src/labels/declutter.rs` | `AnnealingDeclutter`, `energy_function()` |
| New | `src/labels/callout.rs` | `CalloutRenderer`, `CalloutStyle` |
| New | `src/labels/layer.rs` | `LabelLayer`, batch labeling |
| Extend | `src/core/text_overlay.rs` | Per-glyph instance buffer |
| New | `src/shaders/curved_glyph.wgsl` | Per-glyph transform shader |

### 5. GPU Resources and Formats

| Resource | Format | Size @ 1080p | Size @ 4K | Notes |
|----------|--------|--------------|-----------|-------|
| MSDF Atlases | Rgba8Unorm | 32 MiB | Same | Multiple fonts |
| Per-glyph Buffer | Dynamic | 48 bytes × 10000 = 480 KiB | Same | Position, rotation, UV per glyph |
| Callout VB | Dynamic | 64 bytes × 200 = 13 KiB | Same | Rounded rect + pointer |
| Declutter state | CPU | ~2 MiB | ~4 MiB | Candidate positions |

**Total VRAM impact:** ~33 MiB

### 6. Shader Changes (WGSL)

**New: Per-glyph curved text**
```wgsl
struct GlyphInstance {
    world_pos: vec3<f32>,     // Position on path
    rotation: f32,            // Tangent angle
    uv_rect: vec4<f32>,       // Atlas UV
    color: vec4<f32>,
    scale: f32,
    path_offset: f32,         // Distance along path
}

@vertex
fn vs_curved_glyph(glyph: GlyphInstance, ...) -> VertexOutput {
    // Apply per-glyph rotation around center
    let rotated = rotate2d(local_pos, glyph.rotation);
    let world = glyph.world_pos + vec3(rotated, 0.0) * glyph.scale;
    // Project to clip space
}
```

### 7. User-Facing API

**Python:**
```python
# Batch labeling from GeoDataFrame
label_layer = LabelLayer(
    gdf,
    label_field="name",
    style={"font": "DejaVu Serif", "size": 12, "tracking": 0.05},
    placement="curved",  # or "point", "line"
    declutter="annealing"
)
viewer.add_label_layer(label_layer)

# Callout with background
viewer.add_callout(
    text="Elevation: 4808m\nFirst ascent: 1786",
    anchor=summit_pos,
    style={"background": "#ffffffcc", "border_radius": 4}
)

# Typography control
viewer.set_label_style(id, tracking=0.1, font="Garamond")
```

### 8. Quality & Determinism

- **Curved text:** Glyphs sampled at uniform arc-length intervals
- **Annealing:** Seeded RNG for reproducibility; same seed = same layout
- **Stability:** Cache placement results; only re-run on zoom/pan threshold
- **Failure modes:**
  - Path too short for text → truncate or reject
  - Annealing timeout → use best-so-far solution

### 9. Validation & Tests

**Test:** `tests/test_labels_premium.py`
```python
def test_curved_text_follows_bezier():
    """Curved label glyphs follow path tangent."""
    viewer.add_curved_label(bezier_path, "Curved River")
    img = viewer.snapshot()
    glyphs = extract_glyph_positions(img)
    for i, g in enumerate(glyphs[:-1]):
        tangent = compute_path_tangent(bezier_path, g.offset)
        assert angle_diff(g.rotation, tangent) < 5  # degrees

def test_annealing_determinism():
    """Same seed produces same layout."""
    viewer.set_declutter_seed(42)
    labels = add_many_labels(100)
    layout1 = viewer.get_label_positions()
    viewer.clear_labels()
    viewer.set_declutter_seed(42)
    add_many_labels(100)
    layout2 = viewer.get_label_positions()
    assert layout1 == layout2

def test_callout_renders():
    """Callout box with pointer renders correctly."""
    viewer.add_callout("Info", anchor=pos, offset=(50, -20))
    img = viewer.snapshot()
    assert callout_box_visible(img)
    assert pointer_connects_to_anchor(img, pos)
```

### 10. Milestones & Deliverables

| # | Name | Files | Deliverables | Acceptance | Risks |
|---|------|-------|--------------|------------|-------|
| M1 | Curved Path Layout | `src/labels/curved.rs` | Per-glyph positions along path | Glyphs follow tangent | Arc-length parameterization |
| M2 | Per-glyph Shader | `src/shaders/curved_glyph.wgsl` | Rotated glyph rendering | Curved text renders | Instance buffer size |
| M3 | Annealing Declutter | `src/labels/declutter.rs` | Energy minimization | Better placement than greedy | Performance (100ms budget) |
| M4 | Callout Boxes | `src/labels/callout.rs` | Rounded rect + pointer | Callouts render | Z-ordering |
| M5 | Label Layers | `src/labels/layer.rs` | Batch from GeoDataFrame | 1000+ labels handled | Memory |
| M6 | Typography | `src/labels/typography.rs` | Tracking, kerning | Letter-spacing works | Font metrics accuracy |
| M7 | Multi-font | `src/core/text_overlay.rs` | Multiple atlases | Serif + sans together | Atlas management |

---

## Font Packaging & Atlas Strategy

**Cross-platform approach:**
1. Ship pre-generated MSDF atlases (`.png` + `.fnt`) in `assets/fonts/`
2. Default fonts: DejaVu Sans, DejaVu Serif (open-source, good coverage)
3. Atlas generation offline via `msdf-atlas-gen` tool
4. Python helper to generate custom atlases: `forge3d.fonts.generate_atlas(ttf_path)`

**Printed-atlas styling:**
- Thin halos (1-2px) via MSDF outer distance
- Serif fonts for geographic features (mountains, rivers)
- Sans-serif for administrative labels
- Small-caps via separate atlas region or synthetic scaling
- Muted colors: dark gray (#333) text, white (#fff) halos

---

## Recommendation

**Start with Plan 1 (MVP)** for immediate value: basic labeled maps work with minimal investment (~2 weeks). Then evaluate need for cartographic rules (Plan 2) based on user feedback. Plan 3 is for production cartographic applications requiring print-quality output.
