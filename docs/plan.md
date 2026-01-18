# forge3d Roadmap – Implementation-Ready Plan

This document provides an implementation-ready plan for the next 10 features, with concrete structs, functions, file paths verified against the current codebase, acceptance tests, and stop-conditions per AGENTS.md.

---

## Existing Capabilities Summary

| Capability | Status | Key Files (verified) |
|------------|--------|----------------------|
| **Presets/Config** | Complete | `python/forge3d/presets.py`, `python/forge3d/config.py`, `examples/presets/*.json` |
| **Labels/MSDF** | Complete | `src/labels/mod.rs`, `src/labels/types.rs` (LabelStyle, LabelData), `src/labels/layer.rs` |
| **Water Surface** | Basic | `src/core/water_surface.rs` – waves, reflections, foam stub |
| **3D Tiles** | Parsing | `src/tiles3d/renderer.rs` (Tiles3dRenderer), `src/tiles3d/b3dm.rs`, `src/tiles3d/pnts.rs` |
| **OSM Buildings** | Basic | `src/import/osm_buildings.rs` – GeoJSON extrusion |
| **GPU Indirect** | Partial | `src/vector/indirect.rs` – IndirectRenderer |
| **CRS Metadata** | Stored | `python/forge3d/io.py::DEMData.crs` (no reprojection) |
| **Viewer IPC** | Complete | `python/forge3d/viewer_ipc.py`, `src/viewer/ipc/protocol.rs` (IpcRequest enum) |
| **Memory Budget** | Exists | `src/render/memory_budget.rs` (GpuMemoryBudget), `src/util/memory_budget.rs` (constants) |
| **Picking** | Exists | `src/picking/mod.rs` (PickingManager), `src/viewer/input/viewer_input.rs` |

---

## Priority 1: Scene Bundle (`.forge3d`)

**Impact 5 / Effort 2–3** – Reproducible handoff artifact.

### Checklist

- [ ] **1.1 Define bundle schema**
  - **Files**: `src/bundle/mod.rs` (NEW ~100 LOC), `src/bundle/manifest.rs` (NEW ~150 LOC)
  - **Structs**:
    ```rust
    // src/bundle/manifest.rs
    pub struct BundleManifest {
        pub version: u32,           // Schema version (1)
        pub name: String,
        pub created_at: String,     // ISO 8601
        pub checksums: HashMap<String, String>,  // path -> sha256
    }
    ```
  - **Stop-condition**: `BundleManifest` round-trips through serde_json without data loss.

- [ ] **1.2 Python SceneBundle class**
  - **File**: `python/forge3d/bundle.py` (NEW ~200 LOC)
  - **Functions**:
    ```python
    def save_bundle(path: Path, terrain: TerrainData, overlays: list, preset: dict) -> None
    def load_bundle(path: Path) -> tuple[TerrainData, list, dict]
    ```
  - **Stop-condition**: `load_bundle(save_bundle(data))` returns identical data.

- [ ] **1.3 IPC commands for bundle**
  - **Files**: `src/viewer/ipc/protocol.rs` (+20 LOC), `src/viewer/cmd/handler.rs` (+50 LOC)
  - **New IpcRequest variants**:
    ```rust
    SaveBundle { path: String },
    LoadBundle { path: String },
    ```
  - **Wire**: Add to `ipc_request_to_viewer_cmd()` mapping.
  - **Stop-condition**: IPC round-trip `SaveBundle` → `LoadBundle` restores viewer state.

- [ ] **1.4 CLI flags**
  - **File**: `examples/terrain_demo.py` (+30 LOC)
  - **Flags**: `--save-bundle PATH`, `--load-bundle PATH`
  - **Precedence**: CLI `--load-bundle` overrides `--preset`; explicit CLI args override bundle preset.
  - **Stop-condition**: `terrain_demo.py --load-bundle test.forge3d` renders without error.

### Schema (v1)
```
.forge3d/
├── manifest.json
├── terrain/
│   ├── dem.tif
│   └── colormap.json
├── overlays/
│   ├── vectors.geojson
│   └── labels.json
├── camera/
│   └── bookmarks.json
├── render/
│   └── preset.json
└── assets/
    └── hdri/
```

### Acceptance Tests
- `tests/test_bundle_roundtrip.py`:
  - Assert `save_bundle` → `load_bundle` → render produces identical PNG hash.
  - Assert manifest version field == 1.
- `tests/test_bundle_cli.py`:
  - Smoke test: `--save-bundle` creates directory with manifest.json.

### Dependencies
- None new (uses `serde_json`, `std::fs`).

---

## Priority 2: Mapbox Style Spec Import

**Impact 5 / Effort 3–4** – Ecosystem compatibility.

### Checklist

- [ ] **2.1 Style spec parser**
  - **Files**: `src/style/mod.rs` (NEW ~50 LOC), `src/style/parser.rs` (NEW ~250 LOC), `src/style/types.rs` (NEW ~100 LOC)
  - **Structs**:
    ```rust
    // src/style/types.rs
    pub struct StyleLayer {
        pub id: String,
        pub layer_type: LayerType,  // fill, line, symbol
        pub paint: PaintProps,
        pub layout: LayoutProps,
        pub filter: Option<FilterExpr>,
    }
    pub enum LayerType { Fill, Line, Symbol, Background }
    ```
  - **Stop-condition**: Parse Mapbox Streets v8 style.json without panic.

- [ ] **2.2 Property converters**
  - **File**: `src/style/converters.rs` (NEW ~150 LOC)
  - **Functions**:
    ```rust
    pub fn paint_to_vector_style(paint: &PaintProps) -> VectorStyle
    pub fn layout_to_label_style(layout: &LayoutProps) -> LabelStyle
    ```
  - **Stop-condition**: Converted styles produce non-default values for color/width.

- [ ] **2.3 Python API**
  - **File**: `python/forge3d/style.py` (NEW ~150 LOC)
  - **Functions**:
    ```python
    def load_style(path: Path) -> StyleSpec
    def apply_style(spec: StyleSpec, vectors: VectorScene) -> None
    ```
  - **Stop-condition**: `load_style("mapbox-streets.json")` returns StyleSpec with ≥1 layer.

- [ ] **2.4 Integration with render pipeline**
  - **Files**: `python/forge3d/render.py` (+30 LOC), `src/labels/layer.rs` (+20 LOC)
  - **Stop-condition**: `render_polygons(..., style=load_style(...))` applies style colors.

### Supported Properties (v1)
- **fill**: `fill-color`, `fill-opacity`, `fill-outline-color`
- **line**: `line-color`, `line-width`, `line-opacity`
- **symbol**: `text-field`, `text-size`, `text-color`, `text-halo-color`, `text-halo-width`
- **Filters**: `["==", "property", "value"]`, `["all", ...]`, `["any", ...]`

### Acceptance Tests
- `tests/test_style_parser.py`:
  - Parse Mapbox basic style, assert ≥5 layers extracted.
  - Filter expression evaluation returns correct bool.
- `tests/test_style_render.py`:
  - Render with style vs. default, assert pixel diff > 0.

### Dependencies
- None new (JSON parsing).

---

## Priority 3: On-the-fly Reprojection (PROJ)

**Impact 4 / Effort 3** – CRS-agnostic loading.

### Checklist

- [ ] **3.1 Add proj crate (feature-gated)**
  - **File**: `Cargo.toml` (+5 LOC)
  - **Feature**: `proj` (optional, default off)
    ```toml
    [features]
    proj = ["dep:proj"]
    
    [dependencies]
    proj = { version = "0.27", optional = true }
    ```
  - **Stop-condition**: `cargo build --features proj` succeeds on Linux/macOS/Windows.

- [ ] **3.2 Rust reprojection helper**
  - **Files**: `src/geo/mod.rs` (NEW ~30 LOC), `src/geo/reproject.rs` (NEW ~120 LOC)
  - **Functions**:
    ```rust
    #[cfg(feature = "proj")]
    pub fn reproject_coords(
        coords: &[[f64; 2]],
        from_crs: &str,
        to_crs: &str,
    ) -> Result<Vec<[f64; 2]>, GeoError>
    ```
  - **Stop-condition**: WGS84→UTM→WGS84 round-trip error < 1e-6 degrees.

- [ ] **3.3 Python wrapper**
  - **File**: `python/forge3d/crs.py` (NEW ~100 LOC)
  - **Functions**:
    ```python
    def reproject_geom(geom, target_crs: str) -> Geometry
    def transform_coords(coords: np.ndarray, from_crs: str, to_crs: str) -> np.ndarray
    ```
  - **Fallback**: Use `pyproj` if Rust `proj` feature unavailable.
  - **Stop-condition**: Both paths produce identical results within 1e-6.

- [ ] **3.4 Auto-reproject in render_polygons**
  - **File**: `python/forge3d/render.py` (+40 LOC)
  - **Logic**: If `polygon_crs != terrain_crs`, call `reproject_geom()`.
  - **Stop-condition**: Vector overlay with WGS84 coords aligns with UTM terrain.

- [ ] **3.5 Expose terrain CRS**
  - **Files**: `python/forge3d/terrain_params.py` (+10 LOC)
  - **Add**: `terrain_crs: Optional[str]` field to `TerrainParams`.
  - **Stop-condition**: `params.terrain_crs` returns DEM's CRS string.

### Acceptance Tests
- `tests/test_crs_reproject.py`:
  - Round-trip WGS84↔UTM, assert max error < 1e-6.
  - Assert fallback to pyproj works when Rust feature disabled.
- `tests/test_crs_auto.py`:
  - Render WGS84 vectors on UTM terrain, assert alignment (overlay centroid within terrain bounds).

### Dependencies
- Rust: `proj = "0.27"` (optional, feature-gated)
- Python: `pyproj` (fallback, already available)

---

## Priority 4: 3D Buildings Pipeline

**Impact 5 / Effort 4** – Cities on terrain.

### Existing Foundation
- `src/import/osm_buildings.rs` – basic GeoJSON extrusion
- `src/geometry/extrude.rs` – polygon extrusion
- `src/tiles3d/renderer.rs` – Tiles3dRenderer with cache

### Checklist

- [ ] **4.1 Roof type inference**
  - **File**: `src/import/osm_buildings.rs` (+80 LOC)
  - **Enum**:
    ```rust
    pub enum RoofType { Flat, Gabled, Hipped, Pyramidal }
    pub fn infer_roof_type(tags: &HashMap<String, String>) -> RoofType
    ```
  - **Stop-condition**: OSM `building:roof:shape=gabled` → `RoofType::Gabled`.

- [ ] **4.2 Material presets**
  - **File**: `src/import/building_materials.rs` (NEW ~120 LOC)
  - **Structs**:
    ```rust
    pub struct BuildingMaterial {
        pub albedo: [f32; 3],
        pub roughness: f32,
        pub metallic: f32,
    }
    pub fn material_from_tags(tags: &HashMap<String, String>) -> BuildingMaterial
    ```
  - **Stop-condition**: `building:material=brick` → reddish albedo.

- [ ] **4.3 CityJSON parser**
  - **File**: `src/import/cityjson.rs` (NEW ~250 LOC)
  - **Function**:
    ```rust
    pub fn parse_cityjson(data: &[u8]) -> Result<Vec<BuildingGeom>, ImportError>
    ```
  - **Stop-condition**: Parse CityJSON 1.1 sample file, extract ≥1 building.

- [ ] **4.4 Wire to terrain scene**
  - **Files**: `src/tiles3d/renderer.rs` (+50 LOC)
  - **Method**: `Tiles3dRenderer::render_buildings(&self, buildings: &[BuildingGeom], ...)`
  - **Stop-condition**: Buildings visible in terrain render.

- [ ] **4.5 Python API**
  - **File**: `python/forge3d/buildings.py` (NEW ~180 LOC)
  - **Functions**:
    ```python
    def add_buildings(geojson_path: Path, **opts) -> BuildingLayer
    def add_buildings_3dtiles(tileset_path: Path) -> BuildingLayer
    ```
  - **Stop-condition**: `add_buildings("buildings.geojson")` returns layer with vertex_count > 0.

### Acceptance Tests
- `tests/test_buildings_extrude.py`:
  - Extrude simple polygon, assert vertex count == expected.
- `tests/test_buildings_materials.py`:
  - Assert `material=brick` produces distinct albedo from `material=concrete`.
- `tests/test_buildings_cityjson.py`:
  - Parse sample CityJSON, assert building count matches expected.

### Dependencies
- None new (CityJSON is JSON).

---

## Priority 5: Vector Export (SVG/PDF)

**Impact 4 / Effort 3–4** – Print-grade overlays.

### Checklist

- [ ] **5.1 SVG writer**
  - **File**: `src/export/svg.rs` (NEW ~200 LOC)
  - **Function**:
    ```rust
    pub fn vectors_to_svg(
        polygons: &[Polygon],
        lines: &[LineString],
        view_transform: &Mat4,
        width: u32, height: u32,
    ) -> String
    ```
  - **Stop-condition**: Output is valid SVG (parseable by xml parser).

- [ ] **5.2 Label export**
  - **File**: `src/export/svg_labels.rs` (NEW ~120 LOC)
  - **Function**:
    ```rust
    pub fn labels_to_svg_text(labels: &[LabelData], view_transform: &Mat4) -> String
    ```
  - **Stop-condition**: SVG contains `<text>` elements with correct positions.

- [ ] **5.3 PDF wrapper (Python)**
  - **File**: `python/forge3d/export.py` (NEW ~100 LOC)
  - **Functions**:
    ```python
    def export_svg(scene: VectorScene, path: Path, include_labels: bool = True) -> None
    def export_pdf(scene: VectorScene, path: Path, dpi: int = 300) -> None
    ```
  - **Stop-condition**: PDF file is readable by PDF viewer.

- [ ] **5.4 Camera projection**
  - **File**: `src/export/projection.rs` (NEW ~80 LOC)
  - **Function**:
    ```rust
    pub fn project_3d_to_2d(point: Vec3, view_proj: &Mat4, viewport: (u32, u32)) -> (f32, f32)
    ```
  - **Stop-condition**: Known 3D point projects to expected 2D coords ± 1px.

### Acceptance Tests
- `tests/test_export_svg.py`:
  - Export simple polygon, assert SVG contains `<polygon>` element.
  - Assert viewBox dimensions match requested size.
- `tests/test_export_projection.py`:
  - Project cube corners, assert screen coords within viewport.

### Dependencies
- Rust: None (string generation)
- Python: `svgwrite` (optional, for PDF via cairosvg)

---

## Priority 6: Viewer Authoring Tools

**Impact 4 / Effort 4** – Interactive production tool.

### Existing Foundation
- `src/viewer/ipc/protocol.rs` – IpcRequest enum with label commands
- `src/labels/mod.rs` – LabelManager
- `src/picking/mod.rs` – PickingManager
- `python/forge3d/viewer_ipc.py` – `add_label()`, `remove_label()`

### Checklist

- [ ] **6.1 Label picking**
  - **File**: `src/picking/mod.rs` (+80 LOC)
  - **Method**: `PickingManager::pick_label(&self, screen_pos: (f32, f32)) -> Option<LabelId>`
  - **Logic**: Check label screen rects against click position.
  - **Stop-condition**: Clicking on label returns its ID; clicking empty returns None.

- [ ] **6.2 Label drag**
  - **File**: `src/viewer/input/viewer_input.rs` (+60 LOC)
  - **Logic**: On drag start with picked label, track delta; on drag end, update label position.
  - **Stop-condition**: Label position changes after drag gesture.

- [ ] **6.3 IPC queries for style**
  - **File**: `src/viewer/ipc/protocol.rs` (+30 LOC)
  - **New IpcRequest variants**:
    ```rust
    GetLabelStyle { id: u64 },
    SetLabelStyle { id: u64, style: LabelStyleUpdate },
    ```
  - **Stop-condition**: `GetLabelStyle` returns serialized style JSON.

- [ ] **6.4 Annotation/callout via IPC**
  - **Files**: `src/viewer/ipc/protocol.rs` (already has `AddCallout`), verify handler
  - **Stop-condition**: `AddCallout` IPC creates visible callout with leader line.

- [ ] **6.5 Save to bundle (wire to P1)**
  - **Depends on**: Priority 1.3
  - **Stop-condition**: `SaveBundle` IPC exports current viewer state.

### Acceptance Tests
- `tests/test_viewer_label_pick.py`:
  - Add label, send pick IPC at label center, assert label ID returned.
- `tests/test_viewer_authoring.py`:
  - Add label → move via IPC → get position → assert changed.

---

## Priority 7: Web Export (WASM/WebGPU)

**Impact 5 / Effort 5** – Browser distribution.

### Checklist

- [ ] **7.1 WASM feature gate**
  - **File**: `Cargo.toml` (+20 LOC)
  - **Feature**: `wasm`
    ```toml
    [features]
    wasm = ["wasm-bindgen", "web-sys", "console_error_panic_hook"]
    
    [target.'cfg(target_arch = "wasm32")'.dependencies]
    wasm-bindgen = "0.2"
    web-sys = { version = "0.3", features = ["Window", "Document", "HtmlCanvasElement"] }
    console_error_panic_hook = "0.1"
    ```
  - **Stop-condition**: `cargo build --target wasm32-unknown-unknown --features wasm` compiles.

- [ ] **7.2 Browser bundle loader**
  - **File**: `web/src/loader.ts` (NEW ~150 LOC)
  - **Function**: `async function loadBundle(url: string): Promise<SceneData>`
  - **Stop-condition**: Fetches and unpacks .forge3d bundle in browser.

- [ ] **7.3 Viewer shell**
  - **File**: `web/src/viewer.ts` (NEW ~300 LOC), `web/src/controls.ts` (NEW ~150 LOC)
  - **Stop-condition**: Canvas renders terrain mesh (may be placeholder).

- [ ] **7.4 CI build**
  - **File**: `.github/workflows/ci.yml` (+30 LOC)
  - **Job**: `build-wasm` with `wasm-pack build --target web`
  - **Stop-condition**: CI job passes on PR.

- [ ] **7.5 npm package**
  - **Files**: `web/package.json` (NEW), `web/demo/index.html` (NEW)
  - **Stop-condition**: `npm run build` produces `dist/forge3d.js`.

### Acceptance Tests
- `tests/test_wasm_build.py`:
  - Assert `cargo build --target wasm32-unknown-unknown --features wasm` exit code 0.
- Browser E2E (Playwright):
  - Load demo page, assert canvas is visible.

### Dependencies
- `wasm-bindgen`, `web-sys`, `console_error_panic_hook` (feature-gated)

---

## Priority 8: GPU-Driven Rendering Everywhere

**Impact 4 / Effort 4–5** – Massive scale.

### Existing Foundation
- `src/vector/indirect.rs` – IndirectRenderer
- `src/render/memory_budget.rs` – GpuMemoryBudget (EXISTS, not new!)
- `src/util/memory_budget.rs` – budget constants
- `src/tiles3d/traversal.rs` – TilesetTraverser

### Checklist

- [ ] **8.1 Unified indirect draw**
  - **File**: `src/vector/indirect.rs` (+150 LOC)
  - **Method**: `IndirectRenderer::draw_all(&self, ...)`
  - **Stop-condition**: Single draw call renders 10k+ primitives.

- [ ] **8.2 Culling shader polish**
  - **File**: `src/shaders/culling_compute.wgsl` (if exists, else NEW ~150 LOC)
  - **Stop-condition**: Frustum-culled draw count < total count for off-screen objects.

- [ ] **8.3 Wire budget to renderer**
  - **File**: `src/render/memory_budget.rs` (+50 LOC)
  - **Method**: `GpuMemoryBudget::enforce_limit(&mut self, allocations: &mut Vec<Allocation>)`
  - **Stop-condition**: Allocation rejected when budget exceeded.

- [ ] **8.4 Integrate with 3D Tiles**
  - **File**: `src/tiles3d/traversal.rs` (+30 LOC)
  - **Stop-condition**: Tiles3dRenderer uses indirect draw path.

### Acceptance Tests
- `tests/test_gpu_indirect_perf.py`:
  - Render 100k points, assert frame time < 100ms.
- `tests/test_memory_budget.py`:
  - Request allocation exceeding budget, assert rejected.

---

## Priority 9: Advanced Water & Coastal Rendering

**Impact 4 / Effort 4** – Signature polish.

### Existing Foundation
- `src/core/water_surface.rs` – basic waves, reflections
- `src/shaders/water_surface.wgsl` – shader

### Checklist

- [ ] **9.1 Shore distance field**
  - **File**: `src/core/water_surface.rs` (+80 LOC)
  - **Method**: `WaterSurface::compute_shore_distance(&self, terrain_mask: &Texture)`
  - **Stop-condition**: Distance texture has gradient from shore.

- [ ] **9.2 Procedural foam**
  - **File**: `src/shaders/water_surface.wgsl` (+100 LOC)
  - **Stop-condition**: Foam visible within `foam_width` of shore.

- [ ] **9.3 Depth-based color**
  - **File**: `src/shaders/water_surface.wgsl` (+40 LOC)
  - **Stop-condition**: Shallow water lighter than deep water.

- [ ] **9.4 Wind-driven waves**
  - **File**: `src/core/water_surface.rs` (+80 LOC)
  - **Stop-condition**: Wave direction changes with wind parameter.

- [ ] **9.5 SSR integration**
  - **Files**: `src/p5/ssr.rs` (+30 LOC), `src/core/water_surface.rs` (+20 LOC)
  - **Stop-condition**: Water shows SSR reflections when enabled.

### Acceptance Tests
- `tests/test_water_foam.py`:
  - Render water with shore, assert foam pixels (high luminance) near edge.
- `tests/test_water_depth.py`:
  - Compare shallow vs deep ROI, assert luminance difference > threshold.

---

## Priority 10: Semantic Styling + Material Synthesis

**Impact 4 / Effort 4–5** – Atlas-grade automation.

### Checklist

- [ ] **10.1 Rule engine**
  - **Files**: `src/style/rules.rs` (NEW ~200 LOC), `src/style/rule_types.rs` (NEW ~80 LOC)
  - **Structs**:
    ```rust
    pub struct StyleRule {
        pub predicate: Predicate,
        pub style: StyleOutput,
    }
    pub fn evaluate_rules(rules: &[StyleRule], feature: &Feature) -> StyleOutput
    ```
  - **Stop-condition**: Rule `landcover == 'forest'` matches forest features only.

- [ ] **10.2 Landcover→material presets**
  - **File**: `src/style/landcover_materials.rs` (NEW ~150 LOC)
  - **Stop-condition**: `landcover='water'` returns blue color with reflection.

- [ ] **10.3 Scale-dependent styling**
  - **File**: `src/style/scale.rs` (NEW ~120 LOC)
  - **Stop-condition**: Road width changes at zoom threshold.

- [ ] **10.4 Python DSL**
  - **File**: `python/forge3d/semantic_style.py` (NEW ~180 LOC)
  - **Class**: `StyleRules` with `add(predicate, **style)` method.
  - **Stop-condition**: Python rules serialize to JSON matching Rust parser.

### Acceptance Tests
- `tests/test_semantic_rules.py`:
  - Define 3 rules, apply to mixed features, assert correct style per feature.
- `tests/test_scale_dependent.py`:
  - Render at zoom 8 vs zoom 12, assert line width differs.

---

## Implementation Order Summary

| # | Feature | Est. LOC | Key Risk | Blocking | Cargo Feature |
|---|---------|----------|----------|----------|---------------|
| 1 | Scene Bundle | ~500 | Schema stability | None | – |
| 2 | Style Spec | ~550 | Mapbox spec complexity | None | – |
| 3 | Reprojection | ~350 | PROJ linkage | None | `proj` |
| 4 | Buildings | ~630 | CityJSON edge cases | None | – |
| 5 | Vector Export | ~500 | 3D→2D projection | None | – |
| 6 | Authoring | ~200 | UX polish | #1 | – |
| 7 | Web Export | ~650 | WASM+WebGPU compat | #1 | `wasm` |
| 8 | GPU-Driven | ~380 | GPU memory | None | – |
| 9 | Water Polish | ~350 | Visual tuning | None | – |
| 10 | Semantic Style | ~730 | Rule language | #2 | – |

---

## Constraints per AGENTS.md

1. **Files <300 LOC**: Split files exceeding ~300 lines into modules.
2. **Tests in Python only**: All acceptance tests in `tests/`, never Rust tests.
3. **Forced-impact tests**: Each feature must have test proving measurable output change.
4. **Preserve baselines**: New features default off; presets pin old behavior.
5. **CLI precedence**: Explicit CLI args > preset values > defaults.
6. **Feature-gated deps**: Optional crates (proj, wasm) behind Cargo features.
7. **No breaking changes**: New APIs additive; existing signatures unchanged.
8. **Stop-conditions**: Each milestone has explicit pass/fail criteria.
9. **Update docs**: Public API changes require `docs/api/` updates.