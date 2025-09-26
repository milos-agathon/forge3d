# GPT-5-CODEX-MEDIUM TASK

# Title: Workstream F — Geometry & IO (implement & wire 18 tasks per phased plan)

# Mode: Implementor • Medium autonomy • Small, atomic commits • Follow steps in order

## REPO

* Root: .
* Create branch: `wsF-geometry-io-integration`
* Paths to use only:

  * `src/`
  * `shaders/`
  * `python/forge3d/`
  * `examples/`
  * `docs/`
  * `tests/`

## CONSTRAINTS

* Platforms: **win_amd64**, **linux_x86_64**, **macos_universal2**
* GPU budget: **≤512 MiB** host-visible heap (prefer simplified test meshes, compact attributes)
* Toolchain: **cmake≥3.24**, **cargo/rustc**, **PyO3**, **VMA**
* APIs: **WebGPU/WGSL** primary; Vulkan 1.2–compatible design
* Docs: **Sphinx**
* Safety: do **NOT** touch `build/`, `dist/`, `target/`, `_build/`, `.venv/`, large binary assets; **no blind search/replace**

## WORKSTREAM F TASKS (scope only)

* **F1**  Polygon Extrusion (P0)
* **F2**  OSM Building Footprints
* **F3**  Thick Polylines 3D
* **F4**  Import OBJ + MTL
* **F5**  Export to OBJ
* **F6**  Export STL (3D Print)
* **F7**  MultipolygonZ → OBJ
* **F8**  Consistent Normals & Weld
* **F9**  Primitive Mesh Library
* **F10** UV Unwrap Helpers
* **F11** Subdivision Surface
* **F12** Displacement Modifiers
* **F13** PLY Import
* **F14** Mesh Transform Ops
* **F15** Mesh Info & Validate
* **F16** Mesh Instancing (Raster)
* **F17** Curve & Tube Primitives
* **F18** glTF 2.0 Import (+Draco)  **[depends on F4, F9, and B15 (IBL assets); do NOT implement B15 here]**

---

## PLAN — EXECUTE IN ORDER (Phased by weeks from report)

### 0) Scaffolding (tests/docs)

* Add pytest markers **“geometry”** and **“io”**.
* Create Sphinx stubs for Geometry & IO in `docs/api/`.
* **Commit**: “WS-F: test/docs scaffolding”

---

### PHASE 1 — Core Foundation (Weeks 1–3)

**1) F1 — Polygon Extrusion (Week 1; P0)**

* GPU prism mesh generation for 3D regions/buildings; triangulate caps & sides.
* **AC**: 10k polygons extruded in ≤50 ms; valid normals/UVs; watertight where expected.
* **Files**: `src/geometry/extrude.rs`, `tests/test_f1_extrude.py`, `examples/f1_extrude_demo.py`

**2) F9 — Primitive Mesh Library (Week 1–2; P1)**

* Unit primitives (plane, box, sphere, cylinder, cone, torus) + text3D stub; proper normals/UVs.
* **AC**: Unit scale; orientation & winding consistent; parametric tesselation.
* **Files**: `src/geometry/primitives.rs`, `tests/test_f9_primitives.py`, `examples/f9_primitives_demo.py`

**3) F15 — Mesh Info & Validate (Week 2; P1)**

* Diagnostics: bbox/stats, non-manifold/zero-area checks, degenerate triangles, duplicate verts.
* **AC**: Reports match references; detects injected pathologies in tests.
* **Files**: `src/geometry/validate.rs`, `tests/test_f15_validate.py`

**4) F8 — Consistent Normals & Weld (Week 3; P1)**

* Import-time weld, unified winding, smooth group recompute; epsilon controls; deterministic.
* **AC**: No shading seams on primitives; stable hashes across runs.
* **Files**: `src/geometry/weld.rs`, `tests/test_f8_weld.py`

---

### PHASE 2 — Basic I/O Operations (Weeks 4–6)

**5) F14 — Mesh Transform Ops (Week 4; P1)**

* Center/scale/flip/swap utilities; transform normals/tangents/UVs; preserve winding validity.
* **AC**: Valid winding & normals after transforms.
* **Files**: `src/geometry/transform.rs`, `tests/test_f14_transform.py`

**6) F4 — Import OBJ + MTL (Week 4–5; P1)**

* Robust OBJ reader with MTL materials; fast path for large meshes; material groups/submeshes.
* **AC**: 1M triangles ≤200 ms on reference preset; correct materials & smoothing groups.
* **Files**: `src/io/obj_read.rs`, `tests/test_f4_obj_import.py`, `examples/f4_obj_import_demo.py`

**7) F5 — Export to OBJ (Week 5; P1)**

* Triangle/UV/normal export; material library generation; round-trip with F4.
* **AC**: Opens in Blender; topology & materials round-trip.
* **Files**: `src/io/obj_write.rs`, `tests/test_f5_obj_export.py`

**8) F13 — PLY Import (Week 6; P1)**

* Read ASCII/binary; attributes: position/normal/uv/color; endian-safe; large files stream.
* **AC**: BBox/attributes match references; color fidelity within tolerance.
* **Files**: `src/io/ply_read.rs`, `tests/test_f13_ply_import.py`

---

### PHASE 3 — Specialized Export Formats (Weeks 7–9)

**9) F6 — Export STL (3D Print) (Week 7; P1)**

* Binary STL export; enforce watertightness check or document if not.
* **AC**: Netfabb manifold validation passes.
* **Files**: `src/io/stl_write.rs`, `tests/test_f6_stl_export.py`

**10) F7 — MultipolygonZ → OBJ (Week 7–8; P1)**

* Vector-to-mesh conversion: extrude MultiPolygonZ with materials; outward normals.
* **AC**: Opens correctly in Blender; outward normals validated.
* **Files**: `src/converters/multipolygonz_to_obj.rs`, `tests/test_f7_multipolygonz_obj.py`

**11) F10 — UV Unwrap Helpers (Week 8–9; P1)**

* Planar/spherical projections; texel density controls; seam marking helpers.
* **AC**: Valid seams; uniform checkerboard appearance.
* **Files**: `src/uv/unwrap.rs`, `tests/test_f10_uv_unwrap.py`, `examples/f10_uv_unwrap_demo.py`

---

### PHASE 4 — Advanced Mesh Processing (Weeks 10–12)

**12) F11 — Subdivision Surface (Week 10–11; P1)**

* Loop/Catmull–Clark with UV/crease preservation; adaptive levels; crack-free stitching.
* **AC**: Convergent limit surface; no cracks on patch boundaries.
* **Files**: `src/geometry/subdivision.rs`, `tests/test_f11_subdivision.py`, `examples/f11_subdivision_demo.py`

**13) F12 — Displacement Modifiers (Week 11–12; P1)**

* Heightmap/procedural displacement; tangent-space normal updates; bounds growth.
* **AC**: No tile seams; tangent normals updated; bounds expanded correctly.
* **Files**: `src/geometry/displacement.rs`, `tests/test_f12_displacement.py`, `examples/f12_displacement_demo.py`

**14) F17 — Curve & Tube Primitives (Week 12; P1)**

* Bezier/path ribbons and tubes with caps/joins; **width_end** parameter; miter/round/bevel joins.
* **AC**: Smooth curves; `width_end` honored; no self-intersection in standard cases.
* **Files**: `src/geometry/curves.rs`, `tests/test_f17_curves.py`, `examples/f17_curves_demo.py`

---

### PHASE 5 — Specialized 3D Features (Weeks 13–15)

**15) F3 — Thick Polylines 3D (Week 13; P1)**

* 3D roads/edges using quads/cylinders; constant pixel width; z-fight guards.
* **AC**: Constant pixel width across FOV; no z-fighting in stress scenes.
* **Files**: `src/geometry/thick_polyline.rs`, `tests/test_f3_thick_polyline.py`, `examples/f3_thick_polyline_demo.py`

**16) F2 — OSM Building Footprints (Week 14; P1)**

* Ingest OSM footprints; map attributes → extrusion heights; city demo pipeline.
* **AC**: ≥30 FPS downtown @1080p with shadows on reference preset.
* **Files**: `src/import/osm_buildings.rs`, `examples/f2_city_demo.py`, `tests/test_f2_osm_buildings.py`

---

### PHASE 6 — Performance & Modern Formats (Weeks 16–18)

**17) F16 — Mesh Instancing (Raster) (Week 16–17; P1)**

* Hardware instancing + indirect draw; per-instance transforms/material IDs.
* **AC**: 10k instances ≥60 FPS @1080p (low-VRAM profile).
* **Files**: `src/render/instancing.rs`, `tests/test_f16_instancing.py`, `examples/f16_instancing_demo.py`

**18) F18 — glTF 2.0 Import (+Draco) (Week 17–18; P1; depends on F4, F9, B15 assets)**

* glTF 2.0 reader; materials/transforms/animations subset; optional Draco decompression.
* **AC**: Sample models load with materials/transforms; ≥40% disk savings with Draco where available.
* **Files**: `src/io/gltf_read.rs`, `tests/test_f18_gltf_import.py`, `examples/f18_gltf_import_demo.py`

---

## DOCS (finalize)

* `docs/api/geometry.md`, `docs/api/io.md`: how-to per feature, round-trip notes, performance presets, attribute schemas.
* **Commit**: “WS-F: Geometry & IO docs (Sphinx)”

## EXAMPLES (finalize)

Ensure runnable examples exist & use **≤512 MiB** presets:

* `examples/f1_extrude_demo.py`
* `examples/f4_obj_import_demo.py`
* `examples/f5_obj_export_demo.py`
* `examples/f6_stl_export_demo.py`
* `examples/f7_multipolygonz_obj.py`
* `examples/f9_primitives_demo.py`
* `examples/f10_uv_unwrap_demo.py`
* `examples/f11_subdivision_demo.py`
* `examples/f12_displacement_demo.py`
* `examples/f16_instancing_demo.py`
* `examples/f17_curves_demo.py`
* `examples/f18_gltf_import_demo.py`
* **Commit**: “WS-F: runnable examples (low-VRAM presets)”

---

## TEST MATRIX / ACCEPTANCE CRITERIA (must pass)

* **F1**: 10k polys extruded ≤50 ms; valid normals/UVs; watertight as configured.
* **F2**: City demo ≥30 FPS @1080p with shadows (preset).
* **F3**: Constant pixel width; no z-fighting in stress scenes.
* **F4**: 1M tris ≤200 ms; correct materials/groups.
* **F5**: Blender opens; round-trip topology & materials match.
* **F6**: Netfabb manifold validation passes.
* **F7**: Outward normals; Blender import OK.
* **F8**: No seams; deterministic weld hashes.
* **F9**: Unit scale; normals/UVs correct across primitives.
* **F10**: Uniform checker density; seams valid.
* **F11**: No cracks; limit surface convergence verified.
* **F12**: No tile seams; tangent normals updated; bounds expansion correct.
* **F13**: BBox/attributes match; color fidelity ok.
* **F14**: Post-transform normals/winding valid.
* **F15**: Detects non-manifold/zero-area/degenerate/duplicates.
* **F16**: 10k instances ≥60 FPS @1080p (preset).
* **F17**: Smooth curves; `width_end` honored; joins valid.
* **F18**: Sample glTF models load; Draco saves ≥40% disk; materials/transforms correct.

---

## FILES TO CREATE/TOUCH (non-exhaustive)

* `src/geometry/`: `extrude.rs`, `primitives.rs`, `validate.rs`, `weld.rs`, `transform.rs`, `subdivision.rs`, `displacement.rs`, `curves.rs`, `thick_polyline.rs`
* `src/io/`: `obj_read.rs`, `obj_write.rs`, `ply_read.rs`, `stl_write.rs`, `gltf_read.rs`
* `src/import/`: `osm_buildings.rs`
* `src/converters/`: `multipolygonz_to_obj.rs`
* `src/uv/`: `unwrap.rs`
* `src/render/`: `instancing.rs`
* `python/forge3d/`: `geometry.py` (APIs), `io.py` (APIs)
* `examples/`: as listed above
* `docs/api/`: `geometry.md`, `io.md`
* `tests/`: `test_f1_extrude.py`, `test_f2_osm_buildings.py`, `test_f3_thick_polyline.py`, `test_f4_obj_import.py`, `test_f5_obj_export.py`, `test_f6_stl_export.py`, `test_f7_multipolygonz_obj.py`, `test_f8_weld.py`, `test_f9_primitives.py`, `test_f10_uv_unwrap.py`, `test_f11_subdivision.py`, `test_f12_displacement.py`, `test_f13_ply_import.py`, `test_f14_transform.py`, `test_f15_validate.py`, `test_f16_instancing.py`, `test_f17_curves.py`, `test_f18_gltf_import.py`

---

## RUNBOOK (use locally in this order)

```bash
git checkout -b wsF-geometry-io-integration
cargo fmt --check
cargo clippy --all-targets --all-features -D warnings
cargo test -q
pytest -q
pytest -k "geometry or io" -v
sphinx-build -b html docs _build/html
maturin build --release
python examples/f1_extrude_demo.py
python examples/f18_gltf_import_demo.py
```

## DONE WHEN

* All tests above pass on Windows/Linux/macOS runners.
* Examples render/headless within **≤512 MiB** host-visible heap.
* Round-trip IO verified (OBJ/PLY/STL/glTF where applicable).
* Docs build clean; APIs surfaced for key operations.

## GIT HYGIENE

* One feature per commit (code + tests + minimal docs tweak).
* Clear messages prefixed **“WS-F: …”**.
* No vendored binaries or large assets.

## OUTPUT

* Provide final summary of changed files, passing test counts, and a short demo log for examples (first/last 3 lines).
