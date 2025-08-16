## 0) Quick start with the repo

```bash
# In a fresh virtualenv:
pip install maturin

# From the repo root:
maturin develop --release

# Smoke test (creates triangle.png)
python -m examples.triangle
```

Tips:

* To force a backend during testing: `WGPU_BACKEND=metal|vulkan|dx12`.
* macOS needs Xcode CLT; Windows needs MSVC Build Tools (C++ workload).

---

## 1) Deliverables & milestones (Option B)

### Milestone A — Spikes (1–2 weeks total)

**A1. Headless rendering spike** ✅ *Done in the starter*

* **Exit criteria:** Produce a deterministic 512×512 PNG across OS backends; return an `H×W×4` `uint8` NumPy array.

Great—here’s a **fine‑grained task breakdown** for **Milestone A — Spikes → A1. Headless rendering spike** so your team can split work and hit the exit criteria:

> **Exit criteria (A1):**
>
> 1. Produce a **deterministic 512×512 PNG** (same bytes for repeated runs on the same machine; consistent across OS backends with our pinned stack; if a driver produces tiny differences, we allow a fallback SSIM ≥ 0.999).
> 2. Return an **H×W×4 `uint8` NumPy array** via the Python API.

---

## A1.1 Repository + build sanity (0.5d)

**Goal**: Everyone can build and run the starter consistently.

* **Deliverables**

  * Confirm local build: `maturin develop --release` works on macOS, Linux, Windows.
  * Document minimal toolchains (macOS CLT, Windows MSVC Build Tools, Python 3.10–3.12).

* **Acceptance**

  * `python -m examples.triangle` produces `triangle.png`.
  * Version pins committed (Rust toolchain in `rust-toolchain.toml` optional).

* **Notes**: Pin `pyo3`, `wgpu` minor versions in `Cargo.toml`.

---

## A1.2 Deterministic render pipeline setup (0.5d)

**Goal**: Eliminate non‑determinism in the pipeline.

* **Deliverables**

  * Ensure **no MSAA**, **no blending** changes (keep alpha blending default but opaque fragment output).
  * Fixed clear color (e.g., white), fixed viewport/scissor (0..W, 0..H).
  * Target format **`Rgba8UnormSrgb`** (consistent sRGB conversions).

* **Acceptance**

  * Repeated runs on a single machine produce identical PNG bytes (hash match).

* **Notes**: Keep shader math trivial (no time, no random, no derivatives that may vary with MSAA or precision modes).

---

## A1.3 Shader + geometry determinism (0.5d)

**Goal**: Stabilize the exact fragment coverage.

* **Deliverables**

  * Triangle with **canonical** positions/colors; no uniforms used.
  * WGSL uses pure linear interpolation; no branching on edge conditions.

* **Acceptance**

  * Edge pixels are consistent across runs; no flicker or aliasing differences.

* **Notes**: Use CCW winding, cull back face, and avoid degenerate triangles.

---

## A1.4 Off‑screen target & readback path (0.5d)

**Goal**: Correct copy‑to‑CPU with padding handled identically.

* **Deliverables**

  * Off‑screen texture creation (`Rgba8UnormSrgb`, `RENDER_ATTACHMENT|COPY_SRC`).
  * **Row‑padding** removal helper (already in starter): confirm with unit test.
  * Single persistent **readback buffer** sized to the frame (reuse across calls).

* **Acceptance**

  * A 512×512 render → RGBA array has shape `(512,512,4)` and dtype `uint8`.
  * Unit test validates padding logic on synthetic data.

---

## A1.5 Python API surface (0.25d)

**Goal**: Minimal public API for the spike.

* **Deliverables**

  * `Renderer(width, height)`, `render_triangle_rgba()`, `render_triangle_png(path)`, `info()`.
  * Type hints + docstrings.

* **Acceptance**

  * `help(vulkan_forge.Renderer)` shows clear signatures and docstrings.

---

## A1.6 Determinism harness (0.5d)

**Goal**: Automated checks for bitwise identity and quality fallback.

* **Deliverables**

  * Small Python script (and/or pytest) that:

    1. Renders twice → compares **SHA‑256** (expect equal).
    2. Optionally toggles backend via `WGPU_BACKEND` env (`metal|vulkan|dx12`) and repeats.
    3. If bytes differ, compute **SSIM** as a fallback—must be ≥ 0.999.
  * Prints adapter name/back‑end via `info()`.

* **Acceptance**

  * On at least one machine per OS, **two runs** produce identical hashes.
  * Across backends, either hashes match or SSIM ≥ 0.999.

* **Snippet**

  ```python
  import hashlib, os
  from vulkan_forge import Renderer
  def sha256_png_bytes(arr):
      from PIL import Image
      import io
      b = io.BytesIO(); Image.fromarray(arr, "RGBA").save(b, "PNG")
      return hashlib.sha256(b.getvalue()).hexdigest()

  # same-backend determinism
  r = Renderer(512,512)
  h1 = sha256_png_bytes(r.render_triangle_rgba())
  h2 = sha256_png_bytes(r.render_triangle_rgba())
  assert h1 == h2, "Non-deterministic within same run/backend"
  print("SHA256:", h1)
  ```

---

## A1.7 Cross‑backend runners (0.5d) (P)

**Goal**: Exercise Metal/Vulkan/DX12 explicitly.

* **Deliverables**

  * Three tiny scripts or a single script that sets `WGPU_BACKEND=metal|vulkan|dx12` and runs the determinism harness.
  * Readme section: “Forcing a backend”.

* **Acceptance**

  * Each backend available on the host runs and passes determinism criteria.

* **Notes**: On Windows, DX12; on Linux, Vulkan; on macOS, Metal. If a backend isn’t supported, test is skipped with a clear message.

---

## A1.8 CI matrix & artifacts (0.5–1d)

**Goal**: Prove the spike in CI.

* **Deliverables**

  * Update `.github/workflows/wheels.yml` to:

    * Build wheels (already present).
    * Run determinism harness on 512×512.
    * Upload `triangle-<backend>-<sha>.png` as artifacts.

* **Acceptance**

  * CI green on Ubuntu, Windows, macOS.
  * Artifacts show identical hashes per OS/backend runner on reruns.

---

## A1.9 Device diagnostics & failure modes (0.25d)

**Goal**: Useful messages when adapters/backends fail.

* **Deliverables**

  * `Renderer.report_device()` includes adapter name, backend, limits (max texture size).
  * Friendly error on “No suitable GPU adapter found” with tips:

    * Linux: ensure proper drivers; headless often works without a display.
    * Windows: update GPU driver.
    * macOS: CLT/Xcode installed.

* **Acceptance**

  * Running on a VM without a suitable backend shows actionable guidance.

---

## A1.10 Performance sanity (optional, 0.25d)

**Goal**: Ensure render + readback don’t regress.

* **Deliverables**

  * Add timing logs: CPU encode, GPU submit+wait, map+copy.
  * Simple perf threshold in test (e.g., total under 50–100 ms on CI runners).

* **Acceptance**

  * CI prints timing and meets threshold (allow slack on shared runners).

---

## A1.11 Documentation updates (0.25d)

**Goal**: Make it easy for others to reproduce.

* **Deliverables**

  * README sections:

    * “Headless spike: how to run tests”
    * “Backend forcing & expected hashes”
    * “Troubleshooting determinism” (drivers, environment, PNG library versions)

* **Acceptance**

  * A new dev can follow the README to run the spike in <10 minutes.

---

# Suggested assignment & order

* **Day 1**: A1.1, A1.2, A1.3
* **Day 2**: A1.4, A1.5, A1.6
* **Day 3**: A1.7, A1.8
* **Day 4**: A1.9, A1.10 (opt), A1.11

Parallelizable items: **A1.7** (backend scripts) and **A1.8** (CI) can proceed while **A1.6** (harness) is drafted.

---

## Implementation checklist (quick)

* [ ] Pin `wgpu`, `pyo3` versions; optional `rust-toolchain.toml`.
* [ ] Fix pipeline state (no MSAA, sRGB target).
* [ ] Verify row‑padding removal with a unit test.
* [ ] Add SHA‑256 + SSIM fallback checks.
* [ ] Backend‑forcing scripts + README.
* [ ] CI: run tests, upload PNGs with hash in filenames.
* [ ] Device report + helpful errors.
* [ ] (Opt) Timing logs + thresholds.

**A2. Terrain pipeline spike** *(\~3–5 days)*

* **Scope**

  * Add `Renderer.add_terrain(heightmap: np.ndarray[f32], spacing: (f32, f32), exaggeration: f32)` in Rust via `numpy` crate.
  * CPU: build indexed grid (triangle list) once.
  * WGSL: sample height, compute normals (forward differences), single directional light, gamma‑correct + simple tonemap.
  * Render path: write into the same off‑screen texture.
* **Exit criteria:** 1024² DEM renders under 50 ms on M‑series/NVIDIA, readback under 30 ms; image test SSIM ≥ 0.98 across backends.

Perfect—here’s a **fine‑grained breakdown** for **Milestone A → A2. Terrain pipeline spike** (Rust‑first, `wgpu`, Python via `PyO3`). Each task lists **Goal**, **Deliverables**, **Acceptance**, **Deps**, **Estimate**, and whether it can run **(P)** in parallel.

> **A2 exit criteria (recap):**
>
> * Add a `Renderer.add_terrain(...)` path that takes a NumPy DEM and renders a **lit 1024×1024** image through WGSL with simple tonemapping.
> * Measure render and readback time, and keep them within reasonable budgets on a mainstream GPU.

---

## Workstream T0 — API & Data Contract

### T0.1 Public API & validation

* **Goal:** Define and implement the MVP Python→Rust contract.
* **Deliverables:**

  * `Renderer.add_terrain(heightmap: np.ndarray, spacing: tuple[float,float], exaggeration: float = 1.0, *, colormap="viridis")`
  * Accept `float32` (prefer) or `float64` (cast to `f32`), shape `(H, W)`, C‑contiguous.
  * Store metadata: `dx, dy, h_min, h_max, exaggeration`, colormap name.
* **Acceptance:** Invalid dtype/shape/strides raise clear `PyRuntimeError`; docstring explains requirements.
* **Deps:** none
* **Estimate:** 0.5 d

### T0.2 DEM statistics & normalization

* **Goal:** Provide sensible default color/lighting ranges.
* **Deliverables:**

  * Compute `h_min/h_max` (and **optionally** 1–99 percentile clamp to resist outliers).
  * Expose `Renderer.set_height_range(min, max)` to override.
* **Acceptance:** CPU min/max computed; overrides reflected in render.
* **Deps:** T0.1
* **Estimate:** 0.25 d (P)

---

## Workstream T1 — CPU Mesh & GPU Resources

### T1.1 Grid index/vertex generator

* **Goal:** Build a reusable indexed grid for any `(W,H)` heightmap.
* **Deliverables:**

  * `terrain/mesh.rs::make_grid(W, H, dx, dy) -> {vertex_buffer, index_buffer, index_format}`
  * Vertex attrs:

    * `position.xy` in world meters (center the mesh at origin or start at (0,0)—**decide and document**),
    * `uv` in `[0,1]` for height sampling (`u = x/(W-1)`, `v = y/(H-1)`).
  * Use `u16` indices when `(W*H) ≤ 65535`, else `u32`.
* **Acceptance:** For `W=H=1024`, generation ≤ \~40 ms on dev machine; unit tests check index ranges and triangle winding (CCW).
* **Deps:** T0.1
* **Estimate:** 1 d

### T1.2 Height texture upload (R32Float)

* **Goal:** Upload DEM to GPU as a single‑channel float texture.
* **Deliverables:**

  * Create `Texture2D` with `R32Float`, usage `TEXTURE_BINDING | COPY_DST`.
  * `queue.write_texture` with **256‑byte row alignment** handled (pad rows when necessary).
  * Linear clamp sampler.
* **Acceptance:** Validation passes on Metal/Vulkan/DX12; probe a few texels via a temp compute or debug path (optional).
* **Deps:** T0.1
* **Estimate:** 0.5 d (P)

### T1.3 Colormap LUT texture

* **Goal:** Map heights to colors via a small LUT.
* **Deliverables:**

  * 256×1 `RGBA8UnormSrgb` texture with built‑in palettes: `viridis`, `magma`, `terrain`.
  * Uniforms: `h_min`, `h_max` for normalization.
* **Acceptance:** Known scalar inputs map to expected palette colors in a unit test (CPU reference).
* **Deps:** T0.2
* **Estimate:** 0.5 d (P)

---

## Workstream T2 — Uniforms, Camera, and Lighting

### T2.1 Camera math + uniform buffer

* **Goal:** Provide view/projection matrices for terrain.
* **Deliverables:**

  * `core/camera.rs` (orbit + look\_at + perspective).
  * WGSL/host uniform struct (aligned for WGSL rules):

    ```wgsl
    struct Globals {
      view : mat4x4<f32>,
      proj : mat4x4<f32>,
      sun_dir : vec3<f32>,
      exposure : f32,
      spacing : vec2<f32>,     // (dx, dy)
      h_range : vec2<f32>,     // (h_min, h_max)
      exaggeration : f32,
      _pad : f32,
    };
    @group(0) @binding(0) var<uniform> globals : Globals;
    ```
  * Device buffer + bind group layout.
* **Acceptance:** Matrices pass unit tests (look\_at and perspective for known cases).
* **Deps:** none
* **Estimate:** 0.75 d

### T2.2 Sun direction & tonemap

* **Goal:** Basic single‑sun lighting with simple tonemap.
* **Deliverables:**

  * Helpers: `set_sun(elevation, azimuth)` → normalized world `sun_dir`.
  * WGSL functions: `reinhard(x)`, `gamma_correct(x, 2.2)`.
* **Acceptance:** Shader compiles; a test vector through tonemap matches CPU ref ± small epsilon.
* **Deps:** T2.1
* **Estimate:** 0.25 d (P)

---

## Workstream T3 — Terrain Shaders & Pipeline

### T3.1 Terrain WGSL (vertex)

* **Goal:** Reconstruct world position, pass UV.
* **Deliverables:**

  * VS inputs: `position.xy`, `uv`.
  * Sample height in **VS** or FS? For MVP, do height reconstruction in VS to move vertices in Z:

    ```wgsl
    let h = textureSampleLevel(heightTex, heightSamp, in.uv, 0.0).r;
    let z = (h - globals.h_range.x) / (globals.h_range.y - globals.h_range.x);
    let world_z = (h) * globals.exaggeration; // or normalized_z * scale—decide & document
    ```
  * Output `world_pos` to FS if needed; set `@builtin(position)` from `proj * view * vec4(world,1)`.
* **Acceptance:** Compiles; no warnings; coordinates consistent with camera conventions.
* **Deps:** T1.1, T1.2, T2.1
* **Estimate:** 0.5 d

### T3.2 Terrain WGSL (fragment)

* **Goal:** Compute normals from height texture & shade.
* **Deliverables:**

  * Forward difference normals using `dx, dy` spacing:

    ```wgsl
    let h  = textureSampleLevel(heightTex, heightSamp, in.uv, 0.0).r;
    let hx = textureSampleLevel(heightTex, heightSamp, in.uv + vec2(1.0/(W-1.0), 0.0), 0.0).r;
    let hy = textureSampleLevel(heightTex, heightSamp, in.uv + vec2(0.0, 1.0/(H-1.0)), 0.0).r;
    let dpx = vec3(globals.spacing.x, 0.0, (hx - h) * globals.exaggeration);
    let dpy = vec3(0.0, globals.spacing.y, (hy - h) * globals.exaggeration);
    let n = normalize(cross(dpy, dpx));
    ```
  * Lambert + small ambient; color from LUT at normalized height; `tone(gamma(color))`.
* **Acceptance:** Visual: sun from the east lights the east slopes; normal flips when `sun_dir` flips.
* **Deps:** T1.2, T1.3, T2.2
* **Estimate:** 0.75 d

### T3.3 Pipeline state & bindings

* **Goal:** Create render pipeline and bind groups.
* **Deliverables:**

  * Bind group(0): `Globals`.
  * Bind group(1): height texture + sampler.
  * Bind group(2): colormap texture + sampler.
  * Pipeline layout and `RenderPipelineDescriptor` (no depth, MSAA=1).
* **Acceptance:** Validation layers clean on all backends; triangle sample still renders when toggled.
* **Deps:** T2.1, T1.2, T1.3
* **Estimate:** 0.5 d (P)

---

## Workstream T4 — Integration & Output

### T4.1 Scene integration

* **Goal:** Hook terrain layer into `Renderer.render_rgba()`.
* **Deliverables:**

  * `Scene` holds `Vec<Layer>`; `Layer::Terrain { gpu, uniforms, mesh }`.
  * Per‑frame update of `Globals` (view/proj/sun/exposure/h\_range/spacing/exaggeration).
  * Command encoder: begin pass → draw terrain → end.
* **Acceptance:** First image produced from real DEM (even if flat‑shaded initially).
* **Deps:** T1–T3
* **Estimate:** 0.5 d

### T4.2 PNG & NumPy round‑trip

* **Goal:** Provide outputs the same as A1.
* **Deliverables:**

  * `render_rgba() -> np.ndarray[H,W,4] uint8`, `render_png(path)`.
  * Reuse persistent readback buffer; handle row unpadding (already in starter).
* **Acceptance:** 1024×1024 DEM produces PNG and RGBA with correct shapes/dtypes.
* **Deps:** T4.1
* **Estimate:** 0.25 d (P)

---

## Workstream T5 — Tests, Timing, Docs

### T5.1 Synthetic DEM tests

* **Goal:** Deterministic correctness on small inputs.
* **Deliverables:**

  * 64×64 gradient DEM and 2D Gaussian bump DEM in `tests/data/`.
  * Unit test: renders → checks SSIM ≥ 0.99 vs golden images.
  * CPU check: face normal on a plane matches shader normal (approx).
* **Acceptance:** Tests pass locally; goldens stable across backends with tolerance.
* **Deps:** T4.2
* **Estimate:** 0.5 d

### T5.2 Timing harness

* **Goal:** Measure GPU & readback time.
* **Deliverables:**

  * Log encode time, `queue.submit + device.poll` time, map/copy time.
  * Emit a summary dict from `Renderer.render_metrics()` for perf tracking.
* **Acceptance:** On a mainstream GPU, 1024²: **render ≤ \~50 ms**, **readback ≤ \~30–60 ms** (rough targets).
* **Deps:** T4.2
* **Estimate:** 0.25 d (P)

### T5.3 README updates & examples

* **Goal:** Make spike reproducible for others.
* **Deliverables:**

  * `python/examples/terrain_single_tile.py` (loads synthetic DEM, writes PNG).
  * README: “Terrain spike” usage and perf notes.
* **Acceptance:** Fresh dev can reproduce in <10 minutes.
* **Deps:** T4.2
* **Estimate:** 0.25 d (P)

---

## Suggested 4‑day schedule (1–2 engineers)

**Day 1:** T0.1, T0.2 (P), T1.1
**Day 2:** T1.2 (P), T1.3 (P), T2.1
**Day 3:** T2.2 (P), T3.1, T3.2, T3.3 (P)
**Day 4:** T4.1, T4.2 (P), T5.1, T5.2 (P), T5.3 (P)

---

## Technical specifics & guardrails

* **Texture uploads:** `wgpu` requires `bytes_per_row % 256 == 0`. Pad CPU rows (`W * 4` for RGBA8; for R32F DEM, `W * 4` as well) to the next 256‑aligned stride before `queue.write_texture`.
* **Coordinate system:** Pick and document **right‑handed**, +Y up, +Z out of the ground. Ensure consistency between CPU world coords and camera/view matrix.
* **Normals:** Use `cross(dpy, dpx)` for right‑handed orientation; if lighting looks inverted, swap order.
* **sRGB:** Keep render target `Rgba8UnormSrgb`; do lighting in linear, then `gamma_correct` before writing, or rely on sRGB target with linear input (avoid double‑gamma).
* **Bounds:** If `(W*H) > 65,535`, use `u32` indices; consider clamping `W,H ≤ 8192` for MVP to avoid huge buffers.
* **Exposure:** A single scalar is enough; default 1.0.
* **Determinism:** Avoid MSAA; use `textureSampleLevel(..., 0.0)` to avoid LOD/derivative ambiguity in normals.

---

## Definition of Done (A2)

* `add_terrain()` accepts a NumPy DEM and validates it.
* Rendering a **1024×1024** DEM yields lit terrain with colormap, sun lighting, and tonemap.
* `render_rgba()` and `render_png()` work; perf metrics are reported.
* Synthetic tests pass (SSIM ≥ 0.99 vs goldens); README example reproduces.

---

**A3. Geo tessellation spike** *(\~3–5 days)*

* **Scope**

  * In Python, accept GeoPandas (or Shapely geometries). In Rust, use `geo` + `earcutr` (or `lyon`) to triangulate multipolygons + holes.
  * API: `add_polygons(gdf_like, line_color, fill_rgba)`
  * Require CRS to be pre‑projected for MVP; document this clearly.
* **Exit criteria:** Non‑self‑intersecting multi‑polygons render correctly; islands/holes respected; one complex city polygon renders in <100 ms (triangulation included).

Excellent—here’s a **fine‑grained breakdown** for **Milestone A → A3. Geo tessellation spike** (Rust‑first core, Python API via PyO3, wgpu rendering). Each task has **Goal, Deliverables, Acceptance, Deps, Estimate**, and parallelizable items are marked **(P)**.

> **Exit criteria for A3:**
>
> 1. Take a **GeoPandas/Shapely multipolygon** → **robust tessellation (triangles) via earcut** (holes supported).
> 2. Render **filled polygons** and a **line/outline overlay** to a 2D off‑screen target (512–1024 px).
> 3. Validate **ring orientation/holes**, reject **self‑intersecting** geometries, and enforce **CRS policy** (planar coords).
> 4. Provide timing numbers for tessellation & draw, plus minimal tests (golden images or triangle counts).

---

## Workstream G0 — Public API & CRS policy

### G0.1 Python API (public) & docstring

* **Goal:** Define input contract and expose a minimal method.
* **Deliverables:**

  * Python:

    ```python
    Renderer.add_polygons(
        geoms,                     # GeoPandas GeoSeries/GeoDataFrame, Shapely objects, or raw arrays
        *, fill_rgba=(0.2,0.2,0.2,1.0),
        line_rgba=(0,0,0,1), line_width=1.0,
        assume_planar: bool = True
    )
    ```
  * Docstring: **Require pre‑projected planar coordinates** (e.g., WebMercator meters or local CRS). No antimeridian handling in A3.
* **Acceptance:** Clear error if CRS missing or geographic (lat/lon) unless `assume_planar=True`.
* **Deps:** —
* **Estimate:** 0.25 d

---

## Workstream G1 — Python extraction & FFI packing

### G1.1 GeoPandas/Shapely → raw buffers

* **Goal:** Convert user geometries to a stable FFI format without native deps in Rust for parsing.
* **Deliverables:**

  * Python helper: `pack_polygons(geoms) -> (coords, ring_offsets, feature_offsets)` where:

    * `coords`: `np.float32` flat `[x0,y0, x1,y1, …]`
    * `ring_offsets`: `np.uint32` size `R+1` (prefix sums, ring i spans `coords[2*ring_offsets[i]:2*ring_offsets[i+1]]`)
    * `feature_offsets`: `np.uint32` size `F+1` (rings per feature)
  * Handles `Polygon` (1 exterior + holes), `MultiPolygon` (multiple polygons).
  * Strips duplicate consecutive points and enforces closed rings (first==last), but **does not** insert extra closure point into buffers (earcut prefers open rings; we keep open rings internally and ensure closure only for validity checks).
* **Acceptance:** Unit test: pack a donut (square with hole) & a multipolygon → expected counts for coords/rings/features.
* **Deps:** G0.1
* **Estimate:** 0.75 d

### G1.2 Sanity checks & normalization (P)

* **Goal:** Stabilize geometry before FFI call.
* **Deliverables:**

  * Validate min ring length (≥3 unique points), drop tiny rings (area < ε²).
  * Optional **RDP simplification** (tolerance `ε`) to reduce extremely dense rings (behind a flag).
  * Normalize coordinates by feature bbox center (translate to \~origin) and scale to \~`1e4` range to improve earcut numeric stability; keep scale/offset to re‑emit in world coords later.
* **Acceptance:** Pathological inputs (thousands of collinear points) don’t explode runtime; areas preserved within tolerance.
* **Deps:** G1.1
* **Estimate:** 0.5 d

---

## Workstream G2 — Rust FFI, validation & canon

### G2.1 FFI structs & PyO3 binding

* **Goal:** Accept packed arrays from Python without copies.
* **Deliverables:**

  * Rust function:

    ```rust
    fn add_polygons_raw(coords: PyReadonlyArray1<f32>,
                        ring_offsets: PyReadonlyArray1<u32>,
                        feature_offsets: PyReadonlyArray1<u32>,
                        fill_rgba: [f32;4],
                        line_rgba: [f32;4],
                        line_width: f32) -> PyResult<()>
    ```
  * Convert to borrowed slices; validate monotonic offsets and bounds.
* **Acceptance:** Bad inputs raise `PyRuntimeError` with clear message.
* **Deps:** G1.1
* **Estimate:** 0.5 d

### G2.2 Ring orientation & topology checks (P)

* **Goal:** Canonicalize ring winding and reject invalid topology.
* **Deliverables:**

  * For each feature: compute signed area; set **exterior CCW**, **holes CW** (flip if necessary).
  * Detect **self‑intersection** (use `geo` crate’s `is_simple()` on rings or a lightweight segment intersection check).
  * Ensure holes lie inside their exterior (point‑in‑polygon test for a sample point).
* **Acceptance:**

  * Bow‑tie polygon rejected.
  * Hole outside exterior rejected with explanation.
* **Deps:** G2.1
* **Estimate:** 0.75 d

---

## Workstream G3 — Tessellation (earcut) & mesh assembly

### G3.1 Per‑feature earcut (core)

* **Goal:** Triangulate polygons with holes.
* **Deliverables:**

  * Use **`earcutr`** crate: for each feature, build vector of rings (first exterior, then holes) and call earcut.
  * Return indices (u32) into a **feature‑local** vertex list; accumulate into global buffers with index base offset.
* **Acceptance:**

  * For a simple polygon with V vertices and H holes, triangle count ≈ **V − 2 − H**.
  * Triangles are CCW in world coords (check orientation).
* **Deps:** G2.2
* **Estimate:** 1 d

### G3.2 Attribute building & compaction (P)

* **Goal:** Construct GPU‑ready vertex attributes.
* **Deliverables:**

  * Vertex struct: `{ position: vec2<f32>, feature_id: u32 }` (store `feature_id` for flat‑color/ID renders).
  * Optional dedup of identical vertices (hash by quantized XY) to reduce VBO size.
* **Acceptance:** VBO/IBO created; counts logged; optional dedup reduces size on shared borders by >10% on test data.
* **Deps:** G3.1
* **Estimate:** 0.5 d

### G3.3 Stroke mesh (outline) CPU tessellator

* **Goal:** Build a thin triangle strip mesh for each ring as an outline.
* **Deliverables:**

  * For each ring, generate **screen‑space width** approximated in world units using current camera scale (for this spike, allow **constant world‑space width** to keep it simple).
  * **Miter** joins with angle clamp; **butt** caps.
  * Output a separate VBO/IBO for strokes.
* **Acceptance:** Visual outline around exteriors and hole boundaries; no gaps or inverted normals; reasonable performance on \~10k segments.
* **Deps:** G3.1 (can start earlier with ring coords)
* **Estimate:** 1 d

---

## Workstream G4 — GPU upload & render pipelines

### G4.1 Buffer uploads & lifetime management

* **Goal:** Move tessellated geometry to GPU.
* **Deliverables:**

  * Create static vertex/index buffers for fills and strokes with `COPY_DST|VERTEX/INDEX`.
  * One **bind group** for a small `feature_color` SSBO or push constants; for spike, a **single uniform fill** is fine.
* **Acceptance:** Validation clean; memory tracked; debug labels set.
* **Deps:** G3.2, G3.3
* **Estimate:** 0.5 d (P)

### G4.2 Fill pipeline (triangles)

* **Goal:** Draw filled polygons.
* **Deliverables:**

  * WGSL VS: pass through `position.xy` → clip via model/view/proj (for A3 2D, you can map to NDC directly).
  * WGSL FS: output `fill_rgba` (uniform) or color by `feature_id` (optional LUT).
  * Blending: premultiplied alpha **off** (straight alpha OK); no depth for 2D spike.
* **Acceptance:** Donut shows hole correctly (no fill bleeding); multipolygons draw in one pass.
* **Deps:** G4.1
* **Estimate:** 0.5 d

### G4.3 Stroke pipeline (triangles)

* **Goal:** Draw outlines above fills.
* **Deliverables:**

  * Separate pipeline using stroke VBO/IBO; `line_rgba`, `line_width` already baked by CPU expand.
  * Draw ordered after fill; enable alpha if needed.
* **Acceptance:** Closed outlines exactly follow polygon and hole boundaries at requested width.
* **Deps:** G3.3, G4.1
* **Estimate:** 0.5 d (P)

---

## Workstream G5 — Tests, metrics, and determinism

### G5.1 Unit tests: packing & topology

* **Goal:** Validate the data path before GPU.
* **Deliverables:**

  * Tests for: square, donut (square hole), multipolygon (two squares), invalid bow‑tie.
  * Assert ring/feature counts, orientation (exterior CCW, holes CW), and rejection messages.
* **Acceptance:** All pass locally and in CI.
* **Deps:** G1.1, G2.2
* **Estimate:** 0.5 d

### G5.2 Golden image tests (64×64) (P)

* **Goal:** Visual correctness with tolerance.
* **Deliverables:**

  * Render tiny scenes: (1) single polygon, (2) donut, (3) multipolygon overlay.
  * Compare against goldens (SSIM ≥ 0.99); skip if backend unavailable, log info().
* **Acceptance:** Tests pass on Linux (Vulkan) & macOS (Metal) CI jobs.
* **Deps:** G4.2, G4.3
* **Estimate:** 0.5 d

### G5.3 Performance log

* **Goal:** Establish baseline timings.
* **Deliverables:**

  * Log: #features, #rings, #verts, #tris; tessellation time; upload time; draw time; total.
  * Add `Renderer.last_metrics()` to fetch numbers.
* **Acceptance:** For 10k total vertices, tessellation < \~100 ms on dev laptop; draw negligible.
* **Deps:** G3.*, G4.*
* **Estimate:** 0.25 d

---

## Workstream G6 — Docs & examples

### G6.1 Example: city blocks

* **Goal:** Showcase A3 end‑to‑end.
* **Deliverables:**

  * `python/examples/polygons_city.py` that builds synthetic polygon rings (no GeoPandas dependency for the example), calls `add_polygons`, renders PNGs for **fill only** and **fill+stroke** variants.
* **Acceptance:** Runs after `maturin develop`; produces two PNGs.
* **Deps:** G4.\*
* **Estimate:** 0.25 d (P)

### G6.2 README section: “Geo tessellation spike”

* **Goal:** Communicate constraints to users.
* **Deliverables:**

  * Notes on **CRS requirement**, **unsupported cases** (dateline wrap, very tiny rings), and **performance tips** (simplification).
* **Acceptance:** PR reviewed and merged.
* **Deps:** G0.1
* **Estimate:** 0.25 d

---

## Suggested 3–4 day schedule (1–2 engineers)

**Day 1:** G0.1, G1.1, G1.2 (P), G2.1
**Day 2:** G2.2, G3.1
**Day 3:** G3.2 (P), G3.3, G4.1
**Day 4:** G4.2, G4.3 (P), G5.1, G5.2 (P), G5.3, G6.\* (P)

Parallelization tips:

* One engineer: **Python packing (G1) + tests (G5.1)**.
* Another: **Rust FFI/validation (G2) → earcut (G3.1) → GPU (G4)**.
* If you have a third, hand them **stroke mesh (G3.3)** and **golden tests (G5.2)**.

---

## Technical guardrails & choices

* **CRS/units:** **Planar coordinates only** for A3. Users must project before calling (e.g., EPSG:3857 meters). Add a clear error if coordinates look like lat/lon ranges (|x| ≤ 180 & |y| ≤ 90) unless `assume_planar=True`.
* **Orientation:** Standardize to **exterior CCW**, **holes CW** before earcut; while earcut doesn’t strictly require winding, canonicalization helps downstream logic (e.g., stroke expansion).
* **Holes:** Associate holes with the nearest containing exterior; reject if ambiguous or nested incorrectly.
* **Degeneracy:** Drop rings with <3 unique points after dedup; area < ε²; simplify long nearly‑collinear runs with RDP if enabled.
* **Numeric stability:** Translate/scale features locally pre‑earcut; rescale back for rendering.
* **Stroke width:** For spike, use a **world‑space constant** (e.g., meters). If time permits, add simple screen‑space adjustment based on current camera scale.
* **WGPU line primitives:** Don’t rely on wireframe polygon mode or line lists; **always** render outlines as triangle meshes.

---

## Definition of Done (A3)

* `add_polygons(...)` accepts Geo‑like inputs (or packed arrays) and validates them.
* Tessellation produces triangle buffers with correct **holes** and **multipolygons**.
* Fill + outline render correctly into a PNG via off‑screen target.
* Basic tests: packing/topology + golden images pass in CI; timings are logged.

---

**A4. Wheel/CI spike** *(\~1 day)*

* Use the included GitHub Actions workflow.
* **Exit criteria:** Wheels produced on Ubuntu, Windows, macOS (x64 & Apple Silicon if runner available). Post‑build import & render test passes.

> **Exit criteria (A4):**
>
> 1. Build **release wheels** for **Windows (x86\_64)**, **Linux (manylinux2014 x86\_64)**, **macOS (arm64 & x86\_64)**.
> 2. Run a **headless smoke test** that imports the wheel and renders a 512×512 RGBA image.
> 3. Upload wheels as CI artifacts; optionally publish to **TestPyPI** via a manual trigger.
> 4. Jobs are **deterministic** and **fast** (caching enabled); logs include device/backend info.

---

## Strategy choice (decide once)

* **S1 (Recommended)**: **maturin‑only pipeline** with `abi3` wheels (already configured) → **build once per OS/arch** (no need to repeat per Python version). Simplest and fastest.
* **S2**: `cibuildwheel` if you want its ecosystem niceties (e.g., manylinux build images, cross‑arch docker). Slightly more setup.

Below assumes **S1 (maturin)**. If you prefer S2, I’ve put an alternate plan at the bottom.

---

## A4.1 Pin toolchains & metadata (0.5 d)

**Goal:** Make builds reproducible.

* **Deliverables**

  * Add `rust-toolchain.toml`:

    ```toml
    [toolchain]
    channel = "1.77.2"     # pick a stable you’ve tested
    components = ["clippy", "rustfmt"]
    ```
  * Confirm `Cargo.toml` pins: `pyo3 = { features = ["abi3-py310"] }`, `wgpu = "0.19.x"`.
  * Ensure `pyproject.toml` has `requires-python = ">=3.10"` and module name `vulkan_forge._vulkan_forge`.

* **Acceptance**

  * `maturin develop --release` succeeds on all three OSes with the pinned toolchain.

**Deps:** none

---

## A4.2 Matrix design & caching (0.5 d)

**Goal:** Define the CI matrix and caches.

* **Matrix**

  * **Linux**: `ubuntu-22.04` → **manylinux2014\_x86\_64** compatibility (`maturin` handles auditwheel).
  * **Windows**: `windows-2022` → `win_amd64`.
  * **macOS**:

    * `macos-14` (Apple Silicon) → `macosx_arm64`.
    * `macos-13` (Intel) → `macosx_x86_64`.
  * **Python**: because we use **abi3**, build **once** per OS/arch (no need to matrix Python versions).

* **Caching**

  * Use `Swatinem/rust-cache@v2` (Cargo) + pip cache.

* **Acceptance**

  * A single CI run shows caches primed and subsequent runs shave minutes off build time.

**Deps:** A4.1

---

## A4.3 CI workflow skeleton (0.75 d)

**Goal:** Implement one GH Actions workflow to build wheels and run smoke tests.

**Deliverables (YAML sketch)**

```yaml
name: Wheels

on:
  push:
    branches: [ main ]
  pull_request:
  workflow_dispatch:

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-22.04   # manylinux2014_x86_64
            target: linux
          - os: windows-2022   # win_amd64
            target: windows
          - os: macos-13       # macOS x86_64
            target: macos-x86_64
          - os: macos-14       # macOS arm64
            target: macos-arm64
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy,rustfmt
      - uses: Swatinem/rust-cache@v2

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install maturin & test deps
        run: pip install maturin pillow numpy

      - name: Build wheels
        shell: bash
        run: |
          if [[ "${{ matrix.target }}" == "linux" ]]; then
            maturin build --release --strip --compatibility manylinux2014 -o wheels
          elif [[ "${{ matrix.target }}" == "macos-arm64" ]]; then
            # arm64 on arm runner
            maturin build --release --strip -o wheels
          elif [[ "${{ matrix.target }}" == "macos-x86_64" ]]; then
            # x86_64 on Intel runner
            maturin build --release --strip -o wheels
          else
            maturin build --release --strip -o wheels
          fi

      - name: Install built wheel (test)
        run: pip install --no-index --find-links wheels vulkan-forge

      - name: Smoke test render
        env:
          # Prefer software on CI to avoid adapter issues (WARP/D3D12 on Win, Lavapipe on Linux, Metal on mac)
          VULKAN_FORGE_PREFER_SOFTWARE: "1"
        run: |
          python - << 'PY'
          import hashlib, io, sys
          import numpy as np
          from PIL import Image
          import vulkan_forge
          r = vulkan_forge.Renderer(512,512)
          print(r.info())
          arr = r.render_triangle_rgba()
          assert arr.shape == (512,512,4) and arr.dtype == np.uint8
          b = io.BytesIO(); Image.fromarray(arr, 'RGBA').save(b, 'PNG')
          print("SHA256:", hashlib.sha256(b.getvalue()).hexdigest())
          PY

      - uses: actions/upload-artifact@v4
        with:
          name: wheels-${{ matrix.os }}
          path: wheels
```

* **Acceptance**

  * All four jobs green.
  * Artifacts contain the built wheels.

**Deps:** A4.2

---

## A4.4 Fallback adapter option (code change) (0.5 d)

**Goal:** Improve CI reliability on headless runners.

* **Deliverables**

  * In `context.rs` or current `lib.rs`, change device selection to try **hardware** first; if it fails, retry with `force_fallback_adapter=true`. Surface an env var:

    * `VULKAN_FORGE_PREFER_SOFTWARE=1` → set `force_fallback_adapter=true`.
  * Export `Renderer.info()` with adapter name & backend (to log in CI).

* **Acceptance**

  * Linux job uses Lavapipe/Vulkan (or Swiftshader if present); Windows uses D3D12 WARP; macOS uses Metal (software is not a thing—Metal driver is baked in).

**Deps:** none

---

## A4.5 Wheel compliance checks (0.25 d)

**Goal:** Ensure wheels are compliant and metadata is correct.

* **Deliverables**

  * After build, run:

    * **Linux**: `auditwheel show wheels/*.whl`
    * **macOS**: `delocate-listdeps -d wheels/*.whl` *(optional; maturin generally handles this)*
    * **All**: `pipx run twine check wheels/*` (or `python -m twine check wheels/*`)

* **Acceptance**

  * No missing external libs; `twine check` passes.

**Deps:** A4.3

---

## A4.6 sdist build & check (0.25 d)

**Goal:** Source distribution builds and includes Rust sources.

* **Deliverables**

  * `maturin sdist -o dist`
  * Verify `tar -tf dist/*.tar.gz` includes `Cargo.toml`, `src/`, `pyproject.toml`, `python/`.

* **Acceptance**

  * `twine check dist/*` passes; local `pip install dist/*.tar.gz` builds & runs smoke test.

**Deps:** A4.3

---

## A4.7 TestPyPI publish (manual) (0.5 d)

**Goal:** One‑button publish to TestPyPI from CI artifacts.

* **Deliverables**

  * A second job triggered by `workflow_dispatch` with inputs `version` & `prerelease` that:

    * Downloads artifacts from the build job.
    * Publishes to **TestPyPI** using **Trusted Publishing** (preferred) or an API token secret:

      ```yaml
      - name: Publish TestPyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          repository-url: https://test.pypi.org/legacy/
          packages-dir: wheels
      ```
* **Acceptance**

  * Package appears on TestPyPI; `pip install -i https://test.pypi.org/simple vulkan-forge` works on a fresh VM.

**Deps:** A4.3, A4.5

---

## A4.8 Release pipeline to PyPI (0.5 d)

**Goal:** On git tag `vX.Y.Z`, build and publish to PyPI.

* **Deliverables**

  * Separate workflow `release.yml`:

    * Trigger: `on: push: tags: [ "v*" ]`.
    * Steps: rebuild wheels (or reuse artifacts), `sdist`, run checks, then `gh-action-pypi-publish` (Trusted Publishing).

* **Acceptance**

  * Pushing a tag to a temp repo publishes to TestPyPI (first) then PyPI (after toggling URL).

**Deps:** A4.5, A4.6

---

## A4.9 Nightly canary build (optional, 0.25 d)

**Goal:** Catch regressions from dependencies (wgpu, pyo3, toolchains).

* **Deliverables**

  * Add a scheduled CI (`cron: "0 2 * * *"`) that builds wheels (no publish), runs smoke + determinism tests, and posts a badge result.

* **Acceptance**

  * Nightly job runs; failures alert via repo notifications.

**Deps:** A4.3

---

## A4.10 Docs & badges (0.25 d)

**Goal:** Update README badges and instructions.

* **Deliverables**

  * Replace `USER/REPO` slugs in badges.
  * Add “Install wheel from artifacts” instructions for internal testers.

* **Acceptance**

  * Badges show status; instructions are copy‑pasteable.

**Deps:** A4.3

---

## A4.11 Size/performance budget (0.25 d)

**Goal:** Keep wheel sizes reasonable.

* **Deliverables**

  * Ensure `Cargo.toml` has:

    ```toml
    [profile.release]
    codegen-units = 1
    lto = "thin"
    ```
  * `strip = true` in `pyproject.toml` maturin config (already set).
  * CI step prints final wheel sizes; warn if > 20–30 MB.

* **Acceptance**

  * Typical wheel < 10 MB; if bigger, investigate embedded debug info or static libs.

**Deps:** A4.3

---

## Suggested 2‑day schedule

**Day 1:** A4.1–A4.4
**Day 2:** A4.5–A4.11

Parallelizable: A4.5/A4.6 can run while A4.7/A4.8 are drafted.

---

## Pitfalls & guardrails

* **Linux GPU stack in CI**: Don’t rely on a display; use surface‑less off‑screen targets (you already do). Prefer software adapters in CI via `VULKAN_FORGE_PREFER_SOFTWARE=1`.
* **abi3 sanity**: With `pyo3` abi3, you **don’t need** to build per Python version—simplify the matrix to one Python (e.g., 3.12). Keep runtime tests with that Python only.
* **manylinux**: Always pass `--compatibility manylinux2014` on Linux builds.
* **Universal2?** If you want **one** macOS wheel instead of two, maturin can build `--universal2` on macOS. Trade‑off: longer link times, some crates don’t handle it well. Safe to ship two wheels (arm64 & x86\_64) for now.
* **Publishing**: Prefer **Trusted Publishing** over tokens; set up PyPI project owners and GitHub OIDC.

---

## Optional: `cibuildwheel` variant (if you choose S2)

* Replace the build step with:

  ```yaml
  - name: Install cibuildwheel
    run: pip install cibuildwheel==2.20 maturin
  - name: Build wheels
    env:
      CIBW_BUILD: "cp310-*"
      CIBW_SKIP: "pp* *_i686 *-musllinux_*"
      CIBW_ENVIRONMENT: >
        VULKAN_FORGE_PREFER_SOFTWARE=1
    run: cibuildwheel --output-dir wheels
  ```
* Add `pyproject.toml` `[tool.cibuildwheel]` to configure manylinux, macOS universal2, etc.

---

### Milestone B — MVP implementation (4–6 weeks)

**Week 1–2: Core & Terrain**

* Scene graph (in Rust): `Scene`, `Layer` enum {Terrain, Polygons, Lines, Points}, `Camera`, `Sun`.
* Terrain layer:

  * CPU tiling & LOD pyramid (simple quad‑tree, memory‑resident).
  * Per‑tile draw, indexed grid, MSAA=1 for speed, MSAA=4 for stills.
* Python API: thin wrappers, type conversions (NumPy ↔ Rust slices).

Absolutely—here’s a **fine‑grained Week 1–2 plan** for **Milestone B: Core & Terrain** (Option B, Rust‑first). It’s organized as small tickets you can drop into your tracker, each with **goal, deliverables, acceptance criteria, dependencies, and estimates**. Items marked **(P)** can run in parallel.

---

## Sprint goal (Week 1–2)

* Establish a minimal engine core (device/context, frame targets, camera).
* Implement a **single‑tile** terrain pipeline end‑to‑end: **NumPy DEM → GPU → lit render → PNG/NumPy RGBA**.
* Lay foundations for tiling/LOD to extend in Week 3.

---

## Workstream A — Core engine & scaffolding

**A1. Engine layout & error type**

* **Goal:** Create module structure and a single error type.
* **Deliverables:**

  * `src/context.rs` (adapter/device/queue, diagnostics).
  * `src/core/{framegraph.rs, gpu_types.rs}` (skeletons).
  * `src/error.rs` with `Error` + `Result<T>`.
  * Rewire `lib.rs` to use `context`.
* **Acceptance:** Compiles; `Renderer.info()` uses new context; unit tests build.
* **Deps:** none.
* **Estimate:** 0.5–1 day.

**A2. Off‑screen target & format policy**

* **Goal:** Standardize the render target and readback path.
* **Deliverables:**

  * `core/target.rs` with `RenderTarget { texture, view }`, create/destroy, `Rgba8UnormSrgb`.
  * Readback helper w/ row‑padding (refactor current code).
* **Acceptance:** 512×512 triangle renders identically to pre‑refactor; round‑trip to PNG.
* **Deps:** A1.
* **Estimate:** 0.5 day.

**A3. Device diagnostics & feature gating**

* **Goal:** Deterministic device info and conservative limits.
* **Deliverables:**

  * `Renderer.report_device() -> str` (name, backend, limits).
  * Internal `DeviceCaps` (max texture size, MSAA supported).
* **Acceptance:** Reports match backend; MSAA off by default if unsupported.
* **Deps:** A1.
* **Estimate:** 0.5 day.

**A4. Color management & tonemap stub**

* **Goal:** Linear workflow, sRGB output, simple tonemap.
* **Deliverables:**

  * WGSL tonemap functions (`reinhard` + gamma 2.2).
  * Document texture/view format choices.
* **Acceptance:** Shader compiles; triangle path unchanged; unit test of function values (CPU ref).
* **Deps:** A2.
* **Estimate:** 0.5 day. (P)

---

## Workstream B — Camera, uniforms, and math

**B1. Coordinate system & camera spec**

* **Goal:** Fix conventions used across CPU/GPU.
* **Deliverables:**

  * Decision doc: right‑handed, +Y up, camera orbit params, degrees vs radians.
  * `core/camera.rs` (look\_at, perspective/ortho, orbit).
* **Acceptance:** Deterministic view/proj matrices; unit tests compare against known values.
* **Deps:** none.
* **Estimate:** 0.5–1 day.

**B2. Uniform buffer plumbing**

* **Goal:** Provide a small uniform block for camera + sun.
* **Deliverables:**

  * `GpuUniforms { view, proj, sun_dir, exposure }` with `wgpu::Buffer`, bind group layout/index.
  * Update triangle shader to read uniforms (even if not used yet).
* **Acceptance:** Binds do not break the triangle render; passes validation.
* **Deps:** B1, A1.
* **Estimate:** 0.5 day. (P)

---

## Workstream C — Terrain (single tile, end‑to‑end)

**C1. DEM FFI contract**

* **Goal:** Robust acceptance of NumPy DEM arrays in Rust.
* **Deliverables:**

  * Python: `add_terrain(heightmap: np.ndarray[f32|f64], spacing: tuple[float,float], exaggeration: float=1.0, colormap="viridis")`
  * Rust: `PyArray2<f32>` path (cast f64→f32 if needed), C‑contig check, shape `(H,W)`.
  * Store in `TerrainLayer` struct with metadata (dx, dy, z\_scale).
* **Acceptance:** Type/shape errors raise clean `PyRuntimeError`; unit tests for dtype coercion.
* **Deps:** A1.
* **Estimate:** 0.5 day.

**C2. Grid mesh generator (CPU)**

* **Goal:** Build an indexed grid once; reuse VBO/IBO.
* **Deliverables:**

  * `terrain/mesh.rs` with `make_grid(width, height)` producing positions as (x,y) in world‑meters plane and index buffer (u16 when possible, else u32).
  * Vertex layout: `pos_xy`, `uv` (0..1) to sample heightmap in shader.
* **Acceptance:** For 1024×1024, generator under \~40 ms; index type chosen correctly; unit tests on index ranges.
* **Deps:** C1.
* **Estimate:** 1 day.

**C3. DEM upload & sampling texture**

* **Goal:** Bind DEM as a sampled texture on GPU.
* **Deliverables:**

  * Create `Texture2D<f32>` (R32Float), sampler (clamp, linear), bind group.
  * Optionally generate a small prefiltered normal/ambient texture later (not yet).
* **Acceptance:** Texture creation succeeds across backends; validation clean.
* **Deps:** A1, C1.
* **Estimate:** 0.5 day. (P)

**C4. Colormap LUT**

* **Goal:** Map heights to colors in shader via 1D LUT.
* **Deliverables:**

  * Built‑in LUTs (`viridis`, `magma`, `terrain`) baked as 256×1 `RGBA8` textures.
  * Uniforms for `h_min/h_max` normalization (auto from DEM stats in CPU).
* **Acceptance:** Simple CPU test verifies mapping at known heights.
* **Deps:** C1.
* **Estimate:** 0.5 day. (P)

**C5. Terrain pipeline state**

* **Goal:** Create render pipeline for terrain draw.
* **Deliverables:**

  * `terrain/pipeline.rs` with render pipeline (vertex+fragment WGSL).
  * Vertex shader: reconstruct world Z from height texture using `uv`; compute world position.
  * Fragment shader: compute normals (forward differences in texture space), directional light + ambient, tonemap.
* **Acceptance:** Compiles on Metal/Vulkan/DX12; no validation errors; draws with a flat colormap first, then lighting.
* **Deps:** A2, B2, C3, C4.
* **Estimate:** 1–1.5 days.

**C6. Draw path integration**

* **Goal:** Render a single terrain layer into the off‑screen target.
* **Deliverables:**

  * `Scene` holds layers; `Renderer.render_rgba()` encodes passes: clear → terrain.
  * Per‑frame bind updates for uniforms.
  * Optional MSAA toggle off by default (add later).
* **Acceptance:** 1024×1024 DEM renders under \~50 ms on mainstream GPUs (readback excluded), image visually correct.
* **Deps:** A2, B2, C5.
* **Estimate:** 0.5 day.

**C7. Readback & PNG**

* **Goal:** Finish the round‑trip.
* **Deliverables:**

  * `render_rgba() -> np.ndarray[H,W,4]` and `render_png(path)`.
  * Persist a **single** readback buffer sized to frame and reuse per call.
* **Acceptance:** DEM example produces a PNG identical (within SSIM≥0.98) across backends; memory stable across multiple renders.
* **Deps:** C6.
* **Estimate:** 0.5 day.

**C8. Performance & quality passes**

* **Goal:** Hit MVP budgets and polish output.
* **Deliverables:**

  * CPU prepass for `min/max` of DEM; normalize in shader using uniforms.
  * Option to precompute normals on CPU for comparison; keep shader by default.
  * Basic exposure parameter.
* **Acceptance:** 4K render < 200 ms GPU on M‑series/RTX‑class; readback < 60 ms; no major banding after tonemap.
* **Deps:** C7.
* **Estimate:** 0.5 day.

---

## Workstream D — Python API, tests, and docs

**D1. Python API surface (thin)**

* **Goal:** Public methods for Week‑2 demo.
* **Deliverables:**

  * `Renderer.set_camera_orbit(...)`, `Renderer.set_sun(elevation, azimuth)`.
  * `Renderer.add_terrain(...)`.
  * Type hints and docstrings.
* **Acceptance:** Imports clean; help() shows parameters; invalid args raise friendly errors.
* **Deps:** B1, C1.
* **Estimate:** 0.5 day.

**D2. Golden image test**

* **Goal:** Deterministic, cross‑backend test.
* **Deliverables:**

  * Tiny 64×64 synthetic DEM (gradient) checked into `tests/data/`.
  * Test renders → compares against a gold PNG using SSIM/PSNR (allow small tolerance).
* **Acceptance:** Test passes on Vulkan/Metal in CI matrix.
* **Deps:** C6, C7.
* **Estimate:** 0.5 day.

**D3. Example notebook/script**

* **Goal:** Showcase end‑to‑end usage.
* **Deliverables:**

  * `examples/terrain_single_tile.py` producing a 1024×1024 PNG with a nice colormap and sun angles.
* **Acceptance:** Script runs after `maturin develop`; generates image.
* **Deps:** D1, C7.
* **Estimate:** 0.25 day. (P)

**D4. CI updates**

* **Goal:** Exercise terrain in CI.
* **Deliverables:**

  * Extend `.github/workflows/wheels.yml` to run terrain test (headless).
  * Artifact upload of produced PNGs for inspection.
* **Acceptance:** CI green across OSes.
* **Deps:** D2.
* **Estimate:** 0.25 day. (P)

---

## Suggested calendar (10 working days)

**Week 1**

* **Day 1:** A1, A2, B1
* **Day 2:** A3, B2, A4 (P)
* **Day 3:** C1, C2
* **Day 4:** C3 (P), C4 (P)
* **Day 5:** C5 (start)

**Week 2**

* **Day 6:** C5 (finish), C6
* **Day 7:** C7, D1
* **Day 8:** C8 (perf/quality pass)
* **Day 9:** D2 (golden image), D4 (CI)
* **Day 10:** D3 (example), buffer for cross‑platform fixes

Parallelization tips:

* One engineer on **Core/Camera** (A/B) while another starts **Terrain** (C1–C3).
* API/tests (D) can overlap from late Week 1 once `add_terrain` signature is stable.

---

## Technical specifics (to avoid rework)

* **DEM contract (MVP):**

  * `float32` or `float64` (cast to `f32`), shape `(H, W)`, C‑contiguous.
  * `spacing=(dx, dy)` in world units (meters); **Y up**, world XY in meters.
  * Normalize height via `h_min/h_max` uniforms computed on CPU.

* **Vertex layout (terrain grid):**

  * `position.xy` = world plane coords (meters), `uv` = `[0,1]` sampling into R32F height texture.
  * Vertex shader reconstructs `position.z = height(u,v) * exaggeration`.

* **Lighting (fragment WGSL):**

  * Forward difference normal from height texture; sun dir from azimuth/elevation; `albedo = LUT(heightN)`.
  * `color = tonemap(ambient + lambertian)`; write to sRGB target.

* **Performance knobs:**

  * Use `u16` indices up to 65,535 verts; otherwise fall back to `u32`.
  * Persistent readback buffer sized to `(stride * H)`.
  * Single command encoder per frame; one render pass.

---

## Done definition (Week 2)

* `pip install maturin && maturin develop --release`
  → `python -m examples.terrain_single_tile`
  → produces a lit terrain PNG from a NumPy DEM.
* `report_device()` prints backend and limits.
* Golden image test passes in CI on at least **Ubuntu (Vulkan)** and **macOS (Metal)** with **SSIM ≥ 0.98**.
* Public API: `add_terrain`, `set_camera_orbit`, `set_sun`, `render_rgba`, `render_png` documented.

---

**Week 3–4: Vector & Graph layers**

* **Polygons**: earcut tessellation, optional wire overlay.
* **Lines**: screen‑space wide lines via triangle strips (round caps later).
* **Points**: instanced disks/sprites using per‑instance buffers.
* **Graph snapshot**: node scatter + batched edges; optional depth bias.

Excellent—here’s a **fine‑grained, two‑week plan** for **B2. Week 3–4: Vector & Graph layers** (Rust‑first core with Python API via PyO3; rendering on wgpu). It’s split into small tickets you can drop into your tracker, each with **Goal, Deliverables, Acceptance, Deps, Estimate**, and parallelizable items marked **(P)**.

> **B2 exit criteria (recap)**
>
> 1. Ship **Vector layers**: Polygons (fill + outline), Lines (screen‑space width, AA), Points (instanced, size in px).
> 2. Ship **Graph snapshot**: nodes as instanced points, edges as antialiased lines; simple attribute styling.
> 3. Handle large(ish) inputs via batching; produce golden images + timing metrics; document public APIs.

---

## Workstream V0 — Public API & Layer System

### V0.1 Public API definition

* **Goal:** Freeze the Python surface for vectors & graphs.
* **Deliverables:**

  ```python
  # Coordinates must already be planar (projected). See B1 policy.
  Renderer.add_polygons(geoms | packed, *, fill_rgba=(...), line_rgba=(...), line_width=1.0)
  Renderer.add_lines(coords | packed, *, width_px=2.0, rgba=(...), join="miter", cap="butt")
  Renderer.add_points(xy | packed, *, size_px=5.0, rgba=(...), shape="circle")  # circle/square
  Renderer.add_graph(nodes_xy, edges_idx, *,
                     node_size_px=4.0, node_rgba=(...), edge_width_px=1.5, edge_rgba=(...))
  ```

  * Docstrings: planar CRS requirement; size in **pixels** for MVP.
* **Acceptance:** API imports clean; signatures visible in `help()`.
* **Deps:** A3 (packing approach), B1 policy
* **Estimate:** 0.25 d

### V0.2 Layer base & render order

* **Goal:** Consistent draw ordering & shared uniforms.
* **Deliverables:** `enum Layer { Polygons, Lines, Points, GraphNodes, GraphEdges }`, stable sort: fills → lines → points → graph edges → graph nodes (or configurable).
* **Acceptance:** Layers draw in deterministic order; debug labels in command encoder.
* **Deps:** V0.1
* **Estimate:** 0.25 d (P)

---

## Workstream V1 — Common vector core

### V1.1 Packed data contracts (reuse A3)

* **Goal:** Reuse/extend packed formats to avoid Python‑side overhead.
* **Deliverables:**

  * Polygons: `(coords f32 [x,y,...], ring_offsets u32, feature_offsets u32)` (from A3).
  * Lines: `(coords f32, path_offsets u32)` for polylines; each path is a sequence of XYs.
  * Points: `(coords f32 [x,y]*N)`.
* **Acceptance:** Validation errors are precise; no extra copies across FFI.
* **Deps:** A3
* **Estimate:** 0.5 d

### V1.2 Bounds, batching, and visibility

* **Goal:** Keep draw calls reasonable.
* **Deliverables:**

  * Per‑layer AABB; pre‑compute per‑path AABB for lines, per‑feature AABB for polygons.
  * Batch builder to cap **\~100k vertices / draw** (tunable).
  * (Optional) coarse culling: skip batches entirely if off‑camera (frustum test).
* **Acceptance:** Large inputs split into multiple draws; memory footprint logged.
* **Deps:** V1.1
* **Estimate:** 0.75 d (P)

---

## Workstream V2 — Polygons production pass

> A3 did the tessellation spike. Here we make it “product‑ready” and integrate with styles.

### V2.1 Polygon fill pipeline hardening

* **Goal:** Robust fill render with alpha.
* **Deliverables:**

  * FS outputs straight‑alpha; blending configured (`src = One, dst = OneMinusSrcAlpha`).
  * Optional per‑feature color via small SSBO or push constants (single color is okay for MVP; add LUT later).
* **Acceptance:** Donut (hole) renders correctly over varying backgrounds; no halo at seams.
* **Deps:** A3 (triangulation), V0.2
* **Estimate:** 0.5 d

### V2.2 Polygon outline (stroke) refinement

* **Goal:** Clean, consistent outlines for exteriors and holes.
* **Deliverables:**

  * Reuse A3 stroke mesh; ensure layer order: fill first → stroke.
  * Line width currently world‑space or pixel? **For polygons** keep world‑space simple, but clamp min width in device px to avoid vanishing.
* **Acceptance:** 1px equivalent outline visible at all zooms; no cracks on tight angles.
* **Deps:** A3 G3.3, V0.2
* **Estimate:** 0.5 d (P)

---

## Workstream V3 — Lines (screen‑space AA quads)

### V3.1 Polyline packing & validation

* **Goal:** Stable input path for lines.
* **Deliverables:**

  * Python helper `pack_lines(paths)` producing `(coords, path_offsets)`.
  * Reject paths with <2 points; deduplicate consecutive duplicates.
* **Acceptance:** Tests for simple path, L‑shape, and degenerate inputs.
* **Deps:** V1.1
* **Estimate:** 0.5 d

### V3.2 Instanced segment expansion (shader path)

* **Goal:** Antialiased screen‑space width without heavy CPU meshing.
* **Deliverables:**

  * GPU path that draws each **segment** as a **2‑triangle quad** expanded in VS from a unit quad using per‑vertex “side” attribute and **viewport scale**.
  * Provide miters at joins with miter‑limit; fallback to bevel when exceeded. Round joins later (stretch goal).
  * FS does smooth edge AA via distance to edge (simple smoothstep).
* **Acceptance:** Uniform 2px width regardless of zoom; joints look correct for angles ≥ 30°; no gaps.
* **Deps:** V3.1, camera uniforms
* **Estimate:** 1–1.5 d

### V3.3 Caps & joins variants (config)

* **Goal:** Basic join/cap options.
* **Deliverables:** `'cap': 'butt'|'square'|'round'`, `'join': 'miter'|'bevel'` (round later). Implement ‘round’ cap in FS using SDF circle at endpoints.
* **Acceptance:** Visual comparison images; options switch behavior.
* **Deps:** V3.2
* **Estimate:** 0.5 d (P)

### V3.4 Batching & perf logging

* **Goal:** Scale to many segments.
* **Deliverables:**

  * Instance buffer: per‑segment endpoints (`p0, p1`) + color/width; batch draws in \~100k segments.
  * Metrics: #segments, build time, draw time.
* **Acceptance:** 1e6 short segments render in ≤ 300–500 ms at 1080p headless on a mid‑range GPU (target; adjust per hardware).
* **Deps:** V3.2
* **Estimate:** 0.5 d (P)

---

## Workstream V4 — Points (instanced SDF sprites)

### V4.1 Packed points & validation

* **Goal:** Simple input and checks.
* **Deliverables:** `add_points(xy, size_px, rgba, shape)` with a packed `coords` array; drop NaNs.
* **Acceptance:** Correct counts; errors on wrong dtypes/shapes.
* **Deps:** V1.1
* **Estimate:** 0.25 d

### V4.2 Instanced quad expansion

* **Goal:** Screen‑space sized points.
* **Deliverables:**

  * VS expands a unit quad per point using size in **px**; converts to NDC with viewport size.
  * FS draws **SDF circle** (or square) with smooth AA edge; configurable `shape`.
* **Acceptance:** Sizes in pixels stable across zoom; crisp edges at 1–2 px.
* **Deps:** camera uniforms
* **Estimate:** 0.75 d

### V4.3 Attribute styling (size/color arrays)

* **Goal:** Per‑point variability without re‑creating pipelines.
* **Deliverables:**

  * Optional per‑point `size_px` and `rgba` arrays (float32/uint8).
  * Small normalization in shader; clamp sizes to \[1, 64] px by default.
* **Acceptance:** Gradient color/size examples render as expected.
* **Deps:** V4.2
* **Estimate:** 0.5 d (P)

### V4.4 Batching & perf

* **Goal:** Handle very large point sets.
* **Deliverables:** Chunk draws at \~1e5–5e5 instances; reuse instance buffers; metrics.
* **Acceptance:** 1e6 points in ≤ 200–300 ms at 1080p headless on mid‑range GPU (target).
* **Deps:** V4.2
* **Estimate:** 0.5 d (P)

---

## Workstream G — Graph snapshot

### G0 Graph API & ingest

* **Goal:** Define minimal graph inputs.
* **Deliverables:**

  ```python
  add_graph(nodes_xy: np.ndarray[(N,2), f32],
            edges_idx: np.ndarray[(M,2|k), u32],  # 2 for straight segments; k for polyline path indices optional
            *, node_size_px=4.0, node_rgba=(...), edge_width_px=1.5, edge_rgba=(...))
  ```

  * Validate ranges (edge indices < N), reject self‑loops if desired.
* **Acceptance:** Clean errors; stored counts.
* **Deps:** V4.*, V3.*
* **Estimate:** 0.5 d

### G1 Nodes: instanced points (reuse V4)

* **Goal:** Render nodes via the points layer machinery.
* **Deliverables:** Use V4 pipelines with graph‑specific bind groups if needed.
* **Acceptance:** 100k nodes draw within V4 budgets; optional per‑node color/size.
* **Deps:** V4.2
* **Estimate:** 0.25 d (P)

### G2 Edges: straight segments (reuse V3)

* **Goal:** Render edges via the lines pipeline.
* **Deliverables:** Build per‑edge instances from node positions; optional color by degree.
* **Acceptance:** 200k edges render within V3 budgets; no cracks on joins.
* **Deps:** V3.2
* **Estimate:** 0.5 d (P)

### G3 Attribute styling (optional)

* **Goal:** Color/size maps by attribute arrays (degree, community id).
* **Deliverables:**

  * Small in‑shader LUT for categorical communities; numeric ramp for degree.
* **Acceptance:** Visual check with a synthetic graph.
* **Deps:** G1, G2
* **Estimate:** 0.5 d (optional)

---

## Workstream X — Tests, determinism, docs

### X1 Golden images (vectors)

* **Goal:** Visual regression tests for polygons/lines/points.
* **Deliverables:**

  * Tiny scenes: donut, polyline with L‑join, random point cloud; 512×512 goldens; SSIM ≥ 0.99.
* **Acceptance:** CI passes on Linux & macOS.
* **Deps:** V2–V4
* **Estimate:** 0.5 d

### X2 Golden images (graph)

* **Goal:** Graph snapshot test.
* **Deliverables:** Small random graph (N~~1k, M~~2k), layout from spring or fixed grid; goldens.
* **Acceptance:** SSIM ≥ 0.99 vs goldens; perf metrics logged.
* **Deps:** G1–G2
* **Estimate:** 0.25 d (P)

### X3 Performance & memory metrics

* **Goal:** Track budgets.
* **Deliverables:** `Renderer.last_metrics()` extended: vertex/instance counts, #draws, build/upload/draw times, VRAM used (approx by buffer sizes).
* **Acceptance:** Metrics printed during examples and in CI logs.
* **Deps:** V1.2, V3.4, V4.4
* **Estimate:** 0.25 d (P)

### X4 Docs & examples

* **Goal:** User‑facing recipes.
* **Deliverables:**

  * `examples/vectors_basemap.py` (polygons + roads + POIs).
  * `examples/graph_snapshot.py` (nodes/edges).
  * README sections for each layer with API snippets and notes on CRS and pixel sizes.
* **Acceptance:** Both scripts run after `maturin develop` and emit PNGs.
* **Deps:** V2–V4, G1–G2
* **Estimate:** 0.5 d

---

## Suggested 10‑day schedule (2 engineers)

**Week 3**

* **Day 1:** V0.1–V0.2, V1.1, V1.2 (P)
* **Day 2:** V2.1, V2.2 (P)
* **Day 3–4:** V3.1, V3.2; start V3.3 (P)
* **Day 5:** V3.4 (perf), X1 (goldens vectors)

**Week 4**

* **Day 6:** V4.1, V4.2
* **Day 7:** V4.3 (P), V4.4 (P)
* **Day 8:** G0, G1 (P), G2 (P)
* **Day 9:** X2 (graph goldens), X3 (metrics)
* **Day 10:** X4 (examples/docs), buffer for cross‑platform fixes

---

## Technical guardrails & choices

* **CRS:** Enforce planar coordinates for vectors/graph inputs. Refuse lat/lon unless `assume_planar=True` with a loud warning.
* **Pixel‑space sizing:** Lines/points use **pixel sizes** (screen‑space) for MVP. Polygons’ **stroke** may be world‑space with min‑px clamp.
* **Antialiasing:** Implement **shader‑based AA** (sdf/smoothstep) for lines/points; MSAA remains off for determinism.
* **Transparency:** Use straight alpha blending; draw order controls compositing; avoid OIT in MVP.
* **Batching:** Prefer large instance buffers and a few big draws. Keep per‑batch ≤ \~100k–200k instances to avoid huge bind times on some drivers.
* **Determinism:** Avoid derivatives for AA where possible; prefer analytic distances.
* **Perf budgets (targets on mid‑range GPU, 1080p):**

  * **Polygons:** 1M tris ≤ 200–300 ms headless.
  * **Lines:** 1M short segments ≤ 300–500 ms.
  * **Points:** 1M points ≤ 200–300 ms.
  * **Graph:** N=100k, M=200k snapshot ≤ 400–600 ms.

---

## Definition of Done (B2)

* Public APIs for polygons/lines/points/graph are implemented and documented.
* Visual results validated by **golden images** (SSIM ≥ 0.99) across Metal/Vulkan CI jobs.
* Large input batching is in place; metrics are surfaced.
* Examples render clean PNGs demonstrating each layer and the graph snapshot.

---

Great—here’s a **fine‑grained task plan** for the remaining phases, structured so you can drop items into your tracker. I’ve split **Week 5 (Packaging & docs)** and **Week 6 (Perf & polish)** into small tickets with **Goal, Deliverables, Acceptance, Deps, Estimate**, and I close with **MVP Exit Criteria tasks**.
*(Note: “vshade” in your exit criteria looks outdated—package is **vulkan‑forge**. I use that below.)*

---

# Week 5 — Packaging & Docs

## P5.1 Build matrix & reproducible wheels

**Goal:** Produce wheels for Linux (manylinux2014 x86\_64), macOS (arm64 & x86\_64 or universal2), and Windows (x86\_64) with **abi3** (`py3.10+`).
**Deliverables:**

* Update GH Actions workflow:

  * One job per OS/arch (Linux x86\_64, macOS‑arm64, macOS‑x86\_64, Windows x86\_64).
  * `maturin build --release --strip --compatibility manylinux2014` on Linux.
  * Cache via `Swatinem/rust-cache@v2`; pin Rust with `rust-toolchain.toml`.
* Option A: **Two macOS wheels** (arm64 + x86\_64). Option B: **universal2** (one wheel) — choose & document trade‑off.
  **Acceptance:** CI artifacts include wheels for all targets; re‑runs hit caches and finish faster.
  **Deps:** none. **Estimate:** 0.5 d.

## P5.2 Wheel compliance & sdist

**Goal:** Ensure wheels are compliant and an sdist exists.
**Deliverables:**

* Linux: `auditwheel show wheels/*.whl`; macOS (optional): `delocate-listdeps`.
* `maturin sdist -o dist`; `twine check wheels/* dist/*`.
  **Acceptance:** No missing libs; `twine check` passes; local `pip install dist/*.tar.gz` works.
  **Deps:** P5.1. **Estimate:** 0.25 d.

## P5.3 Trusted Publishing & TestPyPI

**Goal:** Manual publish path before 1.0.
**Deliverables:** Secondary workflow (`workflow_dispatch`) that:

* Downloads built wheel artifacts.
* Publishes to **TestPyPI** via `pypa/gh-action-pypi-publish` (OIDC Trusted Publishing).
  **Acceptance:** `pip install -i https://test.pypi.org/simple vulkan-forge` works in a fresh venv.
  **Deps:** P5.1–P5.2. **Estimate:** 0.5 d.

## P5.4 Deterministic image test harness (SSIM)

**Goal:** Golden image tests with SSIM tolerance.
**Deliverables:**

* `tests/golden/` with tiny scenes (triangle, small DEM, donut polygon).
* Pure‑NumPy/Pillow SSIM (or `scikit-image` as dev‑only dep).
* Pytest fixtures to render → compare SSIM (≥ 0.99 default; 0.98 for cross‑backend).
  **Acceptance:** `pytest -q` passes locally and in CI (Linux & macOS).
  **Deps:** A1/A2/A3 spikes done. **Estimate:** 0.5 d.

## P5.5 Golden asset generation & baselines

**Goal:** Reproducible goldens across OS/backends.
**Deliverables:**

* Script to (re)generate goldens on a designated “authoritative” platform (e.g., macOS/Metal).
* Store PNGs + metadata (`scene params, adapter info, hash`).
  **Acceptance:** Fresh goldens regenerate identically on the authoritative machine; CI meets SSIM thresholds.
  **Deps:** P5.4. **Estimate:** 0.25 d.

## P5.6 Example notebooks (3)

**Goal:** User‑facing examples in Jupyter.
**Deliverables:** `examples/notebooks/`

1. **Terrain hillshade** (uses synthetic DEM; deterministic).
2. **City basemap** (synthetic rectangles/roads/points; no heavy deps).
3. **Graph snapshot** (synthetic graph).

* Each notebook: installs `vulkan‑forge`, sets camera, renders, shows PNG, prints metrics.
  **Acceptance:** Run clean via `jupyter nbconvert --execute` in CI (or at least smoke‑execute on one OS).
  **Deps:** A2, A3, B2. **Estimate:** 0.75 d.

## P5.7 Simple video helper (turntable)

**Goal:** Generate turntable videos out‑of‑process via **ffmpeg**.
**Deliverables:**

* Python utility `vulkan_forge.video.turntable()`:

  * Args: `renderer, orbit(center, distance, elev, az_start, az_end, frames), fps, out_path`.
  * Renders frames to a temp dir; calls `ffmpeg -framerate … -i %04d.png -pix_fmt yuv420p out.mp4`.
  * Detects ffmpeg, raises helpful error with install hints.
* CLI entry point `vulkan-forge-turntable` (console\_scripts).
  **Acceptance:** Produces a smooth 360° terrain video locally; falls back gracefully if ffmpeg missing.
  **Deps:** A2 done. **Estimate:** 0.5 d.

## P5.8 Docs refresh (README + API)

**Goal:** Make it easy for newcomers.
**Deliverables:**

* README: install matrix, backend forcing, troubleshooting, links to notebooks.
* API doc (lightweight): `pdoc` or markdown stubs for `Renderer` methods.
  **Acceptance:** New dev can install & run examples in <10 min.
  **Deps:** P5.6. **Estimate:** 0.5 d.

## P5.9 Third‑party licenses

**Goal:** Legal hygiene.
**Deliverables:**

* `cargo-about` (or `cargo-lichking`) config; generate `LICENSES-THIRD-PARTY.txt`.
* Verify PyPI classifiers/license fields.
  **Acceptance:** File generated and included in sdist/wheels; CI step prints summary.
  **Deps:** P5.2. **Estimate:** 0.25 d.

## P5.10 CI polish & artifacts

**Goal:** Better visibility for testers.
**Deliverables:**

* Upload **example PNGs** from tests.
* Cache pip & Cargo; keep CI ≤ \~10–12 min end‑to‑end.
  **Acceptance:** Artifacts visible for each run; reruns are faster.
  **Deps:** P5.1, P5.4. **Estimate:** 0.25 d.

---

# Week 6 — Perf & Polish

## W6.1 Timing & metrics instrumentation

**Goal:** Attribute time to encode/submit/readback.
**Deliverables:**

* `Renderer.render_metrics()` returns dict: `encode_ms, gpu_ms (approx), readback_ms, total_ms, vram_bytes`.
* Optional GPU timestamp queries (wgpu feature‑gated), else CPU wall‑clock around submit/poll.
  **Acceptance:** Metrics printed in examples and CI logs.
  **Deps:** A1/A2. **Estimate:** 0.5 d.

## W6.2 Draw call grouping & sort

**Goal:** Reduce pipeline/BindGroup churn.
**Deliverables:**

* Per‑layer batcher that groups by: pipeline id → bind group set → vertex/index buffer ranges.
* Simple sort before encoding.
  **Acceptance:** On large scenes, draw call count reduced by ≥30% vs naive order; measurable time win.
  **Deps:** B2 vector layers. **Estimate:** 0.75 d.

## W6.3 Persistent staging & readback buffers

**Goal:** Minimize allocations/copies.
**Deliverables:**

* **Upload**: ring buffer for staging (size = worst‑case frame upload), reuse per frame.
* **Readback**: persistent buffer sized to `(padded_row * H)`, reused.
  **Acceptance:** No per‑frame buffer creation in logs; render time variance reduced.
  **Deps:** A1 baseline in place. **Estimate:** 0.5 d.

## W6.4 Adapter/Device diagnostics dump

**Goal:** Actionable environment info for bug reports.
**Deliverables:**

* `vulkan_forge.report_environment()` → JSON string/dict:

  * OS, Python, package version; adapter name, backend, limits/features; wgpu/wgpu‑native versions.
* Wire into examples to save `env.json` next to PNGs.
  **Acceptance:** JSON present; includes all fields; docs show how to attach it to issues.
  **Deps:** none. **Estimate:** 0.5 d.

## W6.5 Prefer‑software flag & fallback

**Goal:** Reliable headless behavior.
**Deliverables:**

* Env var: `VULKAN_FORGE_PREFER_SOFTWARE=1` → `force_fallback_adapter=true`.
* Python param `Renderer(prefer_software: bool=False)` to override.
* Retry logic: hardware → fallback adapter; clear errors if both fail.
  **Acceptance:** CI runs use fallback successfully where needed; `info()` shows adapter type.
  **Deps:** W6.4 (info plumbing). **Estimate:** 0.5 d.

## W6.6 Color/tonemap consistency audit

**Goal:** Avoid double‑gamma & ensure linear workflow.
**Deliverables:**

* Confirm target is `Rgba8UnormSrgb`; ensure shader outputs **linear** colors (GPU hardware does SRGB → display).
* Keep tonemap in linear; avoid extra gamma on write.
  **Acceptance:** Visual parity across backends; no washed‑out or extra‑dark outputs; SSIM improves on goldens.
  **Deps:** A2. **Estimate:** 0.25 d.

## W6.7 Error model & messages

**Goal:** Friendly, consistent Python exceptions.
**Deliverables:**

* Map internal errors to `PyRuntimeError` with clear prefix (`[Device]`, `[Upload]`, `[Tessellation]`).
* Include remediation hints (e.g., “project to planar CRS”, “enable prefer\_software”).
  **Acceptance:** Failing tests show actionable messages; docs include troubleshooting table.
  **Deps:** A2/A3/B2. **Estimate:** 0.5 d.

## W6.8 Logging controls

**Goal:** Debug without noisy defaults.
**Deliverables:**

* Feature‑flag log levels via env var `VULKAN_FORGE_LOG=info|debug|trace`.
* Use `log` + `env_logger` (Rust) gated behind optional feature; default off.
  **Acceptance:** Changing env var changes verbosity; default minimal.
  **Deps:** none. **Estimate:** 0.25 d.

## W6.9 Performance write‑up

**Goal:** Set expectations and tips.
**Deliverables:**

* Doc page/README section: typical frame times for point/line/polygon/terrain; batching guidelines; data size limits.
  **Acceptance:** Published with numbers from your hardware; linked from README.
  **Deps:** W6.1–W6.3. **Estimate:** 0.25 d.

## W6.10 Cross‑platform SSIM sweep

**Goal:** Validate image similarity across OS/backends.
**Deliverables:**

* CI job matrix renders the three example scenes; compares SSIM vs macOS goldens.
* Tolerances: **≥ 0.98** (as per MVP).
  **Acceptance:** All jobs pass; diff artifacts uploaded when failing.
  **Deps:** P5.4–P5.5, Week‑6 polish. **Estimate:** 0.5 d.

---

# MVP Exit Criteria — Task List

> **Target examples reproducible via `pip install vulkan-forge`:**
>
> 1. **DEM hillshade with overlays** (terrain + polygons/lines/points)
> 2. **City basemap polygons/roads**
> 3. **50k‑node graph snapshot**
>    **Cross‑platform SSIM ≥ 0.98**

## E1 Publish candidate wheels

* **Goal:** Make install trivial.
* **Deliverables:** Tag `v0.1.0-rc1`; build wheels; **TestPyPI** publish.
* **Acceptance:** `pip install -i https://test.pypi.org/simple vulkan-forge` works on all OSes.
  **Deps:** P5.1–P5.3. **Estimate:** 0.25 d.

## E2 Example scripts (installable) + datasets

* **Goal:** Reproducible, dependency‑light examples.
* **Deliverables:** Three scripts in `python/examples/` and mirrored notebooks:

  * `terrain_hillshade_with_overlays.py` (synthetic DEM + synthetic vectors).
  * `city_basemap.py` (synthetic blocks/roads/POIs).
  * `graph_50k_snapshot.py` (generate 50k nodes, \~100k edges; deterministic seed).
* Package tiny synthetic datasets or generate on the fly (no downloads).
  **Acceptance:** Each script runs in < 60 s on a mid‑range GPU and emits a PNG (and optional MP4 via P5.7).
  **Deps:** A2, B2. **Estimate:** 0.75 d.

## E3 CLI entry points

* **Goal:** One‑line run after install.
  **Deliverables:** `console_scripts`:

  * `vulkan-forge-example-terrain`
  * `vulkan-forge-example-basemap`
  * `vulkan-forge-example-graph`
    **Acceptance:** Running each CLI generates the expected PNGs.
    **Deps:** E2. **Estimate:** 0.25 d.

## E4 Cross‑platform SSIM CI for examples

* **Goal:** Enforce visual similarity.
  **Deliverables:** CI job renders all three examples and compares to macOS goldens with SSIM ≥ 0.98; uploads diffs on failure.
  **Acceptance:** Matrix green across Linux/Windows/macOS; failing SSIM uploads diff images.
  **Deps:** P5.4–P5.5, W6.10. **Estimate:** 0.5 d.

## E5 Readme “Quick Repro” section

* **Goal:** New users can reproduce in minutes.
  **Deliverables:** Commands to run each CLI and notebook; screenshots; troubleshooting links.
  **Acceptance:** Manual check on a clean machine matches screenshots.
  **Deps:** E2–E3. **Estimate:** 0.25 d.

## E6 Release cut

* **Goal:** Ship v0.1.0.
  **Deliverables:** Release notes; bump version; PyPI publish (Trusted Publishing).
  **Acceptance:** `pip install vulkan-forge` (PyPI) works; wheels pulled by pip for all platforms; examples run.
  **Deps:** E1–E5. **Estimate:** 0.25 d.

---

## Suggested scheduling & ownership

**Week 5 (2 engineers)**

* Eng A: P5.1–P5.3, P5.10, P5.9
* Eng B: P5.4–P5.6, P5.7
* Shared: P5.8

**Week 6 (2 engineers)**

* Eng A: W6.1–W6.3, W6.10
* Eng B: W6.4–W6.8
* Shared: W6.9

**Release (end of Week 6)**

* E1–E6 split across team; 0.5–1 day total.

---

## Checklists (copy/paste)

**Packaging**

* [ ] Linux manylinux2014 wheel
* [ ] macOS arm64 & x86\_64 (or universal2) wheels
* [ ] Windows x86\_64 wheel
* [ ] sdist, `twine check`
* [ ] Trusted Publishing to TestPyPI

**Determinism**

* [ ] Golden tests + SSIM (≥ 0.99 unit, ≥ 0.98 cross‑backend)
* [ ] Goldens regeneration script + metadata

**Docs**

* [ ] 3 notebooks executed (or example scripts)
* [ ] README sections: install matrix, examples, troubleshooting

**Perf & polish**

* [ ] Metrics emitted, staging/readback persistent
* [ ] Draw call grouping
* [ ] `report_environment()` + `prefer_software` flag

**MVP examples**

* [ ] Terrain + overlays PNG
* [ ] Basemap polygons/roads PNG
* [ ] 50k‑node graph PNG
* [ ] CI SSIM gate across OSes

---



---

## 2) FFI & crate layout (already scaffolded)

```
vulkan-forge/
├─ Cargo.toml                      # Rust crate, cdylib
├─ pyproject.toml                  # maturin (build backend)
├─ src/
│  ├─ lib.rs                       # PyO3 module, headless renderer (triangle)
│  └─ shaders/triangle.wgsl
├─ python/
│  ├─ vshade/__init__.py           # re-exports Renderer from Rust
│  └─ examples/triangle.py
└─ .github/workflows/wheels.yml    # CI for wheels + smoke tests
```

**Next files to add during Milestone A2/A3**

* `src/renderer/` (Rust modules): `context.rs`, `scene.rs`, `terrain.rs`, `vectors.rs`.
* `src/shaders/terrain.wgsl`, `fill.wgsl`, `lines.wgsl`.

---

## 3) API sketch (Python, thin over Rust)

```python
import numpy as np
from vshade import Renderer

r = Renderer(width=1920, height=1080)
r.set_clear(0.97, 0.98, 0.99, 1.0)
r.set_camera_orbit(target=(0,0,0), distance=1200, azimuth=120, elevation=35)

# Terrain
r.add_terrain(heightmap=dem_np.astype(np.float32),
              spacing=(30.0, 30.0), exaggeration=1.5,
              colormap="viridis")            # mapped in Rust with a small LUT

# Vectors (require preprojected coords in MVP)
r.add_polygons(polys, fill_rgba=(0,0,0,0.1), line_rgba=(0.2,0.2,0.2,1), line_width=1.5)
r.add_lines(lines, rgba=(0.1,0.1,0.1,1), width=2.0)
r.add_points(xy_array, rgba=(0.8,0.1,0.1,1), size=6.0)

img = r.render_rgba()          # np.ndarray[H,W,4], uint8
r.render_png("frame.png")
```

(For MVP, vectors can accept either packed arrays or a minimal Python-side adapter that extracts coords from GeoPandas/Shapely and passes flat arrays into Rust.)

---


<!-- T41-BEGIN:roadmap-check -->
- [x] T3 — Terrain Shaders & Pipeline
- [x] T4.1 — Scene integration (minimal)
<!-- T41-END:roadmap-check -->
