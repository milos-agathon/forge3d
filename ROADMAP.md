# 0) Quick start with the repo

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

**Goal**: Eliminate non-determinism in the pipeline.

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

## A1.4 Off-screen target & readback path (0.5d)

**Goal**: Correct copy-to-CPU with padding handled identically.

* **Deliverables**

  * Off-screen texture creation (`Rgba8UnormSrgb`, `RENDER_ATTACHMENT|COPY_SRC`).
  * **Row-padding** removal helper (already in starter): confirm with unit test.
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

  * `help(forge3d.Renderer)` shows clear signatures and docstrings.

---

## A1.6 Determinism harness (0.5d)

**Goal**: Automated checks for bitwise identity and quality fallback.

* **Deliverables**

  * Small Python script (and/or pytest) that:

    1. Renders twice → compares **SHA-256** (expect equal).
    2. Optionally toggles backend via `WGPU_BACKEND` env (`metal|vulkan|dx12`) and repeats.
    3. If bytes differ, compute **SSIM** as a fallback—must be ≥ 0.999.
  * Prints adapter name/back-end via `info()`.

* **Acceptance**

  * On at least one machine per OS, **two runs** produce identical hashes.
  * Across backends, either hashes match or SSIM ≥ 0.999.

---

## A1.7 Cross-backend runners (0.5d) (P)

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
* [ ] Verify row-padding removal with a unit test.
* [ ] Add SHA-256 + SSIM fallback checks.
* [ ] Backend-forcing scripts + README.
* [ ] CI: run tests, upload PNGs with hash in filenames.
* [ ] Device report + helpful errors.
* [ ] (Opt) Timing logs + thresholds.

**A2. Terrain pipeline spike** *(\~3–5 days)*

* **Scope**

  * Add `Renderer.add_terrain(heightmap: np.ndarray[f32], spacing: (f32, f32), exaggeration: f32)` in Rust via `numpy` crate.
  * CPU: build indexed grid (triangle list) once.
  * WGSL: sample height, compute normals (forward differences), single directional light, gamma-correct + simple tonemap.
  * Render path: write into the same off-screen texture.
* **Exit criteria:** 1024² DEM renders under 50 ms on M-series/NVIDIA, readback under 30 ms; image test SSIM ≥ 0.98 across backends.

---

## Workstream T0 — API & Data Contract

### T0.1 Public API & validation

* **Goal:** Define and implement the MVP Python→Rust contract.
* **Deliverables:**

  * `Renderer.add_terrain(heightmap: np.ndarray, spacing: tuple[float,float], exaggeration: float = 1.0, *, colormap="viridis")`
  * Accept `float32` (prefer) or `float64` (cast to `f32`), shape `(H, W)`, C-contiguous.
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

* **Goal:** Upload DEM to GPU as a single-channel float texture.
* **Deliverables:**

  * Create `Texture2D` with `R32Float`, usage `TEXTURE_BINDING | COPY_DST`.
  * `queue.write_texture` with **256-byte row alignment** handled (pad rows when necessary).
  * Linear clamp sampler.
* **Acceptance:** Validation passes on Metal/Vulkan/DX12; probe a few texels via a temp compute or debug path (optional).
* **Deps:** T0.1
* **Estimate:** 0.5 d (P)

### T1.3 Colormap LUT texture

* **Goal:** Map heights to colors via a small LUT.
* **Deliverables:**

  * 256×1 `RGBA8UnormSrgb` texture with built-in palettes: `viridis`, `magma`, `terrain`.
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

* **Goal:** Basic single-sun lighting with simple tonemap.
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
  * Per-frame update of `Globals` (view/proj/sun/exposure/h\_range/spacing/exaggeration).
  * Command encoder: begin pass → draw terrain → end.
* **Acceptance:** First image produced from real DEM (even if flat-shaded initially).
* **Deps:** T1–T3
* **Estimate:** 0.5 d

### T4.2 PNG & NumPy round-trip

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

## Suggested 4-day schedule (1–2 engineers)

**Day 1:** T0.1, T0.2 (P), T1.1
**Day 2:** T1.2 (P), T1.3 (P), T2.1
**Day 3:** T2.2 (P), T3.1, T3.2, T3.3 (P)
**Day 4:** T4.1, T4.2 (P), T5.1, T5.2 (P), T5.3 (P)

Parallelization tips:

* One engineer: **Python packing (G1) + tests (G5.1)**.
* Another: **Rust FFI/validation (G2) → earcut (G3.1) → GPU (G4)**.
* If you have a third, hand them **stroke mesh (G3.3)** and **golden tests (G5.2)**.

---

## Technical guardrails & choices

* **CRS/units:** **Planar coordinates only** for A3. Users must project before calling (e.g., EPSG:3857 meters). Add a clear error if coordinates look like lat/lon ranges (|x| ≤ 180 & |y| ≤ 90) unless `assume_planar=True`.
* **Orientation:** Standardize to **exterior CCW**, **holes CW** before earcut; while earcut doesn’t strictly require winding, canonicalization helps downstream logic (e.g., stroke expansion).
* **Holes:** Associate holes with the nearest containing exterior; reject if ambiguous or nested incorrectly.
* **Degeneracy:** Drop rings with <3 unique points after dedup; area < ε²; simplify long nearly-collinear runs with RDP if enabled.
* **Numeric stability:** Translate/scale features locally pre-earcut; rescale back for rendering.
* **Stroke width:** For spike, use a **world-space constant** (e.g., meters). If time permits, add simple screen-space adjustment based on current camera scale.
* **WGPU line primitives:** Don’t rely on wireframe polygon mode or line lists; **always** render outlines as triangle meshes.

---

## Definition of Done (A3)

* `add_polygons(...)` accepts Geo-like inputs (or packed arrays) and validates them.
* Tessellation produces triangle buffers with correct **holes** and **multipolygons**.
* Fill + outline render correctly into a PNG via off-screen target.
* Basic tests: packing/topology + golden images pass in CI; timings are logged.

---

**A4. Wheel/CI spike** *(\~1 day)*

* Use the included GitHub Actions workflow.
* **Exit criteria:** Wheels produced on Ubuntu, Windows, macOS (x64 & Apple Silicon if runner available). Post-build import & render test passes.

> **Exit criteria (A4):**
>
> 1. Build **release wheels** for **Windows (x86\_64)**, **Linux (manylinux2014 x86\_64)**, **macOS (arm64 & x86\_64)**.
> 2. Run a **headless smoke test** that imports the wheel and renders a 512×512 RGBA image.
> 3. Upload wheels as CI artifacts; optionally publish to **TestPyPI** via a manual trigger.
> 4. Jobs are **deterministic** and **fast** (caching enabled); logs include device/backend info.

---

## Strategy choice (decide once)

* **S1 (Recommended)**: **maturin-only pipeline** with `abi3` wheels (already configured) → **build once per OS/arch** (no need to repeat per Python version). Simplest and fastest.
* **S2**: `cibuildwheel` if you want its ecosystem niceties (e.g., manylinux build images, cross-arch docker). Slightly more setup.

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
  * Ensure `pyproject.toml` has `requires-python = ">=3.10"` and module name `forge3d._forge3d`.

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
        run: pip install --no-index --find-links wheels forge3d

      - name: Smoke test render
        env:
          # Prefer software on CI to avoid adapter issues (WARP/D3D12 on Win, Lavapipe on Linux, Metal on mac)
          VULKAN_FORGE_PREFER_SOFTWARE: "1"
        run: |
          python - << 'PY'
          import hashlib, io, sys
          import numpy as np
          from PIL import Image
          import forge3d
          r = forge3d.Renderer(512,512)
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

**Goal:** One-button publish to TestPyPI from CI artifacts.

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

  * Package appears on TestPyPI; `pip install -i https://test.pypi.org/simple forge3d` works on a fresh VM.

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

  * Badges show status; instructions are copy-pasteable.

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

## Suggested 2-day schedule

**Day 1:** A4.1–A4.4
**Day 2:** A4.5–A4.11

Parallelizable: A4.5/A4.6 can run while A4.7/A4.8 are drafted.

---

## Pitfalls & guardrails

* **Linux GPU stack in CI**: Don’t rely on a display; use surface-less off-screen targets (you already do). Prefer software adapters in CI via `VULKAN_FORGE_PREFER_SOFTWARE=1`.
* **abi3 sanity**: With `pyo3` abi3, you **don’t need** to build per Python version—simplify the matrix to one Python (e.g., 3.12). Keep runtime tests with that Python only.
* **manylinux**: Always pass `--compatibility manylinux2014` on Linux builds.
* **Universal2?** If you want **one** macOS wheel instead of two, maturin can build `--universal2` on macOS. Trade-off: longer link times, some crates don’t handle it well. Safe to ship two wheels (arm64 & x86\_64) for now.
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

  * CPU tiling & LOD pyramid (simple quad-tree, memory-resident).
  * Per-tile draw, indexed grid, MSAA=1 for speed, MSAA=4 for stills.
* Python API: thin wrappers, type conversions (NumPy ↔ Rust slices).

---

## Sprint goal (Week 1–2)

* Establish a minimal engine core (device/context, frame targets, camera).
* Implement a **single-tile** terrain pipeline end-to-end: **NumPy DEM → GPU → lit render → PNG/NumPy RGBA**.
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

**A2. Off-screen target & format policy**

* **Goal:** Standardize the render target and readback path.
* **Deliverables:**

  * `core/target.rs` with `RenderTarget { texture, view }`, create/destroy, `Rgba8UnormSrgb`.
  * Readback helper w/ row-padding (refactor current code).
* **Acceptance:** 512×512 triangle renders identically to pre-refactor; round-trip to PNG.
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

  * Decision doc: right-handed, +Y up, camera orbit params, degrees vs radians.
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

## Workstream C — Terrain (single tile, end-to-end)

**C1. DEM FFI contract**

* **Goal:** Robust acceptance of NumPy DEM arrays in Rust.
* **Deliverables:**

  * Python: `add_terrain(heightmap: np.ndarray[f32|f64], spacing: tuple[float,float], exaggeration: float=1.0, colormap="viridis")`
  * Rust: `PyArray2<f32>` path (cast f64→f32 if needed), C-contig check, shape `(H,W)`.
  * Store in `TerrainLayer` struct with metadata (dx, dy, z\_scale).
* **Acceptance:** Type/shape errors raise clean `PyRuntimeError`; unit tests for dtype coercion.
* **Deps:** A1.
* **Estimate:** 0.5 day.

**C2. Grid mesh generator (CPU)**

* **Goal:** Build an indexed grid once; reuse VBO/IBO.
* **Deliverables:**

  * `terrain/mesh.rs` with `make_grid(width, height)` producing positions as (x,y) in world-meters plane and index buffer (u16 when possible, else u32).
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

  * Built-in LUTs (`viridis`, `magma`, `terrain`) baked as 256×1 `RGBA8` textures.
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

* **Goal:** Render a single terrain layer into the off-screen target.
* **Deliverables:**

  * `Scene` holds layers; `Renderer.render_rgba()` encodes passes: clear → terrain.
  * Per-frame bind updates for uniforms.
  * Optional MSAA toggle off by default (add later).
* **Acceptance:** 1024×1024 DEM renders under \~50 ms on mainstream GPUs (readback excluded), image visually correct.
* **Deps:** A2, B2, C5.
* **Estimate:** 0.5 day.

**C7. Readback & PNG**

* **Goal:** Finish the round-trip.
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
* **Acceptance:** 4K render < 200 ms GPU on M-series/RTX-class; readback < 60 ms; no major banding after tonemap.
* **Deps:** C7.
* **Estimate:** 0.5 day.

---

## Workstream D — Python API, tests, and docs

**D1. Python API surface (thin)**

* **Deps:** B1, C1.
* **Estimate:** 0.5 day.

**D2. Golden image test**

* **Goal:** Deterministic, cross-backend test.
* **Deliverables:**

  * Tiny 64×64 synthetic DEM (gradient) checked into `tests/data/`.
  * Test renders → compares against a gold PNG using SSIM/PSNR (allow small tolerance).
* **Acceptance:** Test passes on Vulkan/Metal in CI matrix.
* **Deps:** C6, C7.
* **Estimate:** 0.5 day.

**D3. Example notebook/script**

* **Goal:** Showcase end-to-end usage.
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
* **Day 10:** D3 (example), buffer for cross-platform fixes

Parallelization tips:

* One engineer on **Core/Camera** (A/B) while another starts **Terrain** (C1–C3).
* API/tests (D) can overlap from late Week 1 once `add_terrain` signature is stable.

---

## Technical specifics (to avoid rework)

* **DEM contract (MVP):**

  * `float32` or `float64` (cast to `f32`), shape `(H, W)`, C-contiguous.
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
* **Lines**: screen-space wide lines via triangle strips (round caps later).
* **Points**: instanced disks/sprites using per-instance buffers.
* **Graph snapshot**: node scatter + batched edges; optional depth bias.

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

* **Goal:** Reuse/extend packed formats to avoid Python-side overhead.
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

  * Per-layer AABB; pre-compute per-path AABB for lines, per-feature AABB for polygons.
  * Batch builder to cap **\~100k vertices / draw** (tunable).
  * (Optional) coarse culling: skip batches entirely if off-camera (frustum test).
* **Acceptance:** Large inputs split into multiple draws; memory footprint logged.
* **Deps:** V1.1
* **Estimate:** 0.75 d (P)

---

## Workstream V2 — Polygons production pass

> A3 did the tessellation spike. Here we make it “product-ready” and integrate with styles.

### V2.1 Polygon fill pipeline hardening

* **Goal:** Robust fill render with alpha.
* **Deliverables:**

  * FS outputs straight-alpha; blending configured (`src = One, dst = OneMinusSrcAlpha`).
  * Optional per-feature color via small SSBO or push constants (single color is okay for MVP; add LUT later).
* **Acceptance:** Donut (hole) renders correctly over varying backgrounds; no halo at seams.
* **Deps:** A3 (triangulation), V0.2
* **Estimate:** 0.5 d

### V2.2 Polygon outline (stroke) refinement

* **Goal:** Clean, consistent outlines for exteriors and holes.
* **Deliverables:**

  * Reuse A3 stroke mesh; ensure layer order: fill first → stroke.
  * Line width currently world-space or pixel? **For polygons** keep world-space simple, but clamp min width in device px to avoid vanishing.
* **Acceptance:** 1px equivalent outline visible at all zooms; no cracks on tight angles.
* **Deps:** A3 G3.3, V0.2
* **Estimate:** 0.5 d (P)

---

## Workstream V3 — Lines (screen-space AA quads)

### V3.1 Polyline packing & validation

* **Goal:** Stable input path for lines.
* **Deliverables:**

  * Python helper `pack_lines(paths)` producing `(coords, path_offsets)`.
  * Reject paths with <2 points; deduplicate consecutive duplicates.
* **Acceptance:** Tests for simple path, L-shape, and degenerate inputs.
* **Deps:** V1.1
* **Estimate:** 0.5 d

### V3.2 Instanced segment expansion (shader path)

* **Goal:** Antialiased screen-space width without heavy CPU meshing.
* **Deliverables:**

  * GPU path that draws each **segment** as a **2-triangle quad** expanded in VS from a unit quad using per-vertex “side” attribute and **viewport scale**.
  * Provide miters at joins with miter-limit; fallback to bevel when exceeded. Round joins later (stretch goal).
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

  * Instance buffer: per-segment endpoints (`p0, p1`) + color/width; batch draws in \~100k segments.
  * Metrics: #segments, build time, draw time.
* **Acceptance:** For 1e5 segments, build time and draw time logged; batching prevents giant single draws; AA edges remain stable.
* **Deps:** V3.2
* **Estimate:** 0.5 d

---

## Workstream V4 — Points (instanced)

> **Goal:** Render many points with screen-space sizing.
> **Deliverables:** Instanced quads expanded in VS; per-instance pos/size/color; FS disc SDF for circle; clamp min size; batching.
> **Acceptance:** For 1e6 points on mid-range GPU, stable 30–60 FPS headroom offline; golden image tests for sizes/overlap.

---

## Workstream V5 — Graph snapshot

### V5.1 Node+edge packing

* **Goal:** Efficient static packing for single-frame graph render.
* **Deliverables:** Node positions buffer + edge index buffer; optional per-node size/color.
* **Acceptance:** 1e5 nodes / 2e5 edges pack under a few hundred ms; memory footprint logged.
* **Deps:** V4, V3
* **Estimate:** 0.5–1 d

### V5.2 Render pipelines

* **Goal:** Separate pipelines for edges (lines) and nodes (points).
* **Deliverables:** Reuse lines/points implementations; order edges → nodes to avoid overdraw artifacts.
* **Acceptance:** Golden image tests for a small graph snapshot; AA edges and proper node layering.
* **Deps:** V3, V4
* **Estimate:** 0.5 d

---

## Workstream V6 — Tests & docs

### V6.1 Goldens & metrics

* **Goal:** Deterministic visual baselines.
* **Deliverables:** Tiny goldens for polygons/lines/points/graph; SSIM ≥ 0.99; log counts & timings.
* **Acceptance:** CI passes on Linux/macOS; artifacts uploaded.
* **Deps:** V2–V5
* **Estimate:** 0.5 d

### V6.2 README usage

* **Goal:** Explain vector/graph layer usage & constraints.
* **Deliverables:** Examples and notes on CRS requirements, batching, AA behavior.
* **Acceptance:** Build succeeds; example renders; badges show success; clear instructions.
* **Deps:** A4.3

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

## Workstream D — tests & docs (terrain)

**D2. Golden image test**

* **Goal:** Deterministic, cross-backend test.
* **Deliverables:**

  * Tiny 64×64 synthetic DEM (gradient) checked into `tests/data/`.
  * Test renders → compares against a gold PNG using SSIM/PSNR (allow small tolerance).
* **Acceptance:** Test passes on Vulkan/Metal in CI matrix.
* **Deps:** C6, C7.
* **Estimate:** 0.5 day.

---

## Workstream D — examples (terrain)

**D3. Example notebook/script**

* **Goal:** Showcase end-to-end usage.
* **Deliverables:**

  * `examples/terrain_single_tile.py` producing a 1024×1024 PNG with a nice colormap and sun angles.
* **Acceptance:** Script runs after `maturin develop`; generates image.
* **Deps:** D1, C7.
* **Estimate:** 0.25 day. (P)

---

## Workstream D — CI (terrain)

**D4. CI updates**

* **Goal:** Exercise terrain in CI.
* **Deliverables:**

  * Extend `.github/workflows/wheels.yml` to run terrain test (headless).
  * Artifact upload of produced PNGs for inspection.
* **Acceptance:** CI green across OSes.
* **Deps:** D2.
* **Estimate:** 0.25 day. (P)

---

## Workstream C — performance & quality (terrain)

**C8. Performance & quality passes**

* **Goal:** Hit MVP budgets and polish output.
* **Deliverables:**

  * CPU prepass for `min/max` of DEM; normalize in shader using uniforms.
  * Option to precompute normals on CPU for comparison; keep shader by default.
  * Basic exposure parameter.
* **Acceptance:** 4K render < 200 ms GPU on M-series/RTX-class; readback < 60 ms; no major banding after tonemap.
* **Deps:** C7.
* **Estimate:** 0.5 day.

---

## Workstream B — camera/uniforms

**B1. Coordinate system & camera spec**

* **Goal:** Fix conventions used across CPU/GPU.
* **Deliverables:**

  * Decision doc: right-handed, +Y up, camera orbit params, degrees vs radians.
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

## Workstream A — core scaffolding

**A4. Color management & tonemap stub**

* **Goal:** Linear workflow, sRGB output, simple tonemap.
* **Deliverables:**

  * WGSL tonemap functions (`reinhard` + gamma 2.2).
  * Document texture/view format choices.
* **Acceptance:** Shader compiles; triangle path unchanged; unit test of function values (CPU ref).
* **Deps:** A2.
* **Estimate:** 0.5 day. (P)

**A3. Device diagnostics & feature gating**

* **Goal:** Deterministic device info and conservative limits.
* **Deliverables:**

  * `Renderer.report_device() -> str` (name, backend, limits).
  * Internal `DeviceCaps` (max texture size, MSAA supported).
* **Acceptance:** Reports match backend; MSAA off by default if unsupported.
* **Deps:** A1.
* **Estimate:** 0.5 day.

**A2. Off-screen target & format policy**

* **Goal:** Standardize the render target and readback path.
* **Deliverables:**

  * `core/target.rs` with `RenderTarget { texture, view }`, create/destroy, `Rgba8UnormSrgb`.
  * Readback helper w/ row-padding (refactor current code).
* **Acceptance:** 512×512 triangle renders identically to pre-refactor; round-trip to PNG.
* **Deps:** A1.
* **Estimate:** 0.5 day.

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

---

4) **Unresolved / Needs Author Review**

- *(none detected)*

---

5) **Verification Checklist**

- [ ] IDs are unique and unchanged (A*, T*, G*, V*, etc.).
- [ ] Major section order is unchanged from the original.
- [ ] All anchors/headings remain intact; no anchor text was renamed.
- [ ] Acceptance Criteria sections are present where originally defined; only exact duplicates were removed.
- [ ] Code fences are balanced and unchanged in content.
- [ ] Tables and lists render correctly (no structural changes beyond duplicate removals).
- [ ] Internal references and examples still point to existing sections.
- [ ] No new content or claims were added; wording preserved except for removed duplicates.
- [ ] Front-matter/license blocks (if any) remain byte-identical.
- [ ] Final Markdown passes a renderer preview without errors.

# ROADMAP.md — **forge3d** (WebGPU/wgpu) — Updated with 9 Tutorial Lessons

> This version integrates nine WebGPU tutorial lessons into the existing plan, adds missing foundations (uniforms, bind groups, storage buffers, instancing, cube textures/skybox, canvas/presentation nuances, depth, mips), and preserves prior milestones (determinism, headless/off-screen rendering, terrain spike, wheel/CI). Existing sections from the previous roadmap are retained and expanded where relevant.&#x20;

---

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
* macOS needs Xcode CLT; Windows needs MSVC Build Tools (C++ workload).&#x20;

---

## 1) Deliverables & milestones (Option B)

### Milestone A — Spikes (1–2 weeks total)

**A1. Headless rendering spike** ✅ *Done in the starter*

* **Exit criteria:** Produce a deterministic 512×512 PNG across OS backends; return an `H×W×4` `uint8` NumPy array.&#x20;

---

### A1.1 Repository + build sanity (0.5d)

**Goal**: Everyone can build and run the starter consistently.

* **Deliverables**

  * Confirm local build: `maturin develop --release` works on macOS, Linux, Windows.
  * Document minimal toolchains (macOS CLT, Windows MSVC Build Tools, Python 3.10–3.12).
* **Acceptance**

  * `python -m examples.triangle` produces `triangle.png`.
  * Version pins committed (Rust toolchain in `rust-toolchain.toml` optional).
* **Notes**: Pin `pyo3`, `wgpu` minor versions in `Cargo.toml`.&#x20;

---

### A1.2 Deterministic render pipeline setup (0.5d)

* **Deliverables**: No MSAA; fixed clear color; fixed viewport/scissor; target `Rgba8UnormSrgb`.
* **Acceptance**: Repeated runs produce identical PNG bytes (hash match).&#x20;

---

### A1.3 Shader + geometry determinism (0.5d)

* **Deliverables**: Canonical triangle (no uniforms); pure linear interpolation.
* **Acceptance**: Edge pixels stable; consistent coverage.&#x20;

---

### A1.4 Off-screen target & readback path (0.5d)

* **Deliverables**: Off-screen `Rgba8UnormSrgb` texture; row-padding removal; persistent readback buffer.
* **Acceptance**: `(512,512,4)` `uint8`; unit test validates padding logic.&#x20;

---

### A1.5 Python API surface (0.25d)

* **Deliverables**: `Renderer(width, height)`, `render_triangle_rgba()`, `render_triangle_png(path)`, `info()`; type hints/docstrings.
* **Acceptance**: `help(forge3d.Renderer)` is clear.&#x20;

---

### A1.6 Determinism harness (0.5d)

* **Deliverables**: SHA-256 equality + SSIM ≥ 0.999 fallback; backend toggling via `WGPU_BACKEND`.
* **Acceptance**: Identical hashes per run; SSIM fallback passes across backends.&#x20;

---

### A1.7 Cross-backend runners (0.5d) (P)

* **Deliverables**: Scripts to run Metal/Vulkan/DX12 with determinism harness.
* **Acceptance**: All available backends pass criteria.&#x20;

---

### A1.8 CI matrix & artifacts (0.5–1d)

* **Deliverables**: Wheels + headless determinism test in CI; upload `triangle-<backend>-<sha>.png`.
* **Acceptance**: Green on Ubuntu/Windows/macOS; artifacts stable.&#x20;

---

### A1.9 Device diagnostics & failure modes (0.25d)

* **Deliverables**: `Renderer.report_device()`; clear errors/tips on adapter failure.
* **Acceptance**: Helpful message on unsupported hosts/VMs.&#x20;

---

### A1.10 Performance sanity (optional, 0.25d)

* **Deliverables**: Timing logs for encode/submit/map; perf threshold in CI.
* **Acceptance**: Meets threshold on CI runners.&#x20;

---

### A1.11 Documentation updates (0.25d)

* **Deliverables**: README sections for spike, backend forcing & hashes, troubleshooting.
* **Acceptance**: Fresh dev reproduces in <10 minutes.&#x20;

---

### **A2. Terrain pipeline spike** (\~3–5 days)

*(T0–T5 below summarize the current terrain plan; unchanged, kept verbatim in spirit and trimmed here for focus. Full details remain from the original file and are still in scope.)*&#x20;

* **T0 — API & Data Contract**: Python→Rust contract, height range stats/overrides.
* **T1 — CPU Mesh & GPU Resources**: Indexed grid; DEM as `R32Float` texture; colormap LUT.
* **T2 — Uniforms/Camera/Lighting**: `Globals` UBO with view/proj/sun/exposure/ranges.
* **T3 — Terrain Shaders & Pipeline**: VS height reconstruction; FS normals + Lambert; pipeline layout/bindings.
* **T4 — Integration & Output**: Scene layer; reuse readback; PNG/NumPy parity.
* **T5 — Tests/Timing/Docs**: Synthetic DEMs + goldens; timing harness; example scripts.&#x20;

---

## 2) **New Workstreams from the 9 WebGPU Lessons**

> These add fundamentals we will implement in Rust/**wgpu** with headless targets and tiny browser demos when useful. Each item cites the lesson it came from using `:contentReference[oaicite:0]{index=N}` and an inline file marker.

### U) Uniforms & Bind Groups

**U1. Single-struct uniforms (color/scale/offset)**

* **Deliverables**

  * Rust `wgpu` example+test mirroring a single `@group(0) @binding(0)` uniform struct with `color: vec4f`, `scale: vec2f`, `offset: vec2f`.
  * Bind group set once per draw; update `scale` by aspect each frame.
* **Acceptance**

  * 100 triangles render with per-object color/offset by writing different uniform buffers; aspect-correct scale verified in image test.
  * Ported headless example produces deterministic PNG.
  * Source: &#x20;

**U2. Many objects via **per-object uniform buffers****

* **Deliverables**

  * Allocate one uniform buffer **per object**; create one bind group per object; loop setBindGroup/draw.
* **Acceptance**

  * 100 objects render; functional but measured overhead of bind group churn logged for comparison with storage-buffer approach (S2).
  * Source: &#x20;

**U3. Split uniforms across multiple UBOs**

* **Deliverables**

  * Two uniform buffers: static (`color+offset`) and dynamic (`scale`), each with its own binding; update only dynamic per frame.
* **Acceptance**

  * Byte-accurate updates to dynamic UBO only; frame time improvement vs U2 recorded.
  * Source: &#x20;

---

### S) Storage Buffers & Instancing

**S1. Replace uniforms with storage buffers (one struct per bind)**

* **Deliverables**

  * Mirror U3 but with `var<storage, read>` buffers for the same structs.
* **Acceptance**

  * Visual parity with U3; API plumbing differs; micro-benchmark notes included.
  * Source: &#x20;

**S2. **Array-of-structs** + `@builtin(instance_index)` instancing**

* **Deliverables**

  * Two SSBOs: `array<OurStruct>` and `array<OtherStruct>`; single bind group; draw once with `draw(vertex_count, instance_count)`.
* **Acceptance**

  * 100 instances render via one draw; functional image test; per-instance `scale/offset/color` applied; aspect handled in host.
  * Source: &#x20;

**S3. Split-buffers (static vs dynamic) performance demo**

* **Deliverables**

  * “Minimal changes” variant with one static SSBO (color+offset) and one dynamic SSBO (scale) updated each frame; per-object bind groups.
* **Acceptance**

  * Correctness parity; captures perf impact of per-object bind groups vs S2’s single bind group + instancing.
  * Source: &#x20;

---

### P) Presentation & Canvas/Surface Nuances (for demos)

> Our core is headless, but browser demos and on-screen tests are helpful.

**P1. Canvas configure & presentation format**

* **Deliverables**

  * Tiny demo that configures a canvas WebGPU context using `navigator.gpu.getPreferredCanvasFormat()` and renders the basic triangle.
* **Acceptance**

  * Demo runs; parity triangle renders.
  * Source: &#x20;

**P2. CSS-scaled canvas vs backing store**

* **Deliverables**

  * Document the difference between CSS size and `canvas.width/height`; keep the backing store fixed for determinism tests.
* **Acceptance**

  * Demo renders correctly with CSS fill; roadmap note codifies headless settings.
  * Source: &#x20;

**P3. Resize handling + clamping to device limits**

* **Deliverables**

  * Example with `ResizeObserver`; clamp width/height to `device.limits.maxTextureDimension2D`; re-render on resize.
* **Acceptance**

  * Verified no exceptions at extreme sizes; off-screen path mirrors clamping logic.
  * Source: &#x20;

---

### T) Textures, Samplers, Mips, Cube Maps (Skybox)

**T1. Cube texture sampling + sampler setup**

* **Deliverables**

  * Create a `TextureViewDimension::Cube` (wgpu) and sample in FS using `textureSample` with a linear sampler.
* **Acceptance**

  * Headless sample renders a skybox-like lit background (or a solid diagnostic if images unavailable).
  * Source: &#x20;

**T2. View-direction projection inverse in uniform buffer**

* **Deliverables**

  * Compute `viewDirProjInv` on host; write 4×4 matrix to UBO; reconstruct direction in FS to sample the cubemap.
* **Acceptance**

  * Image changes as camera orbits; verified with a golden series.
  * Source: &#x20;

**T3. Mipmap generation via render pass (GPU)**

* **Deliverables**

  * Off-screen pipeline to generate mips by successively rendering the previous level to the next (linear sampling).
* **Acceptance**

  * Mip chain complete; minification looks correct; no validation errors; measurable bandwidth reduction when zoomed out.
  * Source: &#x20;

**T4. `copyExternalImageToTexture` + `flipY`**

* **Deliverables**

  * Demo load path for external image bitmaps; optional `flipY=false/true` discussion; retained as browser-only helper.
* **Acceptance**

  * Images load into a texture array; visually correct orientation.
  * Source: &#x20;

---

### D) Depth & Stencil

**D1. Depth buffer enablement**

* **Deliverables**

  * Add depth attachment to pipeline (`depth24plus`), `less-equal` compare to allow skybox to pass at infinity.
* **Acceptance**

  * No z-fighting; skybox draws behind all geometry; validation passes.
  * Source: &#x20;

---

### B) Blending & Alpha

**B1. Premultiplied alpha presentation**

* **Deliverables**

  * Note/demonstrate `alphaMode: 'premultiplied'` when configuring the surface; document implications vs off-screen renders (opaque) for determinism.
* **Acceptance**

  * Demo shows correct compositing with translucent content; headless path remains explicitly opaque for bit-exact PNGs.
  * Source: &#x20;

---

### F) Formats & Color

**F1. sRGB render target consistency**

* **Deliverables**

  * Keep `Rgba8UnormSrgb` for targets; clarify gamma in headless and demo paths; validate consistent conversion.
* **Acceptance**

  * Golden images consistent across backends; doc blurb in README.
  * Reinforces existing A1.2 choices.&#x20;

---

### O) Performance Patterns & API Hygiene

**O1. Bind group churn vs single-bind instancing**

* **Deliverables**

  * Micro-bench that compares U2/U3 (per-object bind groups) vs S2 (single bind group + instancing).
* **Acceptance**

  * Timing table committed; recommendation recorded (“prefer S2 for many objects”).
  * Sources:  , &#x20;

**O2. Resize-safe pipelines**

* **Deliverables**

  * Ensure attachments and depth textures are recreated on resize; clamp to device limits.
* **Acceptance**

  * No crashes on continuous resizes; headless mirrors clamping logic.
  * Source:   and &#x20;

---

## 3) Implementation checklist (updated)

* [ ] **Uniforms:** Single struct UBO example (U1). &#x20;
* [ ] **Uniforms (split):** Static + dynamic UBOs (U3). &#x20;
* [ ] **Storage buffers:** SSBO parity + AoS instancing (S1–S2). &#x20;
* [ ] **Minimal SSBO changes:** Static/dynamic split (S3). &#x20;
* [ ] **Presentation:** Canvas CSS & resize demo (P2–P3).  , &#x20;
* [ ] **Depth:** `depth24plus` + `less-equal` (D1). &#x20;
* [ ] **Skybox:** Cube texture + mips + viewDirProjInv (T1–T3). &#x20;
* [ ] **Alpha:** Premultiplied presentation note (B1). &#x20;
* [ ] **sRGB:** Re-assert render target (`Rgba8UnormSrgb`) (F1).&#x20;

---

## 4) Extracted topics from the 9 lessons (normalized)

1. **Basic pipeline bootstrapping** — adapter/device; canvas context; configure with preferred format; minimal VS/FS returning a fixed color; single render pass & submit. &#x20;
2. **CSS size vs backing size** — CSS fills container but you still manage `canvas.width/height` separately for resolution. &#x20;
3. **Resize flow** — `ResizeObserver`, clamp to `device.limits.maxTextureDimension2D`, and re-render. &#x20;
4. **Uniforms (single struct)** — `color/scale/offset` in a UBO; update with `queue.writeBuffer`; aspect-dependent scaling. &#x20;
5. **Uniforms (many objects)** — per-object uniform buffer + per-object bind group; simple loop. &#x20;
6. **Uniforms split** — static vs dynamic values in separate UBOs to reduce writes. &#x20;
7. **Storage buffers, per-object** — SSBOs used like uniforms (`var<storage, read>`). &#x20;
8. **Instancing with SSBO arrays** — `array<struct>` + `@builtin(instance_index)`; single bind group + one instanced draw. &#x20;
9. **Skybox** — cube textures + sampler; premultiplied alpha on surface; enable depth; compute `viewDirProjInv`; GPU mip generation; `copyExternalImageToTexture`.   &#x20;

---

## 5) Gap analysis vs the existing roadmap

**Gaps newly covered**

* **Uniforms & Bind Groups (single, many, split):** Not explicitly specified in A-series; added U1–U3.  ,  , &#x20;
* **Storage Buffers & Instancing:** New S1–S3 cover SSBOs and `@builtin(instance_index)`.  , &#x20;
* **Skybox / Cube textures / Mips:** New T1–T3. &#x20;
* **Depth enablement & less-equal for skybox:** New D1. &#x20;
* **Presentation nuances (CSS sizing, resize, limits):** New P1–P3 (demo-only).  , &#x20;

**Reinforcement overlaps (tutorial deepens existing topics)**

* **sRGB target format** reaffirmed (A1.2 ↔ F1).&#x20;
* **Deterministic pipeline & clear color** shown again in minimal triangle lessons. &#x20;
* **Resource re-creation on resize** complements our off-screen texture lifetime practices. &#x20;

---

## 6) Suggested 4-day schedule (revised) — adding new tracks

**Day 1:** U1, U2 → basic/multi-object uniforms; P1 demo
**Day 2:** U3; S1 (SSBO parity)
**Day 3:** S2 (instancing AoS); S3 (minimal split) + O1 micro-bench
**Day 4:** T1–T3 (cube, mips, viewProjInv) + D1; finalize P2–P3 notes

Parallelization: Skybox (T\*) can proceed independently once a presentable target exists.

---

## 7) CI & packaging (kept)

The wheel/CI spike (A4.1–A4.4) remains unchanged (abi3 wheels via maturin, matrix/caching, smoke test render, fallback adapter option).&#x20;

---

## 8) Definition of Done (updated)

* Headless renders deterministic across backends (A1.\*).&#x20;
* Terrain spike (T0–T5) produces PNG/NumPy parity and golden tests.&#x20;
* **New DoD items:**

  * **Uniforms/Storage:** U1–U3/S1–S3 examples compile and pass image tests (100 instances).  , &#x20;
  * **Skybox:** Cube map with mips renders correctly; depth works as specified. &#x20;
  * **Docs:** Notes on CSS size, resizing & device limits published (demo-only). &#x20;

---

## 9) What changed vs the old roadmap (summary)

* **Added** new workstreams **U/S/P/T/D/B/F/O** to cover uniforms, storage buffers, instancing, presentation/resize, skybox (cube textures, samplers, mips), depth, alpha premultiplication, and perf patterns. (New)
* **Kept** all A1.\* determinism and A2 (terrain) content intact; added references tying sRGB/clear color and resize practices back to lessons. (Unchanged + reinforced)
* **Expanded** implementation checklist to include lesson-derived tasks with acceptance criteria and image tests. (Expanded)
* **Clarified** demo-only vs headless scope (presentation features are demos; headless path remains opaque & deterministic). (Clarified)

---

## Sources (index mapping for `:contentReference[oaicite:0]{index=…}`)

0. **webgpu-simple-triangle.html** &#x20;
1. **webgpu-simple-triangle-with-canvas-css.html** &#x20;
2. **webgpu-simple-triangle-with-canvas-resize.html** &#x20;
3. **webgpu-simple-triangle-uniforms.html** &#x20;
4. **webgpu-simple-triangle-uniforms-multiple.html** &#x20;
5. **webgpu-simple-triangle-uniforms-split.html** &#x20;
6. **webgpu-simple-triangle-storage-buffer-split.html** &#x20;
7. **webgpu-simple-triangle-storage-split-minimal-changes.html** &#x20;
8. **webgpu-skybox.html** &#x20;

---

# New Tasks (Insertion-Ready)

#### \[TX-01] — Textured Quad Baseline (2D Texture + Sampler)

**Section/Phase:** v0.2.x: Rendering › Textures & Samplers
**Summary (1–3 lines):** Add a minimal headless **wgpu** example that draws a textured quad using a 2D texture (`rgba8unorm`) and a sampler; serve as the foundation for filtering/wrapping/mips. &#x20;
**Deliverables:**

* Rust/**wgpu** headless sample: creates `Texture2D` + `Sampler`, bind group layout (texture/sampler), VS (two triangles via `vertex_index`), FS `textureSample(...)`.
* CPU upload path demonstrating `queue.write_texture` with correct `bytes_per_row` padding.
* Python wrapper `Renderer.render_textured_quad_rgba(image: ndarray|None)`; if `None`, render a built-in 5×7 pattern.
  **Acceptance Criteria:**
* Produces deterministic 512×512 PNG with expected pattern (**pixel hash exact**).
* Validation passes on Metal/Vulkan/DX12; target format `Rgba8UnormSrgb` documented.
  **Dependencies:** Headless pipeline + readback from A1.4; sRGB target from A1.2.&#x20;
  **Risks/Mitigations:** Row-padding mistakes → unit test with synthetic strides; add assert on `bytes_per_row % 256 == 0`.
  **Refs:** &#x20;

#### \[TX-02] — Sampler Matrix: Wrapping & Magnification

**Section/Phase:** v0.2.x: Rendering › Textures & Samplers
**Summary:** Implement an 8-case **sampler** matrix (repeat/clamp × repeat/clamp × nearest/linear mag) and snapshot outputs for regression tests. &#x20;
**Deliverables:**

* Create 8 `Sampler`s covering `{addressModeU,V}∈{repeat,clamp-to-edge}`, `magFilter∈{nearest,linear}`; per-case bind group & render.
* Golden images stored; Python harness to compare.
  **Acceptance Criteria:**
* 8 rendered PNGs differ in expected edge/zoom behavior; **hash-stable** per case on a given backend.
  **Dependencies:** TX-01.
  **Risks/Mitigations:** Backend differences at edges → lock to `clamp-to-edge` for baseline and document expected variance.
  **Refs:** &#x20;

#### \[TX-03] — Minification Visuals (Uniform Scale/Offset)

**Section/Phase:** v0.2.x: Rendering › Textures & Samplers › Minification
**Summary:** Add a test that uses a uniform `{scale, offset}` to draw a *small* quad (minification), validating visual differences under filtering. &#x20;
**Deliverables:**

* Uniform buffer with `{scale: vec2f, offset: vec2f}`; VS multiplies positions accordingly.
* Two runs: `magFilter=linear` vs `nearest` (address modes fixed).
  **Acceptance Criteria:**
* PNG pair shows characteristic minification differences (checked with simple frequency-based metric or golden diff).
  **Dependencies:** TX-01, TX-02.
  **Risks/Mitigations:** CSS `image-rendering` is browser-only → headless ignores; document difference.
  **Refs:** &#x20;

#### \[TX-04] — Mipmap Generation (On-GPU)

**Section/Phase:** v0.2.x: Rendering › Textures & Samplers › Mipmaps
**Summary:** Implement a render-pass mipmap generator for `rgba8unorm` textures and validate minification quality with/without mips. ,  &#x20;
**Deliverables:**

* Utility: `generate_mips(device, texture_view)` that renders level N→N+1 using a full-screen pipeline.
* Test: render a checkerboard at 64×, sample at 1× with minification; compare **PSNR/SSIM** for **w/ mips** vs **no mips**.
  **Acceptance Criteria:**
* Mip chain complete (`mip_level_count == 1 + floor(log2(max(w,h)))`); **SSIM(w/ mips) > SSIM(no mips)** for minified view.
  **Dependencies:** TX-01.
  **Risks/Mitigations:** Format constraints → start with `rgba8unorm` only; assert usage includes `RENDER_ATTACHMENT`.
  **Refs:** ,  &#x20;

#### \[IO-IMG-01] — Image Import (`copyExternalImageToTexture`) + flipY

**Section/Phase:** v0.2.x: I/O › Image Ingest (Web demo + headless parity)
**Summary:** Provide a **browser demo** to import `ImageBitmap/HTMLImageElement` with optional `flipY` and mips; add a **headless** PNG loader path for parity tests. ,  &#x20;
**Deliverables:**

* Web: `createImageBitmap(...,{colorSpaceConversion:'none'})` → `copyExternalImageToTexture({source, flipY}, {texture}, size)`.
* Headless: decode PNG (CPU) → upload via `queue.write_texture`; optional flipY in shader or staging.
* Docs: note color-space conversion disable to avoid double sRGB.
  **Acceptance Criteria:**
* Web + headless produce **matching** PNGs for the same input (within byte-exact or SSIM≥0.999 if gamma differs).
  **Dependencies:** TX-01, TX-04.
  **Risks/Mitigations:** Color-space differences across platforms → lock to `Rgba8UnormSrgb`, disable browser conversions.
  **Refs:** ,  &#x20;

#### \[IO-CAN-01] — Canvas/OffscreenCanvas Import

**Section/Phase:** v0.2.x: I/O › Image Ingest (Web demo)
**Summary:** Demonstrate importing from `HTMLCanvasElement`/**OffscreenCanvas** into a GPU texture, with optional on-GPU mip generation. &#x20;
**Deliverables:**

* Web demo: draw dynamic 2D content on a canvas; call `copyExternalImageToTexture(...)`; generate mips if requested.
* Comparison images: minified view with/without mips.
  **Acceptance Criteria:**
* Visual parity between canvas content and textured quad; mip variant shows smoother minification in golden diff.
  **Dependencies:** TX-04.
  **Risks/Mitigations:** Browser-only; mark as demo (non-blocking for headless DoD).
  **Refs:** &#x20;

#### \[IO-VID-01] — Video Import via `copyExternalImageToTexture`

**Section/Phase:** v0.2.x: I/O › Media Ingest (Web demo)
**Summary:** Import frames from `HTMLVideoElement`, generate mips per frame, and render to a quad; gate playback until first frame is ready. &#x20;
**Deliverables:**

* Web demo: `startPlayingAndWaitForVideo(...)` → copy each frame → (optional) `generateMips(...)`.
* Frame-time log and dropped-frame counter.
  **Acceptance Criteria:**
* Sustains ≥30 FPS for 540p sample on a mid-range laptop; no validation errors; visual output updates each frame.
  **Dependencies:** TX-04.
  **Risks/Mitigations:** Browser security/user gesture → click-to-start overlay; handle pause/resume.
  **Refs:** &#x20;

#### \[IO-VID-02] — External Texture (Video)

**Section/Phase:** v0.2.x: I/O › Media Ingest (Web demo; web-only)
**Summary:** Use `texture_external` and `textureSampleBaseClampToEdge(...)` for video sampling (clamped, implementation-defined sampler). &#x20;
**Deliverables:**

* Web demo using `texture_external`; click-to-start, play/pause toggle, requestVideoFrameCallback readiness.
* Doc note: **web-only**, not available in native headless path.
  **Acceptance Criteria:**
* Renders video frames; no out-of-range sampling (edges clamped); stable at ≥30 FPS for 540p sample.
  **Dependencies:** None beyond baseline WebGPU demo infra.
  **Risks/Mitigations:** Platform support variance; feature-detect and show fallback message.
  **Refs:** &#x20;

#### \[IO-CAM-01] — External Texture (Camera)

**Section/Phase:** v0.2.x: I/O › Media Ingest (Web demo; web-only)
**Summary:** Capture camera via `getUserMedia({video:true})` and sample with `texture_external` + `textureSampleBaseClampToEdge(...)`. &#x20;
**Deliverables:**

* Permission flow (overlay + error handling), play/pause toggle, frame callback; render to quad.
* Privacy note and teardown on page hide.
  **Acceptance Criteria:**
* Works with default camera; ≥24 FPS; handles permission denial gracefully.
  **Dependencies:** IO-VID-02 (shared code).
  **Risks/Mitigations:** Permissions/security; provide clear UX + errors.
  **Refs:** &#x20;

---

3. # Reinforcement Overlaps (Advisory)

* **Textures already used in terrain** (height `R32Float`, LUT `RGBA8UnormSrgb`) → tutorials add broader **2D image** sampling patterns and **sampler** options (wrapping/filtering) and **mips**; no task needed for terrain itself. ,   &#x20;
* **Uniform matrices** are already present for terrain camera globals; tutorials use a quad-transform `mat4x4f`—kept as stylistic reinforcement only.  &#x20;
* **sRGB target** and presentation setup match existing determinism choices; tutorials’ import path adds **colorSpaceConversion:'none'** nuance.  &#x20;

---

4. # What Changed (Summary)

* **TX-01** Textured Quad Baseline → *Rendering › Textures & Samplers*.
* **TX-02** Sampler Matrix (wrap/filter) → *Rendering › Textures & Samplers*.
* **TX-03** Minification Visuals via uniforms → *Rendering › Textures & Samplers › Minification*.
* **TX-04** On-GPU Mipmap Generation → *Rendering › Textures & Samplers › Mipmaps*.
* **IO-IMG-01** Image import + flipY (web + headless parity) → *I/O › Image Ingest*.
* **IO-CAN-01** Canvas/OffscreenCanvas import (web) → *I/O › Image Ingest*.
* **IO-VID-01** Video import via `copyExternalImageToTexture` (web) → *I/O › Media Ingest*.
* **IO-VID-02** External video texture (web-only) → *I/O › Media Ingest*.
* **IO-CAM-01** External camera texture (web-only) → *I/O › Media Ingest*.


## New Tasks (Insertion-Ready)

#### \[NEW-SB-01] — Vertex Streams from Storage Buffers (SSBO-as-Vertex)

**Section/Phase:** Milestone B — MVP Implementation › Storage Buffers
**Summary (1–3 lines):** Add support for sourcing vertex positions from `var<storage, read>` buffers in the vertex stage (no `vertexBuffer`), enabling procedural/dynamic meshes and very large shared vertex pools.
**Deliverables:**

* A sample pipeline where the vertex shader fetches `pos[vertex_index]` from a storage buffer and applies per-instance transforms/colors from separate storage buffers.
* Utility to build ring/annulus vertex data on CPU and upload to a storage buffer.
* Bench harness comparing SSBO-vertex vs traditional vertex-buffer path on 100 instances.
  **Acceptance Criteria:**
* Demo renders ≥100 instanced rings (outer/inner radius), positions read only from a storage buffer; colors/offsets/scales from separate storage buffers.
* Frame time at 100 instances within 10% of a baseline VB path for same geometry and state, measured over 300 frames (median excluding first 30 warmup).
* Code path is toggleable at runtime: “VB” ↔ “SSBO” with identical visual result (RMSE of readback over center 256×256 < 1/255).
  **Dependencies:** WebGPU; WGSL storage buffers; existing storage-buffer infra in S-workstream.
  **Risks/Mitigations:** Some drivers may have different perf for SSBO fetch; keep VB path as fallback and gate via feature flag; add CI perf tolerance band.
  **Refs:** `:contentReference[oaicite:0]{index=0}`  &#x20;

---

#### \[NEW-TX-04] — CPU Mip Pyramid Builder & Uploader (rgba8unorm)

**Section/Phase:** Milestone B — MVP Implementation › Textures & Samplers
**Summary (1–3 lines):** Implement CPU-side mip pyramid generation and upload via `queue.writeTexture`, to compare quality/latency vs render-pass-based generation and to support assets/sources without GPU mip access.
**Deliverables:**

* CPU downsampler (box/bilinear) producing full mip chain (1×1) for `rgba8unorm`.
* Uploader that creates a texture with `mipLevelCount = len(mips)` and writes each level via `queue.writeTexture`.
* Visual demo toggling between “blended” and “checker” mip pyramids to reveal filtering behavior.
  **Acceptance Criteria:**
* For a 512×512 source, generated mip count = 10; `queue.writeTexture` calls per level succeed and texture renders.
* Visual parity check: CPU-mipped vs GPU render-pass mipped image—average absolute channel difference over a 256×256 probe < 6/255.
* Runtime toggle between CPU and GPU mip sources with no pipeline changes.
  **Dependencies:** Existing render-pass mip generator (T3).
  **Risks/Mitigations:** CPU gen can stall main thread—perform generation off main tick (microtask/batched chunks) and cache pyramids.
  **Refs:** `:contentReference[oaicite:2]{index=2}` &#x20;

---

#### \[NEW-TX-05] — Mipmap Filter Matrix (mag/min/mipmap combinations + LOD exploration)

**Section/Phase:** Milestone B — MVP Implementation › Textures & Samplers
**Summary (1–3 lines):** Add a matrix demo that renders 8 sampler states (mag ∈ {nearest,linear} × min ∈ {nearest,linear} × mipmap ∈ {nearest,linear}) across varying scale/depth to document practical differences.
**Deliverables:**

* Sampler factory for the 8 combinations; grid renderer showing each combo on tall stretched quads (large z-depth span).
* Click/keyboard to switch between two contrasting mip pyramids (“blended” vs “checker”).
* Optional LOD debug: show computed base mip and weight per pixel row (UI overlay).
  **Acceptance Criteria:**
* All 8 samplers render concurrently; each cell annotated with its (mag,min,mipmap) triple.
* Measurable difference: along a 512-px center scanline, at least 4 of 8 cells differ in >10% pixels vs “linear/linear/linear” baseline.
* Works with CPU-mipped textures from NEW-TX-04.
  **Dependencies:** NEW-TX-04.
  **Risks/Mitigations:** Readback cost—limit probes to 1D scanlines; throttle to on-demand.
  **Refs:** `:contentReference[oaicite:2]{index=2}` `:contentReference[oaicite:1]{index=1}` &#x20;

---

#### \[NEW-VF-01] — Packed Vertex Color Attribute (unorm8x4) + Interleaved Layout

**Section/Phase:** Milestone B — MVP Implementation › Formats & Color
**Summary (1–3 lines):** Introduce interleaved vertex streams with positions (f32x3) and colors packed as `unorm8x4`, validating correct attribute decoding and sRGB output path.
**Deliverables:**

* Mesh builder that writes color bytes into a `Float32Array`’s backing buffer via `Uint8Array` view; set `vertex.buffers[].attributes.format = 'unorm8x4'`.
* Reference cube/hand mesh showcasing per-quad color groups.
* Unit test that verifies GPU output matches CPU-decoded color within 2/255 tolerance on a 64×64 probe.
  **Acceptance Criteria:**
* Pipeline renders colored cube/hand using a single interleaved vertex buffer; no per-vertex WGSL conversions.
* Attribute formats logged and validated at pipeline creation; decoding matches expected colors in readback.
  **Dependencies:** Existing vertex-buffer path.
  **Risks/Mitigations:** Endianness/padding mistakes—add struct layout doc and buffer stride/assert checks.
  **Refs:** `:contentReference[oaicite:3]{index=3}` `:contentReference[oaicite:4]{index=4}` &#x20;

---

#### \[NEW-SG-01] — Scene Graph Core: TRS Hierarchy, World-Matrix Propagation, Draw Walker

**Section/Phase:** Milestone B — MVP Implementation › Scene Graph
**Summary (1–3 lines):** Implement a minimal node system (TRS + parent/child), compute local/world matrices, and traverse to draw attached meshes with per-object uniforms (matrix+color).
**Deliverables:**

* `SceneGraphNode { localMatrix, worldMatrix, translation, rotation, scale, parent, children }` with `updateWorldMatrix()`; re-compute order after structural edits.
* Draw walker that multiplies `viewProjection * worldMatrix` into a per-object UBO and issues `setVertexBuffer/draw`.
* Demo: articulated “hand/arm” hierarchy animating finger joints.
  **Acceptance Criteria:**
* Adding/removing/reparenting nodes updates `worldMatrix` correctly (randomized fuzz test of 10k ops; world transforms CPU-verified).
* Depth test (`depth24plus`) and back-face culling enabled; meshes render in correct front/back order across camera orbits.
* Visual animation matches the tutorial pose evolution within RMSE < 8/255 on a 128×128 readback over 120 frames.
  **Dependencies:** Camera/uniform infrastructure.
  **Risks/Mitigations:** Update-order bugs—maintain dirty flags and topological update; add unit tests for parent/child invariants.
  **Refs:** `:contentReference[oaicite:3]{index=3}` &#x20;

---

#### \[NEW-SG-02] — Aim/Attachment Utilities & Projectile Demo (“Hand Shoot”)

**Section/Phase:** Milestone B — MVP Implementation › Scene Graph
**Summary (1–3 lines):** Add utilities to “aim” a node at a target using basis extraction, attach temporary child nodes (shots), and simulate forward motion in world space.
**Deliverables:**

* `aim(eye, target, up)` and `cameraAim` helpers; `getAxis(worldMatrix, axisIndex)` to extract forward/up vectors.
* Attachment API to spawn a child at a fingertip transform; per-frame update of a simple projectile list with lifetime and velocity.
* Demo scene that fires shots along the finger’s forward axis; UI toggle to animate vs manual pose.
  **Acceptance Criteria:**
* Fired projectiles follow the node’s current forward axis (angle error < 2° vs computed world matrix axis over first 10 frames).
* After window resize, depth texture is recreated and rendering remains correct (no NaNs; depth view dimensions match canvas).
* Shots auto-expire at 5s and detach without leaving GPU resources (no leaking bind groups/buffers verified by resource counters).
  **Dependencies:** NEW-SG-01.
  **Risks/Mitigations:** Numerical drift—normalize velocity each step; clamp timestep for dropped frames.
  **Refs:** `:contentReference[oaicite:4]{index=4}` `:contentReference[oaicite:5]{index=5}`  &#x20;

---

#### \[NEW-SG-03] — Scene Graph Editing: Reparent, Remove, and Stable Traversal

**Section/Phase:** Milestone B — MVP Implementation › Scene Graph
**Summary (1–3 lines):** Provide robust APIs for reparenting/removing nodes and ensure a stable traversal order for rendering after structural edits.
**Deliverables:**

* `setParent(newParent|null)` with cycle prevention; `remove()`; stable child list with insertion indices.
* Fuzz test that performs random reparent/remove operations and validates traversal produces consistent draw order snapshots.
* Example “file cabinets” hierarchy highlighting nested parts and open/close toggles.
  **Acceptance Criteria:**
* 10k random reparent/remove ops complete with zero cycles; traversal count matches node count and remains deterministic across runs.
* Visual toggle opens/closes subtrees and updates in the next frame without missed updates or stale transforms.
  **Dependencies:** NEW-SG-01.
  **Risks/Mitigations:** Event storms—coalesce edits per frame and mark subtree dirty bit for recompute.
  **Refs:** `:contentReference[oaicite:5]{index=5}`&#x20;

---


**Rationale:** Demonstrates passing data from vertex to fragment via an inter-stage struct and default interpolation. \[ref:1]
**Deliverables:**

* WGSL module `triangle_color_struct.wgsl` with `struct VSOut { @builtin(position) position: vec4f, @location(0) color: vec4f }`, vertex returns `VSOut`, fragment takes `VSOut` and returns color. \[ref:1]
* Render test that draws a single triangle to an offscreen 256×256 texture and saves `triangle_struct.png`. \[ref:1]
  **Acceptance Criteria:**
* Center pixel RGB is within ±0.1 of `(0.33, 0.33, 0.33)` (given vertices colored red/green/blue), alpha within ±0.01 of 1.0. \[ref:1]
* No validation errors creating the pipeline or running the draw. \[ref:1]
  **Dependencies:** WebGPU runtime (core), WGSL, offscreen render/readback utility.
  **Risks/Mitigations:** Floating-point tolerances may vary → use epsilon checks (±0.1).

#### Link VS→FS by `@location` without a shared struct

**Rationale:** Inter-stage variables connect by location index; a fragment entry point can take a parameter annotated with `@location(0)` directly. \[ref:1]
**Deliverables:**

* WGSL module `triangle_color_location.wgsl` with `@fragment fn fs(@location(0) color: vec4f) -> @location(0) vec4f`. \[ref:1]
* A/B render harness comparing this shader against `triangle_color_struct.wgsl`; output `diff_location_vs_struct.txt`. \[ref:1]
  **Acceptance Criteria:**
* Mean absolute per-channel pixel difference across the frame ≤ 1/255 vs the struct version. \[ref:1]
* Both pipelines compile and render without errors. \[ref:1]
  **Dependencies:** WebGPU runtime (core), WGSL.
  **Risks/Mitigations:** Mismatched locations lead to black/UB → unit test asserts declared locations align in VS/FS.

#### Use `@builtin(position)` in FS to compute a checkerboard

**Rationale:** `@builtin(position)` is NOT an inter-stage variable; in FS it provides the pixel center coordinate; use it to generate a pattern. \[ref:1]
**Deliverables:**

* WGSL module `checker_by_fs_position.wgsl` where FS reads `@builtin(position)` and returns red/cyan based on `(u32(x)/8 + u32(y)/8) % 2`. \[ref:1]
* Render `64×64` offscreen target to `checker_64.png`. \[ref:1]
  **Acceptance Criteria:**
* Pixel (0,0) is red; pixel (8,0) is cyan; pixel (8,8) is red. \[ref:1]
* Resizing the target to `80×64` preserves tile size in pixels (pattern tied to pixel coords). \[ref:1]
  **Dependencies:** WebGPU runtime (core), WGSL.
  **Risks/Mitigations:** Integer casts must use `vec2u`/`u32` precisely → add compile-time type checks.

#### Split shaders into separate VS/FS modules

**Rationale:** VS and FS can be compiled from separate modules; shared strings are convenience only. \[ref:1]
**Deliverables:**

* `vs_module.wgsl` (returns `@builtin(position)` only) and `fs_module.wgsl` (takes `@builtin(position)`), plus pipeline setup that links them. \[ref:1]
* A/B image diff vs the single-module checkerboard: `separate_modules_diff.txt`. \[ref:1]
  **Acceptance Criteria:**
* L1 mean pixel difference between single-module and split-module outputs ≤ 1/255. \[ref:1]
* Pipeline creation succeeds with distinct `GPUShaderModule`s and the expected entryPoints. \[ref:1]
  **Dependencies:** WebGPU runtime (core), WGSL.
  **Risks/Mitigations:** Entry point/IO mismatch → CI check validates entryPoint names and IO signatures.

#### Implement and exercise WGSL interpolation attributes

**Rationale:** Inter-stage variables support interpolation type (`perspective` default, `linear`, `flat`) and sampling (`center` default, `centroid`, `sample`; `first`/`either` for flat). \[ref:1]
**Deliverables:**

* WGSL variants `interp_perspective_center.wgsl`, `interp_linear_center.wgsl`, `interp_flat_first.wgsl`, `interp_flat_either.wgsl`, `interp_linear_sample_msaa4.wgsl` that annotate `@location(n)` with `@interpolate(...)`. \[ref:1]
* Test harness that draws a full-screen triangle for each variant; for the `sample` case, use a 4× MSAA color target and an atomic counter in a storage buffer to count FS invocations; report `invocations_report.json`. \[ref:1]\[ref:2]
  **Acceptance Criteria:**
* All variants compile and render without validation errors. \[ref:1]
* `flat_first` output is uniform across the triangle and matches the first vertex’s value (within ±1 ULP if float; exact if integer). \[ref:1]
* With `@interpolate(linear, sample)` and MSAA=4, the recorded fragment invocation count is ≥ 3.5× the non-sampled variant over a full-screen draw (edges excluded), indicating per-sample execution. \[ref:1]\[ref:2]
  **Dependencies:** WebGPU runtime (core), WGSL; MSAA render target for `sample`. \[ref:1]\[ref:2]
  **Risks/Mitigations:** GPU/driver differences near edges → exclude 2-pixel border from invocation stats.

#### Enforce “integers must be flat-interpolated”

**Rationale:** Integer inter-stage variables require `@interpolate(flat)`; verify correctness and catch violations. \[ref:1]
**Deliverables:**

* Positive WGSL `int_flat_ok.wgsl` using `@location(1) @interpolate(flat) myInt: u32`. \[ref:1]
* Negative WGSL `int_flat_error.wgsl` with an integer inter-stage var lacking `flat`, plus a compile test that expects failure; report `int_flat_test.log`. \[ref:1]
  **Acceptance Criteria:**
* `int_flat_ok.wgsl` compiles and renders; per-triangle value in FS equals the chosen vertex’s value. \[ref:1]
* Compiling `int_flat_error.wgsl` produces a validation/compile error; test passes only if an error is detected. \[ref:1]
  **Dependencies:** WebGPU runtime (core), WGSL; CI harness to assert compile failures.
  **Risks/Mitigations:** Error messages may vary → assert on failure presence (non-success), not exact text.