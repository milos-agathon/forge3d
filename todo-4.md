# P4 — Image-Based Lighting (IBL) Pipeline: **Strict Engineering Spec (v2)**

> **Goal:** Ship high-quality environment lighting (diffuse irradiance + GGX specular prefilter + BRDF LUT) with **offline/first-frame precompute**, **deterministic caching**, **unified WGSL bindings**, **Python/CLI wiring**, and **CI-verifiable acceptance tests**. The spec below removes ambiguity by fixing names, sizes, formats, bindings, tolerances, file paths, and exit criteria.

---

## Global constraints (apply to all milestones)

* **Texture formats**

  * Cubemaps (env/spec/irr): `Rgba16Float`
  * BRDF LUT: `Rgba16Float` (2D)
* **Default sizes** (overridable by quality preset; must not exceed memory budget)

  * Base environment cube: **512²**; Irradiance: **64²**; BRDF LUT: **256×256**; Specular mip count: **log2(base)+1**
* **WGSL binding layout (must match across all shaders)**

  * `@group(2) @binding(0) envSpecular : texture_cube<f32>`
  * `@group(2) @binding(1) envIrradiance : texture_cube<f32>`
  * `@group(2) @binding(2) envSampler  : sampler` (filtering + clamp-to-edge)
  * `@group(2) @binding(3) brdfLUT     : texture_2d<f32>`
* **File locations & names**

  * Shaders:

    * `src/shaders/ibl_equirect.wgsl` (`cs_equirect_to_cubemap`)
    * `src/shaders/ibl_prefilter.wgsl` (`cs_irradiance_convolve`, `cs_specular_prefilter`)
    * `src/shaders/ibl_brdf.wgsl` (`cs_brdf_lut`)
    * Shared eval include: `src/shaders/lighting_ibl.wgsl`
  * Rust:

    * Core: `src/core/ibl.rs`
    * Cache: `src/lighting/ibl_cache.rs`
    * Wrapper: `src/ibl_wrapper.rs`
    * Pipelines: `src/pipeline/pbr.rs`, `src/terrain_renderer.rs`
  * Python examples/tests: `examples/ibl_gallery.py`, `examples/terrain_demo.py`, `tests/test_ibl.py`, `tests/test_p4_ibl_cache.py`
* **Cache file**: `*.iblcache` (see Milestone 1.5) with **sha256** over source HDR + params; versioned `v: u32` to invalidate on breaking changes.
* **Determinism**: GPU compute pass order fixed; all sampling kernels use **fixed sample counts** per mip; no random seeds.
* **Numeric safety**: All WGSL math `saturate()` inputs and clamp divisions; no NaNs/inf; assert in debug.
* **Acceptance images**: Save as **opaque RGB PNG** (no alpha), with fixed filenames (listed below) for CI diffing.

---

## Milestone 1 — **Precompute kernels & cache** (compute shaders only)

### 1.1 Equirectangular → Cubemap

* **Shader**: `ibl_equirect.wgsl::cs_equirect_to_cubemap`
* **Inputs**: `texture_2d<f32> hdrEq`, `sampler hdrSamp`; **Output**: `texture_cube<f32> envBase`
* **Face orientation**: OpenGL cubemap convention (+X, −X, +Y, −Y, +Z, −Z); right-handed; y-up.
* **Workgroup**: `8×8` (required)
* **Exit criteria**

  * 6 faces written (check array layer writes), **no seams** at edges (bilinear sample with edge clamps)
  * Sanity image: `reports/p4_env_base.png` (6× strip, 16 px gutters)
  * Max absolute seam delta across face edges: **≤ 1.5e-3** (linear space)

### 1.2 Diffuse irradiance (Lambertian convolution)

* **Shader**: `ibl_prefilter.wgsl::cs_irradiance_convolve`
* **Output**: `texture_cube<f32> envIrradiance` of size **64²**
* **Sampling**: Stratified hemisphere (cos-weighted) **128 samples/texel** (fixed)
* **Exit criteria**

  * Pure-diffuse sphere brightens uniformly; average luminance vs envBase is **monotonic** with env intensity
  * Image: `reports/p4_irradiance_cube.png` (strip)
  * **No pixel > 1.0** in linear for unit-intensity HDR

### 1.3 GGX specular prefilter (mip chain)

* **Shader**: `ibl_prefilter.wgsl::cs_specular_prefilter`
* **Mip mapping**: roughness → mip: `mip = roughness² * (mipCount-1)`
* **Samples/texel**: mip0=1024, mip1=512, mip2=256, … min 64
* **Exit criteria**

  * For a chrome sphere (F0=1, roughness sweep), highlight size increases with mip; **no fireflies**
  * Image: `reports/p4_specular_cube_mips.png` (mips stacked)
  * `NaN/Inf` count = **0**

### 1.4 BRDF 2D LUT (split-sum)

* **Shader**: `ibl_brdf.wgsl::cs_brdf_lut`
* **Domain**: `x=NdotV∈[0,1]`, `y=roughness∈[0,1]`
* **Exit criteria**

  * `LUT.xy ∈ [0,1]`, edges finite; border clamps valid
  * Image: `reports/p4_brdf_lut.png`
  * Sample probes at `(0,0),(0.5,0.5),(1,1)` logged

### 1.5 On-disk cache

* **Rust**: `ibl_cache.rs`
* **Metadata** (JSON header then binary blobs):
  `{ "v":1, "sha256":"…", "hdr_dims":[w,h], "base":512, "irr":64, "brdf":256, "mips":10, "format":"Rgba16F" }`
* **Exit criteria**

  * First run computes, second run loads with a measured **≥3× speedup** (timers logged)
  * Corrupted cache → **graceful invalidate** (checksum mismatch)

---

## Milestone 2 — **Bindings & runtime integration**

### 2.1 Unified WGSL bindings

* Ensure **all** shading paths include `lighting_ibl.wgsl` and declare group(2) bindings **exactly** as in *Global constraints*.
* **Exit criteria**: `cargo build --all-features` succeeds; reflection logs show identical layouts in `pbr.wgsl` and `terrain_pbr_pom.wgsl`.

### 2.2 PBR mesh path migration

* **Files**: `src/shaders/pbr.wgsl`, `src/pipeline/pbr.rs`
* Replace any 2D env use with cubemap + LUT. Remove 2D fallback (guard behind `feature="ibl2d_debug"` if kept).
* **Exit criteria**

  * Visual parity with terrain path within **ΔE (CIELab) ≤ 2.0** on a 512² test sphere
  * No bind-group mismatches; draw does not rebind samplers redundantly (validated via renderdoc markers if available)

### 2.3 Shared IBL evaluator

* **Include**: `lighting_ibl.wgsl` must provide
  `fn eval_ibl(n: vec3<f32>, v: vec3<f32>, base_color: vec3<f32>, metallic: f32, roughness: f32, f0: vec3<f32>) -> vec3<f32>`
* **Implementation**: Lambert irradiance + GGX prefilter × BRDF LUT (split-sum). **No random sampling at runtime.**
* **Exit criteria**: Bit-exact output across terrain/mesh for same inputs on a test set (checked in tests).

---

## Milestone 3 — **Python bridge & CLI**

### 3.1 Wrapper API (PyO3)

* `IBL.from_hdr(path: &str) -> PyResult<IBL>`
* `set_base_resolution(u32)`, `set_cache_dir(&Path)`, `configure_quality(IBLQuality)`
* Lazy GPU init; CPU-only import OK (no device creation).
* **Exit criteria**: `examples/terrain_demo.py --gi ibl --hdr assets/sky_4k.hdr` runs end-to-end.

### 3.2 Example/CLI

* `examples/ibl_gallery.py` flags: `--ibl-res`, `--ibl-cache`, `--rotate <deg/s>`, `--frames <n>`
* **Exit criteria**: Flags propagated; cache reused between runs (log “cache hit”).

### 3.3 Viewer commands

* In `src/viewer/mod.rs`: `ibl on|off`, `ibl intensity <f32>`, `ibl rotate <deg>`, `ibl load <path>`, `ibl cache <dir>`, `ibl res <u32>`
* **Exit criteria**: Commands work at the same prompt; snapshot pipeline unaffected.

---

## Milestone 4 — **Tests & acceptance**

### 4.1 Unit tests (`tests/test_ibl.py`)

* Cache round-trip: cold→hot timing **improves ≥3×** or explicit cache-hit metric
* Quality mapping: `IBLQuality::{Low,Med,High}` → exact sizes/mips
* LUT bounds: random UV samples produce `[0,1]` with **no NaNs**

### 4.2 Integration tests (`tests/test_p4_ibl_cache.py`)

* Tiny sizes (e.g., base=128, irr=32, brdf=128) to keep CI fast
* Render sphere with `ibl on/off` → images differ (SSIM **≤ 0.95**)
* **Deterministic**: two runs with same cache are byte-identical

### 4.3 Acceptance gallery

* Write PNGs to `reports/`:

  * `p4_env_base.png`, `p4_irradiance_cube.png`, `p4_specular_cube_mips.png`, `p4_brdf_lut.png`, `p4_meta.json`
* `p4_meta.json` keys (strict):
  `{"base":512,"irr":64,"brdf":256,"mips":10,"cache_used":true,"timings":{"compute_ms":...,"load_ms":...}}`
* **Exit criteria**: All images non-empty; meta parses; timing invariants hold.

---

## Milestone 5 — **Docs & examples**

* `docs/user/ibl_overview.rst` (or `docs/environment_mapping.md`) must contain:

  * GPU supples the linear HDR frames (from M4 resources + M1 dir light).
  * CPU must apply tone mapping and build the 3x5 grid.

What it proves Tone-map stage is correct, monotone, and does not introduce clipping beyond configured tolerance.

Backend expectation • GPU supplies the linear HDR frames (from M4 resources + M1 dir light). • CPU must apply tone mapping and build the 3×5 grid.

CLI

```python
python examples/m5_generate.py --hdr assets/snow_field_4k.hdr --outdir reports/m5
```

Outputs • m5_tonemap_compare.png (columns: Linear, Reinhard, ACES × roughness) • m5_meta.json

Acceptance checks • E1: Linear column equals the pre-tonemap baseline pixel-for-pixel. • E2: Clipped channel ratio for Reinhard/ACES ≤ 0.01%. • E3: Reinhard and ACES curves monotone over [0, 32].

Meta must include

{ "milestone": "M5", "backend": "gpu", // frames came from GPU; tone-map on CPU is implied "tone_curves": ["linear", "reinhard", "aces"], "exposure": 1.0, "clip_fraction": {"reinhard": 0.00005, "aces": 0.00012}, "accept": {"E1": true, "E2": true, "E3": true} }

---

## Milestone 6 — **CI, lint, warnings**

* GPU-dependent tests **skip gracefully** when adapter missing (pytest markers)
* Byte-level tests for cache use **pure CPU** paths (serialize/deserialize only)
* `cargo clippy --all-features -- -D warnings` passes
* **Exit criteria**: CI green on Linux/macOS/Windows; no new warnings.

What it proves GPU BRDF output matches a deterministic CPU reference within strict error tolerances; debug D/G/F outputs and per-case light overrides work.

Backend expectation • Both CPU and GPU are required.

CLI
```bash
python examples/m6_generate.py --outdir reports/m6
--tile 256 --samples 1024 --seed 1337 --cases default,highlight,glancing pytest -q tests/test_m6_validation.py
```

Outputs • m6_diff_heatmap.png (visual diff; optional) • m6_diff.csv (per-pixel errors; optional when large) • m6_meta.json

Acceptance checks • F1_rms: GPU vs CPU RMS error ≤ 0.002 in linear space • F2_p999: 99.9th percentile absolute error ≤ 0.01 • F3_hash: CSV/tile hashes match across repeated runs (same seed)

Meta must include

```json
{ "milestone": "M6", "backend": "gpu", "cpu_reference": {"backend": "cpu", "cpu_impl": "forge3d.cpu_ref.v1"}, "cases": [ { "name": "default", "light_dir": [0.4082, 0.4082, 0.8165], "metrics": {"rms": 0.0012, "p999": 0.0089}, "accept": {"F1_rms": true, "F2_p999": true} } ], "hashes": {"csv_sha256": "", "gpu_tile_sha256": ""}, "random_seed": 1337, "accept": {"F3_hash": true} }
```

---

## Milestone 7 — **Performance & memory budgets**

* Respect `lighting/memory_budget.rs` hard caps:

  * Max VRAM for IBL assets: **≤ 64 MiB** (default quality)
  * Degrade order: base cube ↓, then specular mips ↓, then irradiance ↓ (never below 32²), LUT last
* **Exit criteria**: On a 4 GB iGPU target profile, IBL alloc succeeds and renders; logs include chosen quality tier.

What it proves The project can be installed, documented, and reproduced; smoke GPU render confirms install.

Backend expectation • No new shading, but GPU smoke test required to prove runtime.

CLI
```bash
pip install . forge3d --help python -m pip install -r requirements-dev.txt pytest -q
```

Outputs • m7_meta.json (environment, versions, smoke test hashes) • Built wheels/sdist in dist/

Acceptance checks • G1: Wheel builds on target platforms; pip install OK. • G2: CLI works; forge3d --version prints coherent version + git SHA. • G3: Smoke render succeeds on GPU; hash recorded. • G4: Docs build (if present) with zero errors.

Meta must include

```json
{ "milestone": "M7", "backend": "gpu", "package": {"version": "0.4.0", "wheel": "forge3d-0.4.0-cp311-manylinux.whl"}, "env": {"python": "3.11.9", "platform": "win_amd64"}, "smoke": {"render_hash": "", "elapsed_ms": 412}, "accept": {"G1": true, "G2": true, "G3": true, "G4": true} }
```

---

## “Do / Don’t” checklist (must-pass reviews)

* **Do**

  * Keep all WGSL alpha/roughness math clamped and branch-free when possible
  * Use **clamp-to-edge** samplers for cubemap lookups
  * Emit **opaque RGB** PNGs (strict)
  * Log sizes, mips, timings, cache status in a single **info** line per run

* **Don’t**

  * Don’t sample the equirect HDR at runtime in PBR paths
  * Don’t vary sample counts per frame (determinism)
  * Don’t write RGBA outputs for galleries/debug

---

## Acceptance gate (PR checklist)

1. All milestone exit criteria met with artifacts in `reports/` and `p4_meta.json` present.
2. CI passes on 3 OSes; GPU-less runners skip heavy tests without failures.
3. Visual parity test mesh↔terrain **ΔE ≤ 2.0**.
4. Cache hot run ≥3× faster than cold (or explicit cache-hit metric shown).
5. Docs published and linked; bindings table present.

---

## Recommended implementation order

1. **M1** Precompute + cache → 2) **M2** Bindings/runtime → 3) **M3** Python/CLI → 4) **M4** Tests → 5) **M5** Docs → 6) **M6** CI/Lint → 7) **M7** Perf/budgets.

---

### Appendix A — Fixed function/entry names

* WGSL:

  * `cs_equirect_to_cubemap`, `cs_irradiance_convolve`, `cs_specular_prefilter`, `cs_brdf_lut`, `eval_ibl(...)`
* Rust (public):

  * `IBLRenderer::from_hdr(Path)`, `generate_irradiance_map()`, `generate_specular_map()`, `generate_brdf_lut()`, `configure_cache(...)`, `try_load_cache()`, `write_cache()`

---

### Appendix B — Error handling (examples)

* Cache load: return `Err(CacheError::HashMismatch)` → recompute; never panic
* Binding mismatch: `anyhow!("IBL bind layout mismatch: expected group(2) ...")` with expected vs found dump
* Image write: Fail if not RGB8/16 PNG; include path and reason