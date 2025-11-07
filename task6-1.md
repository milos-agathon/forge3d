# Surgical Change Spec (Rust + WGSL + Python glue)

## 1) WGSL changes (single file: `src/shaders/brdf_tile.wgsl`)

### Add to uniform struct (keep std140 alignment; pad to 16B multiples)

```wgsl
struct BrdfParams {
  // …existing fields…
  light_dir: vec3<f32>;   // new; normalized by CPU; default = [1/√3, 1/√3, √(1/3*2)]
  _pad0: f32;             // alignment
  debug_kind: u32;        // 0=Full, 1=Donly, 2=Gonly, 3=Fonly
  _pad1: vec3<u32>;       // alignment to 16 bytes
};
@group(0) @binding(0) var<uniform> params: BrdfParams;
```

### Use the uniform, not a hardcoded light

```wgsl
let L = normalize(params.light_dir);
```

### Route the term selection (no color grading, linear space only)

```wgsl
// compute D, G, F, spec = (D*G*F)/(4*NoV*NoL), diff, etc. as today
var rgb: vec3<f32>;
switch params.debug_kind {
  case 1u: { rgb = vec3<f32>(D); }                      // “lobe” preview, grayscale
  case 2u: { rgb = vec3<f32>(G); }                      // grayscale
  case 3u: { rgb = F; }                                 // show Fresnel RGB
  default: { rgb = final_color_linear; }                // existing path (Full)
}
```

**Back-compat guard:** set default `light_dir` and `debug_kind=0` in Rust; the Full path must remain numerically identical.

---

## 2) Rust offscreen renderer (files under `src/` where `render_brdf_tile` lives)

### API extension (no breaking changes)

* Keep existing `render_brdf_tile(…)` untouched.
* Add an overload (or builder) with optional params:

```rust
pub struct BrdfTileOverrides {
    pub light_dir: Option<[f32; 3]>,   // None => default baked vector
    pub debug_kind: Option<u32>,       // None => 0 (Full)
}

pub fn render_brdf_tile_with_overrides(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &mut TextureTarget,
    base_params: &BrdfParamsCpu,   // existing CPU-side params
    overrides: &BrdfTileOverrides
) -> anyhow::Result<()>;
```

### Uniform packing

* Extend CPU-side `BrdfParamsCpu` to include `light_dir: [f32;3]`, `debug_kind: u32`.
* **Defaults:**

  * `light_dir_default = [0.40824828, 0.40824828, 0.81649655]` (unit, 1/√6 * {√2, √2, 2})
  * `debug_kind_default = 0`
* Normalize `light_dir` if provided; if `len < 1e-8`, fall back to default.
* Write into the uniform buffer with correct 16-byte alignment (pad fields as in WGSL).
* **Do not** change pipeline state or sampler state; only the uniform contents.

---

## 3) Python glue (minimal)

* In `python/forge3d/renderer.py` (or your wrapper), add:

  ```python
  def render_brdf_tile_full(..., light_dir=None):
      # calls the new Rust function; light_dir=None keeps default
  def render_brdf_tile_debug(..., light_dir=None, debug_kind=0):
      # same, passes debug_kind in {0,1,2,3}
  ```
* Update **M6** only:

  * `examples/m6_generate.py` uses `render_brdf_tile_debug(..., debug_kind=1|2|3)` to capture D/G/F images, and `debug_kind=0` for Full.
  * For multi-case runs, pass the case’s `light_dir`.

**Do not** touch M1–M5 behavior; their calls remain the old API (or call the new one with `None, 0` to be safe).

---

## 4) Tests (extend, don’t break)

* `tests/test_m6_validation.py`:

  * Skip if GPU unavailable (existing behavior).
  * Assert we can:

    1. Render Full with defaults (hash equals current baseline).
    2. Render Full with a **different light_dir** and get a **different** hash (sanity).
    3. Render D/G/F (debug_kind 1/2/3) successfully (non-empty, finite stats).
  * Keep RMS/P99.9 parity thresholds unchanged.

---

## 5) Acceptance and Back-compat

* **Bit-identical** output for all existing images when:

  * `light_dir` not provided **and**
  * `debug_kind == 0`.
* No pipeline layout or bind-group index changes visible to callers (you can add fields to the same uniform block).
* `cargo test` + `pytest` green; `python examples/m1..m5_generate.py` produce identical PNGs vs main; `m6_generate.py` now supports multi-case + D/G/F GPU paths.

---

> **ROLE**: Senior graphics/engine programmer.
> **Repo**: `forge3d`.
> **Goal**: Implement M6.1 unblock with **minimal edits** that keep all prior outputs bit-identical by default.

**Implement exactly this:**

1. **WGSL** (`src/shaders/brdf_tile.wgsl`):

   * Add `light_dir: vec3<f32>` + padding, and `debug_kind: u32` + padding to `BrdfParams`.
   * Replace hardcoded light with `normalize(params.light_dir)`.
   * Route `debug_kind`:

     * `0`: Full (existing path, no math changes).
     * `1`: D-only grayscale `vec3(D)`.
     * `2`: G-only grayscale `vec3(G)`.
     * `3`: F-only RGB `F`.
   * Keep all math/precision/clamps exactly the same for Full.

2. **Rust** (offscreen renderer module where `render_brdf_tile` is defined):

   * Add `BrdfTileOverrides { light_dir: Option<[f32;3]>, debug_kind: Option<u32> }`.
   * Add `render_brdf_tile_with_overrides(...)` (see spec above), writing the new fields into the existing uniform buffer with correct 16-byte alignment.
   * Defaults: `light_dir=[0.40824828, 0.40824828, 0.81649655]`, `debug_kind=0`.
   * Leave the old `render_brdf_tile(...)` intact and have it call the new function with `None`.

3. **Python glue**:

   * Expose `render_brdf_tile_full(..., light_dir=None)` and `render_brdf_tile_debug(..., light_dir=None, debug_kind=0)` that call into the new Rust function.
   * Modify **only** `examples/m6_generate.py` to use these for multi-case validation and for D/G/F captures.

4. **Tests** (`tests/test_m6_validation.py`):

   * Keep existing parity checks.
   * Add:

     * Default Full render hash equals baseline.
     * Changing `light_dir` changes the hash.
     * D/G/F renders succeed (nonzero, finite stats).

**Hard requirements:**

* When `light_dir=None` and `debug_kind=0`, **all prior PNGs remain 100% identical** (CI must not churn).
* No new dependencies. No pipeline layout churn beyond uniform struct growth.
* Normalize any user-provided `light_dir` (fallback to default if degenerate).
* Be explicit with WGSL std140 alignment padding.

**Deliverables:**

* Minimal commits:

  * `wgsl`: uniform struct + debug switch + light_dir usage.
  * `rust`: new overrides struct + wrapper + uniform packing + defaults.
  * `python`: thin wrappers + M6 script changes.
  * `tests`: added assertions.
* Message:
  `M6: parameterized light + debug term selector (D/G/F); default bit-identical; enable multi-case GPU validation`

**Run locally:**

```bash
cargo test
pytest -q
python examples/m1_generate.py --outdir reports   # unchanged images
python examples/m6_generate.py --outdir reports   # now supports multi-case + D/G/F
```
