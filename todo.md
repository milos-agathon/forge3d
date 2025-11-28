## 1. High‑level behavior

- **Goal:** Add a GPU height curve that:
  - Only affects **macro displacement** and **normals**.
  - Leaves **overlay scalar, clamps, and any “height in meters” logic** unchanged.
  - Keeps existing frames **bitwise‑identical when `height_curve_strength == 0`** (within FP noise).

- **Definitions in shader:**
  - `h_raw`: raw heightmap sample (meters) from `height_tex`.
  - `h_clamped`: clamp(h_raw, u_shading.clamp0.x, u_shading.clamp0.y (already used for occlusion/overlay).
  - `t_geom`: normalized height for geometry, using clamp range:
    - `t_geom = clamp((h_raw - h_min) / (h_max - h_min), 0..1)` with `h_min = clamp0.x`, `h_max = clamp0.y`.
  - `t_curved = curve01(t_geom)`; final curved height (meters):
    - `h_disp = h_min + t_curved * (h_max - h_min)`.

- **Usage:**
  - **Vertex displacement:** use `h_disp` (not `h_raw`) in `vs_main` when computing `world_z`.
  - **Normals:** `calculate_normal` must sample **neighbors using `h_disp`** before Sobel.
  - **IBL:** stays as is, but now sees normals from curved surface (via `blended_normal`).
  - **Overlays & colormap LUT:** continue to use `h_clamped` / current `height_norm` pipeline (no curve).

---

## 2. New Python API python/forge3d/terrain_params.py

Extend TerrainRenderParams dataclass (config surface) with **defaulted** fields, appended at the end:

- **Fields:**
  - `height_curve_mode: str = "linear"`
  - `height_curve_strength: float = 0.0`
  - `height_curve_power: float = 1.0`
  - (Milestone 2) `height_curve_lut: np.ndarray | None = None`
    - 1D float32 array of length 256, values in `[0, 1]`.

- **Validation in __post_init__:**

  - `height_curve_mode in {"linear", "pow", "smoothstep", "lut"}`.
  - `0.0 <= height_curve_strength <= 1.0`.
  - `height_curve_power > 0.0` (e.g. `>= 0.1`).
  - If `height_curve_mode == "lut"`:
    - `height_curve_lut` must be a 1D float32 array of length 256 with all finite values in `[0, 1]`.
  - If `height_curve_mode != "lut"`:
    - `height_curve_lut` is allowed but ignored.

- **Backwards compatibility:**  
  All existing call sites remain valid (new fields have defaults).

---

## 3. New Rust fields (src/terrain_render_params.rs)

Extend native TerrainRenderParams struct:

```rust
pub struct TerrainRenderParams {
    // existing fields...
    pub height_curve_mode: String,
    pub height_curve_strength: f32,
    pub height_curve_power: f32,
    // no need to store LUT here; it's GPU-side state
    // ...
}
```

- **Parsing in TerrainRenderParams::new:**

  - Extract from the Python object:

    ```rust
    let height_curve_mode: String = params.getattr("height_curve_mode")?.extract()?;
    let height_curve_strength = to_finite_f32(
        params.getattr("height_curve_strength")?.as_gil_ref(),
        "height_curve_strength",
    )?;
    let height_curve_power = to_finite_f32(
        params.getattr("height_curve_power")?.as_gil_ref(),
        "height_curve_power",
    )?;
    ```

  - Validate:

    - Mode in `{"linear", "pow", "smoothstep", "lut"}` (same messages as Python).
    - Clamp strength to `[0.0, 1.0]` in Rust (`max(0.0).min(1.0)`).
    - Enforce `height_curve_power > 0.0` (error if not).

  - Store in struct; add getters if needed.

- **Decoding:**  
  No changes to DecodedTerrainSettings are strictly required; build_shading_uniforms already receives `params: &TerrainRenderParams` and can read curve fields directly.

---

## 4. Shading uniforms packing (terrain_renderer.rs::build_shading_uniforms)

Extend `TerrainShadingUniforms` in WGSL and the CPU packing.

### 4.1 WGSL struct change (terrain_pbr_pom.wgsl)

Append a new vec4 at the end:

```wgsl
struct TerrainShadingUniforms {
    triplanar_params : vec4<f32>,
    pom_steps        : vec4<f32>,
    layer_heights    : vec4<f32>,
    layer_roughness  : vec4<f32>,
    layer_metallic   : vec4<f32>,
    layer_control    : vec4<f32>,
    light_params     : vec4<f32>,
    clamp0           : vec4<f32>,
    clamp1           : vec4<f32>,
    clamp2           : vec4<f32>,
    height_curve     : vec4<f32>, // x=mode, y=strength, z=power, w=reserved
};
```

`@group(0) @binding(5) var<uniform> u_shading : TerrainShadingUniforms;` remains.

### 4.2 CPU packing

In build_shading_uniforms:

- At the end (after the existing exposure/gamma/colormap_strength block), append:

```rust
let mode_f = match params.height_curve_mode.as_str() {
    "linear" => 0.0,
    "pow" => 1.0,
    "smoothstep" => 2.0,
    "lut" => 3.0,
    _ => 0.0,
};
let strength = params.height_curve_strength.clamp(0.0, 1.0);
let power = height_curve_power_max(params.height_curve_power); // helper: max(eps, value)

uniforms.extend_from_slice(&[mode_f, strength, power, 0.0]);
```

- Optionally update `Vec::with_capacity` from 44 to 48.

**Invariant:** With `height_curve_strength == 0.0`, shader receives mode/power but must evaluate to identity.

---

## 5. WGSL logic: curve and sampling terrain_pbr_pom.wgsl

### 5.1 Helper functions

Add below `sample_height`:

```wgsl
fn get_height_geom_t(h_raw: f32) -> f32 {
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let range = max(h_max - h_min, 1e-6);
    return clamp((h_raw - h_min) / range, 0.0, 1.0);
}

fn apply_height_curve01(t: f32) -> f32 {
    let mode = u32(u_shading.height_curve.x + 0.5);
    let strength = clamp(u_shading.height_curve.y, 0.0, 1.0);
    if (strength <= 0.0) {
        return t;
    }

    var curved = t;
    if (mode == 1u) { // pow
        let p = max(u_shading.height_curve.z, 0.01);
        curved = pow(t, p);
    } else if (mode == 2u) { // smoothstep
        curved = t * t * (3.0 - 2.0 * t);
    } else if (mode == 3u) { // lut (Milestone 2)
        curved = height_curve_lut_sample(t); // see 6.2
    }

    return mix(t, curved, strength);
}

fn sample_height_geom(uv: vec2<f32>) -> f32 {
    let uv_clamped = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_raw = textureSample(height_tex, height_samp, uv_clamped).r;
    let t = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    return h_min + apply_height_curve01(t) * (h_max - h_min);
}
```

### 5.2 Vertex displacement

In `vs_main`:

- Replace:

```wgsl
let height = textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
let world_z = height * h_exag;
```

- With a level-aware curved sample:

```wgsl
let h_raw = textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
let t = get_height_geom_t(h_raw);
let h_min = u_shading.clamp0.x;
let h_max = u_shading.clamp0.y;
let h_disp = h_min + apply_height_curve01(t) * (h_max - h_min);
let world_z = h_disp * h_exag;
```

### 5.3 Normal reconstruction

In `calculate_normal`, replace all eight `sample_height(...)` calls with `sample_height_geom(...)`:

```wgsl
let tl = sample_height_geom(uv - offset_x - offset_y);
...
let br = sample_height_geom(uv + offset_x + offset_y);
```

`vertical_scale` stays as is (`u_terrain.spacing_h_exag.z * 0.5`).

### 5.4 Keep overlay & POM raw

- **Do not change:**
  - `height_sample = sample_height(parallax_uv);`
  - `height_clamped`, `height_norm`, `lut_u`, `overlay_rgb`, overlay blends.
  - POM `parallax_occlusion_mapping`’s internal `sample_height` usage (intentionally stays in raw height domain, not curved).

---

## 6. LUT mode (Milestone 2)

### 6.1 GPU binding (Rust)

- Extend terrain bind group layout with **one new texture binding**:

  - `@group(0) @binding(9): texture_2d<f32>` for a 256×1 R32Float LUT (1D emulated as 2D).

- In TerrainRenderer::new_internal:
  - Add `wgpu::BindGroupLayoutEntry` for binding 9 (texture, non-filterable or linear as desired).
  - Create a default 256×1 identity LUT on GPU for cases where mode==lut but user did not supply one.

- Add per-render call:
  - A method in TerrainRenderer to upload/replace height LUT based on Python config (Milestone 2). For now, spec only; implementation details left to 5.1.

### 6.2 WGSL LUT sample

Add:

```wgsl
@group(0) @binding(9)
var height_curve_lut_tex : texture_2d<f32>;
@group(0) @binding(10)
var height_curve_lut_samp : sampler;
```

Add function:

```wgsl
fn height_curve_lut_sample(t: f32) -> f32 {
    let u = clamp(t, 0.0, 1.0);
    let uv = vec2<f32>(u, 0.5);
    return textureSample(height_curve_lut_tex, height_curve_lut_samp, uv).r;
}
```

---

## 7. Tests and verification

### 7.1 Python/Rust bridge

- Extend tests/test_terrain_render_params_native.py:

  - Create config with non-default curve fields (e.g. `"pow"`, `0.5`, `2.0`) and assert:
    - `native.height_curve_mode == "pow"`.
    - `native.height_curve_strength == pytest.approx(0.5)`.
    - `native.height_curve_power == pytest.approx(2.0)`.

- Add negative tests for invalid modes and out-of-range strengths/powers raising `ValueError`.

### 7.2 Rendering invariants

- New test in tests/test_terrain_renderer.py

  - Render a simple heightmap twice with same seed:
    - Once with `height_curve_strength = 0.0`.
    - Once with `height_curve_mode = "pow"`, `height_curve_strength = 0.0`.
  - Assert frames are identical (or within 1 u8 step per channel).

- Visual regression (manual / non-asserted in CI):
  - Use a simple radial hill DEM, render with increasing `height_curve_strength` and `pow` vs `smoothstep`, verify:
    - Overlay colormap “bands” stay anchored at the same input heights.
    - IBL reflections track the curved silhouette (spec highlights follow curved profile).

---

## 8. Milestones for ChatGPT 5.1

- **M1 – Core analytic curves (no LUT):**
  - Implement Python/Rust fields + validation.
  - Extend `TerrainShadingUniforms` + packing.
  - Implement `get_height_geom_t`, `apply_height_curve01`, `sample_height_geom`.
  - Wire into `vs_main` and `calculate_normal`.
  - Ensure overlay & POM remain raw/unaffected.
  - Add bridge + BC tests.

- **M2 – LUT mode plumbing:**
  - Add LUT bindings in Rust + WGSL.
  - Add `height_curve_lut` handling in Python and GPU upload in Rust.
  - Implement `height_curve_lut_sample` and wire into `apply_height_curve01` for mode `lut`.
  - Add lightweight tests that LUT mode runs and affects geometry.

- **M3 – Visual/IBL checks (optional):**
  - Add a debug example or test harness to confirm IBL anchoring and overlay semantics.

---


Assess the current level of implementation of Phase 5 from todo-6.md. If the requirements are met, do nothing. If the requirements are not met, think very hard to turn these into an extremely surgically precise and strict prompt for ChatGPT 5.1 (high reasoning) to accomplish the missing requirements

you must read fully @AGENTS.md to get familiar with my codebase. Next, you must carefully read @todo-5.md as a whole. Then you must fully implement P5.8. Test after every change. Do not stop until you meet all the requirements for P5.8

you must read fully @AGENTS.md  to get familiar with my codebase. Next, you must carefully read @p6.md as a whole. Then you must design an extremely surgically precise and accurate and specific set of requirements for ChatGPT 5.1. (high reasoning)  with coherent milestones and clear deliverables

python -m pytest tests/test_terrain_renderer.py -q is failing again. Consult logs and fix the failing tests in one go

python examples/terrain_demo.py --dem assets/Gore_Range_Albers_1m.tif --hdr assets/snow_field_4k.hdr --size 3200 1800 --msaa 8 --z-scale 5.0 --albedo-mode material --colormap '#e4bc6b, #f99157, #ec5f67, #ab7967' --colormap-strength 0.0 --colormap-interpolate --colormap-size 1024 --cam-phi 135 --cam-theta 20 --cam-radius 250 --sun-azimuth 135 --sun-intensity 2.0 --water-detect --gi ibl --ibl-intensity 0.5 --output examples/out/terrain_material.png --overwrite 

# POM ON
python examples/terrain_demo.py --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr --size 1600 900 --render-scale 1.0 \
  --cam-theta 20 --cam-radius 250 --cam-phi 135 \
  --msaa 8 --z-scale 2.0 --albedo-mode colormap --colormap-strength 1.0 \
  --gi ibl --ibl-intensity 0.1 --sun-intensity 1.0 \
  --colormap '#ab7967, #ec5f67, #f99157, #e4bc6b' --colormap-interpolate --colormap-size 1024 \
  --output examples/out/terrain_pom_on.png --overwrite

# POM OFF
python examples/terrain_demo.py --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr --size 1600 900 --render-scale 1.0 \
  --cam-theta 20 --cam-radius 250 --cam-phi 135 \
  --msaa 8 --z-scale 2.0 --albedo-mode colormap --colormap-strength 1.0 \
  --gi "" --pom-disabled --ibl-intensity 0.1 --sun-intensity 1.0 \
  --colormap '#ab7967, #ec5f67, #f99157, #e4bc6b' --colormap-interpolate --colormap-size 1024 \
  --output examples/out/terrain_pom_off.png --overwrite