You are an expert Rust + wgpu + WGSL graphics engineer working inside the `forge3d` repo.
Your mission is to implement **P5.4 – GI Composition & Ordering + Quality knobs** and prove it via code, images, and metrics.

You must work in **small, auditable milestones** and keep the QA constraints intact.
Do *not* weaken, bypass, or “cheat” any tests.

---

## 0. Scope (you must implement exactly this)

We have three screen-space GI effects:

* **AO** (ambient occlusion) – multiplicative darkening.
* **SSGI** (screen-space global illumination) – diffuse bounce light.
* **SSR** (screen-space reflections) – glossy / mirror specular.

Your job is to **centralize their composition into a single compute pass** with physically sane behavior:

1. **AO multiplies diffuse only.**

   * No darkening of specular or emissive components.

2. **SSGI adds onto diffuse only**, on top of direct + IBL + AO, but:

   * Overall pixel energy must not exceed the **baseline+IBL** energy by more than **5%** (histogram cap).

3. **SSR replaces or lerps specular** using roughness/Fresnel:

   * Perfect mirrors (roughness≈0, high Fresnel) → mostly SSR.
   * Very rough surfaces (roughness→1) → mostly original specular (noisy SSR suppressed).

You must implement:

* **WGSL**

  * `src/shaders/gi/composite.wgsl` – **one compute kernel** that reads AO / SSGI / SSR results and writes the final lighting buffer.
* **Rust**

  * `src/passes/gi.rs` – orchestrates GI sub-passes, toggles, and parameter propagation to the composite pass.

---

## 1. Global constraints (never violate these)

1. **No QA cheating**

   * Do not write constants or shortcuts that force tests to pass (e.g., always scaling back to 0.95 of baseline).
   * Do not bypass or comment out existing QA checks.
   * Do not silently clamp everything so hard that effects become visually invisible.

2. **Energy & separation constraints**

   * Turning **AO** on/off may affect only the **diffuse** term (within ±1/255 per channel due to numerical noise).
   * Turning **SSGI** on/off may only affect the **diffuse** term.
   * Turning **SSR** on/off may only affect the **specular** term.
   * For all combinations, **per-pixel luminance** with all GI effects enabled must be ≤ **1.05 × baseline+IBL luminance** (5% cap), except for negligible tails (<0.1% pixels) if the QA code allows.

3. **File boundaries**

   * GI composition logic belongs in `src/shaders/gi/composite.wgsl`.
   * High-level orchestration belongs in `src/passes/gi.rs`.
   * Do not move unrelated logic into these files; keep changes minimal and well-scoped.

---

## 2. Milestone 1 – Recon & data model

**Goal:** Understand the existing GI pipeline so the composite pass uses the correct buffers and conventions.

**Tasks**

1. Inspect existing GI passes and shaders:

   * Find AO output texture(s) and their format/range.
   * Find SSGI output texture(s) and their format/range.
   * Find SSR output texture(s) and how specular is currently combined.
   * Identify the **lighting buffer** (the HDR buffer that currently accumulates diffuse + spec + IBL).

2. Identify how **G-buffer** information is laid out:

   * Where do you read:

     * Diffuse albedo.
     * Specular / F₀ / metallic.
     * Roughness.
   * How are view direction, normal, and Fresnel computed in the current shading path?

3. Document (in comments in the code) the existing lighting model split:

   * A clear conceptual separation between:

     * `L_diffuse_base`  = direct diffuse + IBL diffuse *before* AO/SSGI.
     * `L_spec_base`     = direct specular + IBL specular *before* SSR.

**Deliverables**

* Short inline comments in relevant Rust/WGSL files summarizing:

  * Which textures hold AO, SSGI, SSR.
  * What each channel means (e.g., “AO is in R, range [0,1]”).
* No functional changes yet.

---

## 3. Milestone 2 – Design the composition math (on paper / in comments)

**Goal:** Define the exact formulas for AO, SSGI, SSR composition before writing WGSL.

**Requirements**

Work out and write (as comments) something equivalent to:

1. **AO (diffuse only)**

   ```text
   L_diffuse_ao = L_diffuse_base * mix(1.0, ao, ao_weight)
   ```

   * `ao` in [0,1], where 0 = full occlusion (dark), 1 = no occlusion.
   * `ao_weight` in [0,1] is a quality knob (1.0 = full AO).

2. **SSGI (diffuse additive, energy-bounded)**

   ```text
   L_diffuse_gi = L_diffuse_ao + ssgi_weight * L_ssgi
   ```

   But with an energy budget:

   ```text
   L_budget = L_base_diffuse_spec_ibl   // baseline+IBL total luminance
   L_with_gi = L_diffuse_gi + L_spec_base

   luminance(L_with_gi) <= 1.05 * luminance(L_budget)
   ```

   Implement this as a scalar or per-channel clamp that:

   * Preserves hue as much as possible.
   * Only scales down *excess* GI contribution; does not dim everything uniformly.

3. **SSR (specular replacement/lerp)**

   ```text
   k_ssr = k_fresnel * (1.0 - roughness)^n * ssr_weight   // 0..1
   L_spec_final = mix(L_spec_base, L_spec_ssr, k_ssr)
   ```

   * `k_fresnel` from existing Fresnel term or F₀.
   * `n` (e.g., 2–4) controls how fast SSR fades with roughness.
   * `ssr_weight` is a user quality knob in [0,1].

4. Define exactly how AO, SSGI, SSR toggles are represented in uniforms:

   * `ao_enable` (bool or 0/1).
   * `ssgi_enable`.
   * `ssr_enable`.
   * Quality weights: `ao_weight`, `ssgi_weight`, `ssr_weight`.

**Deliverables**

* WGSL comments at the top of `src/shaders/gi/composite.wgsl` with the chosen formulas.
* Rust-side comments in `src/passes/gi.rs` describing the uniform struct and knobs.

---

## 4. Milestone 3 – Implement `gi/composite.wgsl`

**Goal:** Write the **single compute kernel** that performs AO/SSGI/SSR composition.

**Tasks**

1. Create `src/shaders/gi/composite.wgsl` with:

   * Bindings for:

     * AO texture.
     * SSGI texture.
     * SSR texture.
     * G-buffer texture(s) (for roughness / Fresnel inputs as needed).
     * Baseline lighting buffer (before AO/SSGI/SSR).
     * Final lighting buffer (output).
     * Uniform buffer with GI composition params (toggles + weights + energy cap).

   * A compute entry point that:

     1. Loads baseline diffuse + spec components for the pixel.
     2. Conditionally applies AO (if enabled).
     3. Conditionally adds SSGI and enforces the 5% energy cap.
     4. Conditionally applies SSR to specular using roughness/Fresnel.
     5. Writes the final RGB (and alpha if used) to the lighting buffer.

2. Ensure **precision and ranges**:

   * Use HDR formats correctly (no unnecessary clamp to [0,1] before tonemapping).
   * Handle NaNs and infs defensively (replace with 0).

3. Expose quality knobs via the uniform:

   * `ao_weight`, `ssgi_weight`, `ssr_weight`, floats [0,1].
   * `energy_cap` (e.g., `1.05`).

**Deliverables**

* Compilable `src/shaders/gi/composite.wgsl`.
* No API breakages in other shaders.

---

## 5. Milestone 4 – Integrate composition in `src/passes/gi.rs`

**Goal:** Wire AO, SSGI, SSR outputs into the new composite pass and expose toggles/weights.

**Tasks**

1. In `src/passes/gi.rs`:

   * Identify the current sequence of:

     * AO pass.
     * SSGI pass.
     * SSR pass.
     * Any prior composition or blending.

   * Replace the ad-hoc blending with a call to **one composite pass**:

     * Ensure AO/SSGI/SSR passes write into well-defined intermediate textures.
     * The composite pass reads these and the baseline lighting and overwrites the final lighting buffer.

2. Add a GI params struct:

   ```rust
   pub struct GiCompositeParams {
       pub ao_enable: bool,
       pub ssgi_enable: bool,
       pub ssr_enable: bool,
       pub ao_weight: f32,
       pub ssgi_weight: f32,
       pub ssr_weight: f32,
       pub energy_cap: f32, // expected 1.05
   }
   ```

   * Make sure these are uploaded to the composite WGSL uniform.

3. Hook the params up to any existing CLI / viewer commands you already use for GI toggling. If none exist yet, add minimal commands:

   * `:ao on|off`
   * `:ssgi on|off`
   * `:ssr on|off`
   * `:ao-weight <float>`
   * `:ssgi-weight <float>`
   * `:ssr-weight <float>`

**Deliverables**

* Updated `src/passes/gi.rs` that:

  * Builds and runs.
  * Invokes the new composite pass after AO/SSGI/SSR.
* Viewer or example commands that can toggle/adjust each effect.

---

## 6. Milestone 5 – GI stack ablation example & image

**Goal:** Generate the required **4-column ablation image**.

**Tasks**

1. Add or update a P5 example (e.g. `examples/p5_gi_stack_ablation.rs`) that:

   * Renders the P5 reference scene from a fixed camera.
   * Produces **four separate frames**:

     1. Baseline (no AO, no SSGI, no SSR).
     2. +AO only.
     3. +AO+SSGI.
     4. +AO+SSGI+SSR.

2. Programmatically stitch these four frames side-by-side into:

   * `reports/p5/p5_gi_stack_ablation.png` (4 columns, same height as other P5 images).

3. Ensure all four columns share *identical* camera, exposure, tonemapping, and post-processing; only GI toggles may differ.

**Deliverables**

* `reports/p5/p5_gi_stack_ablation.png` generated by running a single command, e.g.:

  ```bash
  cargo run --release --example p5_gi_stack_ablation
  ```

* The image clearly showing the incremental visual contribution of AO, SSGI, SSR.

---

## 7. Milestone 6 – Extend `p5_meta.json` instrumentation

**Goal:** Include composition order, weights, and timings in `p5_meta.json`.

**Tasks**

1. Locate the P5 reporting code that writes `reports/p5/p5_meta.json`.

2. Extend the JSON schema to include, at minimum:

   ```json
   "gi_composition": {
     "order": ["baseline", "ao", "ssgi", "ssr"],
     "weights": {
       "ao_weight": <float>,
       "ssgi_weight": <float>,
       "ssr_weight": <float>
     },
     "toggles": {
       "ao_enable": true,
       "ssgi_enable": true,
       "ssr_enable": true
     },
     "timings_ms": {
       "ao": <float>,
       "ssgi": <float>,
       "ssr": <float>,
       "composite": <float>
     }
   }
   ```

3. Make sure timings come from the real pass timings (GPU timer queries or CPU measurements already present), not fake constants.

**Deliverables**

* Updated `p5_meta.json` after running the P5 GI example, showing:

  * Correct `order`, `weights`, `toggles`, and `timings_ms`.

---

## 8. Milestone 7 – Verification against acceptance criteria

**Goal:** Prove that the implementation satisfies the **energy** and **component isolation** requirements.

**Tasks**

1. Implement or extend a small QA helper (Rust side) that renders:

   * Baseline (no GI).
   * Baseline+AO.
   * Baseline+AO+SSGI.
   * Baseline+AO+SSGI+SSR.

2. For each rendered image, compute per-pixel luminance and:

   * Confirm that with all GI features on:

     ```text
     luminance_all(x,y) <= 1.05 * luminance_baseline_ibl(x,y)
     ```

     for all but perhaps an extremely small number of outliers (if your histogram check permits).

3. For each effect toggle, confirm **component isolation**:

   * When toggling **AO** (with SSGI and SSR off):

     * Diffuse component changes as expected.
     * Specular component changes are within ±1/255 per channel (comparison in HDR or pre-tonemap space; document your choice).
   * Similarly for **SSGI** and **SSR**, each only affecting its intended component within ±1/255.

   Implement this as either:

   * A debug mode in Rust that compares two frames and asserts component deltas, OR
   * A lightweight test harness that dumps metrics into `p5_meta.json` (e.g., max_diff_diffuse, max_diff_specular per toggle).

4. Do **not** satisfy these constraints by blindly scaling all lighting down. The comparison must be made against the correctly defined **baseline+IBL** frame.

**Deliverables**

* Either assertions/tests in code or explicit metrics in `p5_meta.json` showing:

  * Max luminance ratio ≤ 1.05.
  * Max unintended component delta ≤ 1/255.

---

## 9. Milestone 8 – Final summary for the human reviewer

At the end, output a clear summary (in the chat) with:

1. **Files edited** and a 1–2 sentence description per file.
2. **Final formulas** actually used in the composite pass (as WGSL-style pseudocode).
3. **Key metrics** extracted from `p5_meta.json`:

   * `gi_composition.order`
   * `gi_composition.weights`
   * `gi_composition.timings_ms`
   * Any QA metrics for energy and component isolation.
4. How to reproduce everything:

   ```bash
   cargo run --release --example p5_gi_stack_ablation
   # and any additional commands you added
   ```
