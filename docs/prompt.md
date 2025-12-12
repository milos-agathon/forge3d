You are acting as a **principal real-time rendering engineer** (Rust + WGSL + WebGPU/wgpu + Python). You are working in the `forge3d` repo and MUST implement the next milestone while preserving the **frozen production default**.

## CRITICAL RULES

* You must fully read **AGENTS.md** to get familiar with my codebase rules.
* **Do not regress existing passing tests.**
* After every edit you make: **run the relevant command(s)** (build/tests/render/validator). If output doesn’t meet expectations: keep iterating (edit → run → measure) until it does. **Do not stop early.**
* **Do NOT modify thresholds** in `GORE_STRICT_PROFILE` or any validation ranges to make tests pass.
* Preserve **P5.0 behavior and output characteristics** as a **selectable preset**.
* The **P5.0 CLI command** must still render and the **P5.0 metrics must remain ≥ current** (no regressions).
* Any new feature must be:

  * **Off by default** (unless explicitly part of P6 preset).
  * **Backwards compatible** with existing CLI/API/config.
* Avoid hand-waving: every change must map to a concrete file/function/struct/uniform/shader code path.
* Performance: do not add **>10% GPU frame cost** at 1920×1080 relative to P5.0 baseline (measure and report).

# INPUT ARTIFACTS (AUTHORITATIVE)

1. `docs/gore_p5_preset.md` is the source of truth.

* P5.0 remains production default until P6 passes.
* Do not relax `GORE_STRICT_PROFILE`. Gradient targets are “truth from Gore”.
* Ignore water in all future requirements and validations (water detection/color is non-goal).

2. **Style-match harness**: `style_match_eval_v2.py` (provided) is the authoritative comparator for ROIs and exclusions.

* ROIs are defined in canonical 1920×1080 and scaled to working res.
* Top-right exclusion must be excluded from **all** region metrics.
* Water is a non-goal; do not try to “fix” water appearance. (Masking in validator may still be required per below.)

# TARGET MILESTONE

## P6.1 – Color Space Correctness + Output Encoding (Engine + Shader)

**Goal:** Fix two known correctness bugs that currently block matching:

1. **Colormap sampling uses linear format but receives sRGB bytes** (wrong luminance/chroma + wrong interpolation space).
2. **Final output uses pow-gamma** in the main path instead of exact **sRGB EOTF**.

### HARD ACCEPTANCE (MUST ALL PASS)

A) **P5.0 preset must not regress**:

* All existing P1–P5 tests remain green.
* P5.0 renders successfully with the historical look.
* P5.0 validator metrics must remain **≥ current** (no regressions).

B) **P6 preset must enable corrected behavior**:

* P6 uses **sRGB-correct colormap sampling**.
* P6 uses **exact `linear_to_srgb()`** at final output (NOT pow-gamma), while the render target remains `Rgba8Unorm` (linear).

C) **No test relaxation** anywhere.

D) Water is excluded from **metric computations**:

* If validator includes water pixels in gradient/color/luma, implement masking there.
* Do not “fix” water rendering.

E) Performance:

* P6 corrected path must not cost **>10%** GPU frame time vs P5 at 1920×1080.

# REQUIRED APPROACH (YOU MUST IMPLEMENT THIS)

## 1) Colormap color space fix (P6 only, P5 preserved)

**Problem statement (known):**

* Colormap texture currently created as `wgpu::TextureFormat::Rgba8Unorm` (linear).
* Hex colors are parsed as raw sRGB bytes and uploaded without sRGB→linear conversion.

**Required implementation:**

* Add a **new, explicit toggle** in terrain params/config that selects colormap sampling color space:

  * `colormap_srgb: bool` (or equivalent name)
  * Default **false** so P5 remains unchanged.
  * P6 preset sets it to **true**.

**When `colormap_srgb=true`:**

* Create the colormap texture as `wgpu::TextureFormat::Rgba8UnormSrgb`.
* Keep hex parsing as raw sRGB bytes (correct for sRGB texture upload).
* Ensure shader sampling yields **linear** colors automatically (wgpu sRGB sampling semantics).

**When `colormap_srgb=false`:**

* Preserve legacy behavior exactly (linear format with raw bytes).

## 2) Final output encoding fix: keep `Rgba8Unorm` target, use exact `linear_to_srgb()` (P6 only, P5 preserved)

**Problem statement (known):**

* Main shading path currently outputs using pow-gamma (`gamma_correct()`), not exact sRGB.

**Required implementation:**

* Add a **new explicit toggle** in terrain params/config:

  * `output_srgb_eotf: bool` (or similar)
  * Default **false** (P5 legacy).
  * P6 preset sets **true**.

**When `output_srgb_eotf=true`:**

* Replace the final `gamma_correct(final_color, gamma)` step with exact:

  * `linear_to_srgb(clamp(final_color, 0..∞))` then clamp to 0..1 for output
* Keep render target: `wgpu::TextureFormat::Rgba8Unorm` (LINEAR).
* Do not apply both `linear_to_srgb()` and `gamma_correct()`; it must be **one** encode step.

**When `output_srgb_eotf=false`:**

* Preserve P5 legacy behavior (pow-gamma).

**Note:** Keep `--gamma` functional for legacy mode, but in P6 encode mode it should either:

* be ignored with a warning, or
* apply only as a post artistic grade (document clearly).
  Prefer simplest: **ignore gamma when `output_srgb_eotf=true`**, and log a one-line warning.

## 3) Plumbing (Rust + Python) and presets

* Add config fields in `python/forge3d/terrain_params.py` for:

  * `colormap_srgb`
  * `output_srgb_eotf`
* Add CLI flags in `examples/terrain_demo.py`:

  * `--colormap-srgb` (store_true; default false)
  * `--output-srgb-eotf` (store_true; default false)
* Propagate through Rust render params → uniform / pipeline creation:

  * `colormap_srgb` affects colormap texture format (creation path).
  * `output_srgb_eotf` affects shader output encoding path (uniform boolean or specialization constant).
* Must be backwards compatible.

## 4) Validation + reporting

* Run:

  * P5 preset render + validator
  * P6 preset render + validator
  * Style match harness (v2) against reference for both outputs (for tracking; not replacing strict validator)
* Do not modify thresholds.

# SPECIFIC FILE-LEVEL DELIVERABLES (MANDATORY)

You must produce all of the following:

## 1) Code changes

* `python/forge3d/terrain_params.py`

  * Add `colormap_srgb: bool`
  * Add `output_srgb_eotf: bool`
  * Ensure serialization/deserialization and defaults (false) are stable.

* `examples/terrain_demo.py`

  * Add CLI flags:

    * `--colormap-srgb`
    * `--output-srgb-eotf`
  * Wire end-to-end.

* `src/terrain/mod.rs` (or exact module where the colormap texture is created)

  * Implement dual texture format creation:

    * `Rgba8Unorm` for legacy (P5)
    * `Rgba8UnormSrgb` for corrected (P6)

* `src/terrain/renderer.rs`

  * Keep terrain color target format as:

    * `wgpu::TextureFormat::Rgba8Unorm`
  * Add params propagation for the two toggles.

* `src/shaders/terrain_pbr_pom.wgsl`

  * Ensure `linear_to_srgb()` exists and is used in the **main output path only when enabled**.
  * Add a uniform bool (or packed int) to select encoding mode:

    * legacy pow-gamma vs exact sRGB EOTF.
  * Keep debug paths intact; do not break water debug modes.

* `tests/validate_terrain_p4.py` (or strict validator used by `GORE_STRICT_PROFILE`)

  * If water pixels contaminate metrics: implement masking in metric computation.
  * Do not relax thresholds.

## 2) Presets

* `presets/p5_gore_shader_only.json`

  * Must set:

    * `colormap_srgb=false`
    * `output_srgb_eotf=false`
  * Must preserve historical output characteristics.

* `presets/p6_gore_detail_normals.json` (or your current P6 preset file)

  * Must set:

    * `colormap_srgb=true`
    * `output_srgb_eotf=true`
  * Keep other P6 params unchanged unless required by correctness.

## 3) Reports / artifacts

* `reports/p5/p5_terrain_csm.png` (P5 preset render)
* `reports/p6/p6_terrain_csm.png` (P6 preset render)
* `reports/p6/p6_meta.json`

  * Must include:

    * git commit hash (if available)
    * exact CLI commands executed
    * the two toggles values
    * measured timings (frame/render time)
    * validator metrics output
    * style-match v2 metrics JSON path(s) and key ROI summary numbers (SSIM/ΔE/edge ratios)

# VALIDATION REQUIREMENTS (MANDATORY)

You must ensure these commands work (copy-paste):

```bash
python -m tests.validate_terrain_p4 --profile GORE_STRICT_PROFILE --render p5
python -m tests.validate_terrain_p4 --profile GORE_STRICT_PROFILE --render p6
```

And additionally run style-match tracking:

```bash
python style_match_eval_v2.py --ref assets/Gore_Range_Albers_1m.png --cand reports/p5/p5_terrain_csm.png --outdir reports/p5/style_match
python style_match_eval_v2.py --ref assets/Gore_Range_Albers_1m.png --cand reports/p6/p6_terrain_csm.png --outdir reports/p6/style_match
```

* `--render p5` must use P5 preset and PASS all P5 metrics (no regressions).
* `--render p6` must pass the strict profile (and water masked out if necessary).
* Water must be excluded from metric computations for both modes.

# IMPLEMENTATION RULES

* Determinism: fixed seeds where relevant.
* No silent fallbacks: if a required asset/preset is missing, fail with a clear message.
* Do not change default behavior: **new toggles default false**.
* Keep the performance budget.

# OUTPUT FORMAT (WHAT YOU MUST RETURN)

Return a single response with these sections in order:

1. **Plan (Checklist)** — concrete steps (no vague items).
2. **Files Changed** — exact paths + what changed (brief but specific).
3. **Commands to Run** — copy-pastable commands for:

   * rendering P5 + P6
   * running validation
   * running style_match_eval_v2 comparisons
4. **Acceptance Evidence** — table with:

   * P5 validator metrics (key ones from doc)
   * P6 validator metrics (key ones)
   * GPU timing comparison (P5 vs P6)
   * PASS/FAIL per acceptance item A–E
5. **Notes / Risks** — only real risks, and how you mitigated them.

If anything fails, you must:

* explain the failure precisely,
* propose the smallest corrective diff,
* and keep P5.0 intact.

---
