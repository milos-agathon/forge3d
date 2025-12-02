Here’s a clean, execution-ready plan (with concrete artifacts) to tackle the “flake” root cause you identified: **height Sobel normals sampling at an implicit LOD while offsets assume LOD0 texels** (mip mismatch), plus making the new debug modes *provably* non-no-op.

---

## Milestone 0 — Lock a reproducible baseline (so fixes are measurable)

**Goal:** one command that always reproduces flakes.

**Deliverables**

* `reports/flake/baseline_material.png` (flakes present)
* `reports/flake/baseline_colormap.png` (no flakes)
* `reports/flake/repro_cmd.txt` (exact CLI + env vars)

**Acceptance**

* Baseline material render shows visible salt/flake artifacts at the same ROI each run.

---

## Milestone 1 — Make debug modes trustworthy (plumbing + non-uniform asserts)

Right now your attached outputs suggest **some debug modes are producing a flat clear-color image** (solid blue), which means either:

* the mode isn’t actually selected in shader, or
* the branch returns a constant, or
* the path is being short-circuited / not writing expected values.

**Work**

1. **End-to-end debug mode wiring**

   * Rust: log `VF_COLOR_DEBUG_MODE` final resolved value (already hinted).
   * WGSL: add a one-line “mode stamp” overlay (e.g., add `mode/255` to one channel) so you can see *which* mode actually drew.

2. **Non-uniformity tests**

   * For modes that should vary spatially (24/25/26/27), assert image variance is above epsilon.

**Deliverables**

* `reports/flake/debug_grid.png` (2×3 grid: mode 23–27 + baseline)
* `tests/test_debug_modes_nonuniform.py` (fails if mode 26/27 are flat)
* `reports/flake/debug_mode_log.txt` (captured stdout with resolved mode values)

**Acceptance**

* Mode 26 (Height LOD) shows smooth gradients (not a flat frame).
* Mode 27 (Normal Blend) clearly varies across the screen.
* Tests fail if those become flat again.

---

## Milestone 2 — LOD-consistent Sobel normals (the real fix)

**Goal:** every one of the 9 Sobel taps samples the **same explicit LOD**, and the offsets are computed in **that LOD’s texel units**.

**Work**

1. `compute_height_lod(uv)` from `dpdx/ddy(uv)` footprint (screen-space UV derivatives).
2. `sample_height_geom_level(uv, lod)` using `textureSampleLevel`.
3. `calculate_normal_lod_aware(uv)`:

   * compute `texel_size_lod = 1.0 / textureSize(height_tex, lod_int)`
   * apply Sobel offsets using `texel_size_lod`
   * take all 9 samples at the same `lod`
4. Keep `calculate_normal_ddxddy(world_pos)` as “ground truth” comparator.

**Deliverables**

* `reports/flake/before_after.png` (baseline vs fixed)
* `reports/flake/mode25_ddxddy.png` (should be clean/stable)
* `reports/flake/mode24_no_height_normal.png` (should closely match fixed in flake regions)
* `reports/flake/mode26_height_lod.png` (verifies LOD field is sane)

**Acceptance**

* Material mode no longer exhibits the fine “salt” flakes at grazing angles.
* Mode 25 (ddx/ddy normal) and the fixed Sobel look qualitatively consistent (no high-frequency popping).
* LOD visualization behaves smoothly (no hard discontinuities except expected transitions).

---

## Milestone 3 — Bandlimit/fade strategy for extreme minification (polish + robustness)

Even with LOD-consistent Sobel, very far/minified views can still introduce high-frequency normal energy. Your “fade normal blend from LOD 1→4” idea is exactly the right kind of stabilization knob.

**Work**

* Implement `normal_blend = lerp(user_blend, 0.0, smoothstep(lod_lo, lod_hi, lod))`
* Optionally clamp maximum normal perturbation magnitude as LOD increases.

**Deliverables**

* `reports/flake/normal_blend_curve.png` (plot or table of blend vs lod)
* `reports/flake/mode27_normal_blend.png` (visual proof it’s applied)
* `p5_meta.json` (or similar) updated with parameters: `{lod_lo, lod_hi, blend_min, blend_max}`

**Acceptance**

* No “popping” when orbiting camera (temporal stability improves).
* Far-field stays stable without washing out near-field detail.

---

## Milestone 4 — Regression guardrails (so flakes never come back)

**Work**

* Add a simple “flake score” metric to CI:

  * render mode 23 (no specular) and baseline material
  * compute high-frequency energy (e.g., Laplacian magnitude percentile) in a fixed ROI
  * require it below a threshold for the fixed configuration

**Deliverables**

* `tests/test_flake_regression.py` (produces `reports/flake/flake_score.json`)
* `reports/flake/flake_score.json` (p95, p99, and pass/fail)

**Acceptance**

* CI fails if someone reintroduces implicit-LOD height sampling or mismatched texel offsets.

---

### What I’d do first (if you want the most leverage)

1. **Milestone 1** (debug modes non-uniform + grid proof) — because without this, you can’t trust the diagnosis images.
2. **Milestone 2** (LOD-consistent Sobel) — because it targets the exact mismatch you documented.

If you paste the current implementations of `compute_height_lod()` and the debug-mode `switch`/`if` chain in `fs_main`, I can point out the most likely reason those modes are still producing solid-color frames and propose the precise fix.
