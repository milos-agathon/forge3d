## Implementation Status (Dec 2025)

**All milestones completed.** The following changes were made to `terrain_pbr_pom.wgsl`:

### Milestone 1 — Debug Modes ✓
- **Mode 23** (`DBG_FLAKE_NO_SPECULAR`): Diffuse-only rendering (no IBL specular)
- **Mode 24** (`DBG_FLAKE_NO_HEIGHT_NORMAL`): Uses `base_normal` instead of height-derived normal
- **Mode 25** (`DBG_FLAKE_DDXDDY_NORMAL`): Uses `n_dd = cross(dpdx(world_pos), dpdy(world_pos))` as ground truth
- **Mode 26** (`DBG_FLAKE_HEIGHT_LOD`): Visualizes computed LOD level (black=0, white=max)
- **Mode 27** (`DBG_FLAKE_NORMAL_BLEND`): Visualizes effective normal_blend after LOD fade

### Milestone 2 — LOD-aware Sobel ✓
- Added `sample_height_geom_level(uv, lod)` for explicit LOD sampling
- Added `compute_height_lod(uv)` to compute LOD from screen-space UV footprint
- Added `calculate_normal_lod_aware(uv)` that uses consistent LOD for all 9 Sobel taps
- Added `calculate_normal_ddxddy(world_pos)` for derivative-based normal comparison
- Default path now uses LOD-aware Sobel (`calculate_normal_lod_aware`)

### Milestone 3 — Minification Fade ✓
- `normal_blend` now fades based on LOD: full at LOD 0-1, fades to 0 by LOD 4
- Prevents sparkles at distance/grazing angles even with correct mip selection

### Triplanar Requirements ✓
- **T1**: `compute_triplanar_weights` normalizes weights so wx + wy + wz = 1
- **T2**: World-position-based UVs prevent stretching on steep slopes
- Uses `base_normal` (not `blended_normal`) for stable triplanar weights

**To test debug modes:** Set `VF_COLOR_DEBUG_MODE=<mode>` environment variable.

---

## Milestone 1 — Prove (or falsify) “it’s specular caused by the height-normal”

### Work

1. Add **one “specular off” switch** in shader (or debug mode):

   * `specular_contrib = vec3(0)` (leave diffuse/IBL intact)
2. Add **one “height normal off” switch**:

   * `height_normal = base_normal` (or `normal_blend = 0`)
3. Add **a derivative-based geometric normal** as a third comparison:

   * `n_dd = normalize(cross(dpdx(world_pos), dpdy(world_pos)))`
   * Use it for shading normal in a debug mode.

### Deliverables

* `reports/.../flake/flake_baseline.png`
* `reports/.../flake/flake_no_specular.png`
* `reports/.../flake/flake_no_height_normal.png`
* `reports/.../flake/flake_ddxddy_normal.png` (shading with derivative normal)

### Acceptance

* If `no_specular` kills flakes ⇒ it’s *spec aliasing*.
* If `no_height_normal` kills flakes ⇒ it’s *height-normal bandwidth*.
* If `ddxddy_normal` looks clean ⇒ current Sobel pipeline is the problem (very likely).

---

## Milestone 2 — Make Sobel height-normal **LOD-aware** (this is the big fix)

Right now:

* `sample_height_geom()` uses `textureSample()` → implicit LOD chosen by hardware
* but Sobel offsets use `texel_size = 1 / textureSize(level0)` → **wrong** if the sampling is happening at mip>0
  This mismatch creates exactly the kind of shimmering “salt/pepper flakes”.

### Work

Replace Sobel with a version that:

1. Computes a consistent mip level from the **screen-space footprint of uv**
2. Uses that mip level for **all 9 samples**
3. Scales Sobel offsets by the mip’s texel size

**Core idea (WGSL sketch):**

```wgsl
let dims = vec2<f32>(textureDimensions(height_tex)); // level 0 dims
let ddx_uv = dpdx(uv);
let ddy_uv = dpdy(uv);

// footprint in texels
let rho = max(length(ddx_uv * dims), length(ddy_uv * dims));
let lod = clamp(log2(max(rho, 1e-8)), 0.0, f32(textureNumLevels(height_tex) - 1u));

let mip_scale = exp2(lod);
let texel_uv = mip_scale / dims;

// then do Sobel taps with textureSampleLevel(height_tex, ..., lod)
```

And in `sample_height_geom`, add a `lod` parameter:

```wgsl
fn sample_height_geom_level(uv: vec2<f32>, lod: f32) -> f32 {
    let uv_clamped = clamp(uv, vec2(0.0), vec2(1.0));
    let h_raw = textureSampleLevel(height_tex, height_samp, uv_clamped, lod).r;
    // ... transform ...
    return h;
}
```

### Deliverables

* New renders:

  * `reports/.../flake/height_normal_before.png`
  * `reports/.../flake/height_normal_after.png`
* New debug output:

  * `reports/.../flake/dbg_height_lod.png` (visualize `lod / max_lod`)
* Optional metric:

  * `flake_score_before`, `flake_score_after` in manifest (HF energy on *specular-only* image is a good metric)

### Acceptance

* Flakes reduced dramatically, especially in far-field + grazing regions.
* `dbg_height_lod` shows sane LOD gradients (not constant 0, not chaotic).

---

## Milestone 3 — Add a **minification fade** for height-normal contribution (safety belt)

Even with correct mip selection, if the height-normal amplitude is too strong when minified, specular can still sparkle.

### Work

Fade `normal_blend` (or reduce Sobel strength) as `lod` increases:

* Example: start fading at `lod > 1`, fully off by `lod > 4` (tune later)

### Deliverables

* `reports/.../flake/flake_after_lod_fade.png`
* `reports/.../flake/dbg_effective_normal_blend.png` grayscale

### Acceptance

* Remaining sparkles disappear without making near-field terrain look “mushy”.

---

## Milestone 4 — Only after flakes are gone: return to SpecAA

Once your height-normal is genuinely mip-filtered and stable, SpecAA tests become meaningful again. Before that, you’re trying to band-aid a signal-processing bug.

### Deliverables

* `fig_specaa_sparkle_test.png` (synthetic) goes PASS *or* marked XFAIL with explicit rationale.
