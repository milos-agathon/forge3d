# prompt


You must fully read AGENTS.md to get familiar with my codebase rules. 
You must then rigorously implement the following changes to make the output of this code more similar to the attached image. 

## RULES

1. Keep your existing **P2 + P3 validated harness**.
2. Add these **P4 tests** (L1–L2, G1–G2, C1–C3) as **additional acceptance checks**.
3. Tell the model explicitly:

   * “Do not regress existing passing tests.”
   * “You now additionally must pass P4-L, P4-G, P4-C; focus especially on raising valley gradients and midtone occupancy while increasing hue variation in the colormap.”

You must assess the changes needed in both the backend and frontend. After every edit you make, immediately run the code. If the output doesn't meet the expectations: keep iterating (edit → run tests) until you meet them fully. Do not stop early. Do not claim success without a green test run. Do not stop until you are confident that your results meet the expectations.


## CODE

You must run this code as you make changes
  
  python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/hdri/brown_photostudio_02_4k.hdr \
  --size 1920 1080 --render-scale 1 --msaa 8 \
  --ibl-res 1024 --ibl-intensity 0.5 \
  --cam-radius 1400 --cam-phi 135 --cam-theta 35 \
  --exposure 0.95 --sun-azimuth 135 --sun-elevation 15 \
  --sun-intensity 0.35 --sun-color 1.0 0.72 0.85 \
  --gi ibl,ssao --sky hdri \
  --albedo-mode colormap \
  --colormap "#dcd098,#c07848,#b85565,#301815" \
  --colormap-strength 1.0 --colormap-interpolate --colormap-size 1024 \
  --shadows csm --cascades 4 --shadow-map-res 4096 \
  --output examples/output/terrain_csm.png --overwrite

## OBJECTIVE

This document consolidates all current milestones and requirement layers for
matching `terrain_csm.png` to the reference `Gore_Range_Albers_1m.png`.

The pipeline is:

- **Base requirements:** `R0–R8`
- **Milestone P1:** Baseline reproduction & sanity checks
- **Milestone P2:** Luminance & dynamic-range locking
- **Milestone P3:** Gradient / spatial-structure & global hue
- **Milestone P4:** Valley structure, midtones, and hue variation

All metrics apply **only to terrain pixels (non-water)**.

---

## 0. Base Requirements – R0–R8

These are *global* constraints that must hold for all milestones P1–P4.

### R0 – Non-Water Mask

- Use DEM/land mask + blue dominance to exclude water:
  - `nonwater = land_mask & not(water_like)`.
- All metrics below are computed only on `nonwater`.

### R1 – Luminance & Tone Curve

- Luminance `L = 0.2126R + 0.7152G + 0.0722B` in consistent (linear or sRGB) space.
- Terrain luminance bounds:
  - `0.04 ≤ L_min ≤ 0.08`
  - `0.80 ≤ L_max ≤ 0.90`
- Quantile targets (±0.03 unless overridden by P2–P4):

  | q   | ref value |
  |-----|-----------|
  | q01 | 0.077     |
  | q05 | 0.100     |
  | q25 | 0.209     |
  | q50 | 0.348     |
  | q75 | 0.494     |
  | q95 | 0.648     |
  | q99 | 0.769     |

- Crushed/blown limits:
  - `< 0.5%` of pixels with `L < 0.04`.
  - `< 0.5%` of pixels with `L > 0.85`.

### R2 – Local Contrast & Terrain Detail

- Gradient magnitude `g` of `L` (forward differences).
- Reference stats (terrain):

  - `mean(g_ref) ≈ 0.050`
  - `median(g_ref) ≈ 0.032`
  - `q90(g_ref) ≈ 0.118`
  - `q99(g_ref) ≈ 0.272`

- Target bands (±15% around reference) unless tightened by P2/P3.

### R3 – Global Color Statistics

- Convert terrain pixels to HSV.
- Reference (terrain):

  - `h_mean_ref ≈ 0.087` (warm tan, ~25°)
  - `h_std_ref ≈ 0.100`
  - `s_mean_ref ≈ 0.387`
  - `s_std_ref ≈ 0.130`

- No strong magenta / purple:  
  fraction of pixels with `h ∈ [0.70,0.95]` and `s > 0.20` ≤ 1%.

### R4 – Color by Luminance Band

Bands:

- A: `0.05 ≤ L < 0.20` (shadows)
- B: `0.20 ≤ L < 0.50` (midtones)
- C: `0.50 ≤ L < 0.80` (highlights)

Reference pattern:

- Shadows: saturated, slightly red-brown.
- Midtones: earthy tan/rose.
- Highlights: paler, less saturated warm tan.

Monotone hue progression: `mean(h_A) < mean(h_B) < mean(h_C)`.

### R5 – Hypsometric Ramp

- Single continuous 1D ramp; no discrete steps.
- Small elevation step `Δz = 1/2048` must produce:

  `max(|ΔR|,|ΔG|,|ΔB|) ≤ 5/255`.

- Anchor colors (approx, sampled from ramp, *not* shaded image):

  - `z ≈ 0.2` → olive-tan.
  - `z ≈ 0.5` → muted warm brown/rose.
  - `z ≈ 0.9` → pale warm tan (not white).

### R6 – Direct Lighting & Ambient Composition

- Sun direction: azimuth `315° ± 5°`, elevation `37° ± 5°`.
- `lambert = max(dot(N, L_dir), 0.0)`.
- Ambient floor in shader: `0.22 ≤ ambient_floor ≤ 0.38`.
- AO clamp: `ao = max(ao, 0.65)` (≤ 35% darkening).
- Shadows: `shadow = clamp(shadow, 0.30, 1.0)`.
- Specular contribution for terrain ≤ 25% of total RGB per pixel.

### R7 – CSM Shadow Quality

- No pure-black shadows: shadowed terrain `L ≥ 0.08`, `<0.5%` below `0.04`.
- Penumbra width: 2–6 px from fully lit to fully shadowed.
- No visible cascade seams (no long straight steps in `shadow`).
- Depth bias tuned to avoid acne & “Peter Panning” (>1 px separation).

### R8 – Camera & Projection

- Vertical exaggeration within ±10% of reference.
- Perspective FOV ≤ 25°, or orthographic.

---

## Milestone P1 – Baseline Gore Range Reproduction

> **Goal:** Reproduce a visually plausible Gore Range terrain render with correct camera, basic hillshade, and color ramp, without enforcing tight statistical matching.

### Inputs

- DEM: `assets/Gore_Range_Albers_1m.tif`
- HDR: `brown_photostudio_02_4k.hdr` (or equivalent).
- Command-line interface for `terrain_demo.py`.

### Tasks

- [ ] Implement non-water mask (R0).
- [ ] Set camera, DEM, z-scale, and FOV to match reference framing (R8).
- [ ] Implement basic lighting (sun direction ~NW, ambient, AO, shadows) (R6, R7).
- [ ] Implement hypsometric ramp with approximate earthy tan-brown-beige colors (R5).
- [ ] Ensure 16-bit/8-bit export with correct sRGB conversion (R1).

### Acceptance Criteria

- [ ] Image visually recognizes as the same area as `Gore_Range_Albers_1m.png`.
- [ ] All **structural** R0–R8 constraints satisfied in “loose” bands:
  - No pure black/white terrain pixels.
  - Correct sun direction by inspection.
  - Water masked out of metrics.
- [ ] P1 validation script passes (basic histogram sanity, camera match, no obvious artifacting).

---

## Milestone P2 – Luminance & Dynamic-Range Lock

> **Goal:** Make the luminance distribution of `terrain_csm.png` statistically match the reference (shadows, mids, highlights) while preserving R0–R8.

### Additional Requirements (P2)

#### P2-L – Luminance Quantiles & Dynamic Range

- Precise quantile matching (terrain only):

  For `q ∈ {1,5,25,50,75,95,99}` with reference `q_ref`:

  - `|q - q_ref| ≤ 0.030`.

- Dynamic ratio:

  ```python
  q10 = quantile(L, 0.10)
  q90 = quantile(L, 0.90)
  ratio = mean(L[L>=q90]) / mean(L[L<=q10])
````

* `5.5 ≤ ratio ≤ 8.0`.

* Crushed/blown terrain:

  * `< 0.1%` pixels with `L < 0.05`.
  * `< 0.1%` pixels with `L > 0.85`.

#### P2-G – Gradient Distribution (Global)

* Gradient stats must lie within:

  * `0.045 ≤ mean(g) ≤ 0.055`
  * `0.027 ≤ median(g) ≤ 0.040`
  * `0.10 ≤ q90(g) ≤ 0.14`
  * `0.23 ≤ q99(g) ≤ 0.30`

#### P2-Band – Luminance Bands

* Fractions per band:

  * Band A: `0.18 ≤ pA ≤ 0.30`
  * Band B: `0.45 ≤ pB ≤ 0.60`
  * Band C: `0.10 ≤ pC ≤ 0.25`

#### P2-Shader Guards

* Ambient floor, AO clamp, shadow floor, and lighting composition must follow R6/R7 + structural form:

  ```glsl
  float diffuse_raw = mix(ambient_floor, 1.0, lambert);
  float diffuse_lit = diffuse_raw * max(ao, 0.65) * clamp(shadow, 0.30, 1.0);
  float ibl_term    = ibl_diffuse_factor * ambient_floor;
  float lighting_factor = diffuse_lit + ibl_term;
  ```

### Acceptance Criteria

* [ ] All P1 tests pass.
* [ ] P2-L, P2-G, P2-Band constraints pass.
* [ ] No regression on R0–R8.

---

## Milestone P3 – Gradient Structure, Spatial Variation & Global Hue

> **Goal:** Match the *texture* of terrain (ridge/valley detail) and global color statistics, avoiding “flat beige fog”.

### Additional Requirements (P3)

#### P3-L – tightened quantiles & band occupancy

* Same reference quantiles, but stricter bounds (e.g. ±0.01–0.02); dynamic ratio:

  * `6.4 ≤ ratio ≤ 7.2`.

* Bands:

  ```text
  0.18 ≤ pA ≤ 0.30
  0.45 ≤ pB ≤ 0.60
  0.18 ≤ pC ≤ 0.30
  ```

#### P3-G – Gradients (Global)

* Global targets:

  ```text
  0.047 ≤ mean(g)   ≤ 0.053
  0.029 ≤ median(g) ≤ 0.035
  0.112 ≤ q90(g)    ≤ 0.125
  0.245 ≤ q99(g)    ≤ 0.300
  ```

* Band-wise:

  ```text
  0.045 ≤ mean(g_B) ≤ 0.055
  0.044 ≤ mean(g_A) ≤ 0.055
  0.035 ≤ mean(g_C) ≤ 0.050
  mean(g_C) ≤ mean(g_B) + 0.005
  ```

#### P3-T – Spatial Uniformity (3×3 tiles)

* Split terrain into 3×3 tiles; compute median luminance per tile (`m_i`).

  ```text
  0.32 ≤ μ_tiles ≤ 0.40
  0.07 ≤ σ_tiles ≤ 0.12
  (m_max - m_min) ≥ 0.20
  ```

#### P3-C – Global Hue & Saturation

* Global HSV (terrain):

  ```text
  0.075 ≤ h_mean ≤ 0.095
  0.055 ≤ h_std  ≤ 0.120
  0.36 ≤ s_mean ≤ 0.41
  0.10 ≤ s_std  ≤ 0.15
  ```

* Bandwise hue/sat with monotone `mean(h_A) < mean(h_B) < mean(h_C)`.

### Acceptance Criteria

* [ ] All P1 + P2 tests pass.
* [ ] All P3-L, P3-G, P3-T, P3-C constraints satisfied.
* [ ] Visual inspection: no uniform fog, clearly articulated ridges and valleys.

---

## Milestone P4 – Valley Structure, Midtones & Hue Variation

> **Goal:** Fix remaining mismatches: valley gradients, midtone occupancy, and hue variation in the colormap (avoid “single warm tint” look).

### Additional Requirements (P4)

#### P4-L – Band Occupancy & Mid Quantiles (Strict)

* Bands:

  ```text
  0.22 ≤ pA ≤ 0.26
  0.49 ≤ pB ≤ 0.57
  0.20 ≤ pC ≤ 0.26
  ```

* Mid quantiles:

  ```text
  0.20 ≤ q25 ≤ 0.22
  0.34 ≤ q50 ≤ 0.37
  0.48 ≤ q75 ≤ 0.51
  ```

#### P4-G – Valley Gradients

* For Band A (`0.05 ≤ L < 0.20`):

  ```text
  0.024 ≤ mean(g_A) ≤ 0.034
  q90(g_A) ≥ 0.060
  ```

* Band relationships:

  ```text
  mean(g_A) ≤ mean(g_B)
  mean(g_B) ≤ mean(g_C) + 0.010
  ```

#### P4-C – Hue Variation & Progression

* Global hue:

  ```text
  0.075 ≤ h_mean ≤ 0.095
  0.060 ≤ h_std  ≤ 0.130
  ```

* Band B (midtones):

  ```text
  0.08 ≤ mean(h_B) ≤ 0.11
  0.08 ≤ std(h_B)  ≤ 0.15
  ```

* Hue progression by band:

  ```text
  0.02 ≤ mean(h_A) ≤ 0.07
  0.06 ≤ mean(h_B) ≤ 0.11
  0.09 ≤ mean(h_C) ≤ 0.13
  mean(h_A) < mean(h_B) < mean(h_C)
  ```

### Acceptance Criteria

* [ ] All P1, P2, P3 tests pass (no regressions).
* [ ] All P4-L, P4-G, P4-C constraints pass.
* [ ] Final `terrain_csm.png` is visually and statistically within tolerance of `Gore_Range_Albers_1m.png` for lighting, contrast, and color.

---

## Implementation Notes

* Each requirement group (R*, P2, P3, P4) should be mapped to concrete tests in a validation harness, e.g.:

  * `validate_R.py` – structural & basic stats (R0–R8).
  * `validate_P2.py` – luminance & dynamic range.
  * `validate_P3.py` – gradients, tiles, global hue.
  * `validate_P4.py` – valley/midtone/hue refinement.

* CI should run all validators after each change to shaders or terrain parameters and fail if any milestone’s tests regress.
