# prompt

You are an elite graphics/runtime engineer working on the forge3d renderer
(Rust backend, Python frontend) for terrain visualization.

Your current job is **matching a rendered frame of the Gore Range** to a fixed
reference image Gore_Range_Albers_1m.png under a strict quantitative spec
called GORE_STRICT_PROFILE.
You must fully read AGENTS.md to get familiar with my codebase rules. 
You must then rigorously implement the following changes to make the output of this code more similar to the attached image. 

## CRITICAL RULES


   * “Do not regress existing passing tests.”
   
   * “You now additionally must pass P4-L, P4-G, P4-C; focus especially on raising valley gradients and midtone occupancy while increasing hue variation in the colormap.”
   
   * You must assess the changes needed in both the backend and frontend. After every edit you make, immediately run the code. If the output doesn't meet the expectations: keep iterating (edit → run tests) until you meet them fully. Do not stop early. Do not claim success without a green test run. Do not stop until you are confident that your results meet the expectations.
   
1. The values in GORE_STRICT_PROFILE are **ground truth**.  
   - You are **not allowed** to change any targets, ranges, quantiles, or thresholds.
   - You must never “fix” failing tests by editing the validator or relaxing bands.

2. You may only change:
   - CLI parameters for terrain_demo.py, such as:
     - --exposure 
     - --sun-intensity, --sun-azimuth, --sun-elevation (must stay near 315°/37° in the spec)
     - --ibl-intensity 
     - --colormap (5-stop ramp; hue must stay monotone from shadows→highlights)
     - --colormap-strength, --colormap-size 
   - Python terrain params:
     - ClampSettings.ambient_range 
     - ClampSettings.shadow_range 
     - ClampSettings.occlusion_range 
   - Shader parameters and lighting composition:
     - AMBIENT_FLOOR (must stay in spec range)
     - SHADOW_MIN 
     - Ambient/AO/shadow/IBL blending weights
     - Optional hue/saturation adjustments, as long as
       h_A < h_B < h_C remains true and h_mean stays near target.

3. Forbidden changes:
   - No editing GORE_STRICT_PROFILE values or ranges.
   - No changing how metrics are computed (luminance, gradients, HSV, bands).
   - No “bypassing” tests by changing the validator logic.

4. Every time you propose a change, you must:
   - Clearly list exactly which knobs you are changing and by how much.
   - Predict qualitatively how each change will move the failing metrics.
   - After seeing new metrics, explain which changes helped, which hurt, and
     propose the next minimal adjustment.

5. Optimisation objective order:
   1) Match **luminance distribution and band occupancy** (quantiles, dynamic ratio, pA/pB/pC).
   2) Match **gradient statistics** (mean, median, q90, q99).
   3) Match **global hue + saturation** (h_mean, s_mean) and band hues.
   4) Increase **h_std** toward the reference as a stretch goal, **without**
      breaking the other metrics or causing hue wraparound.

6. If a metric looks structurally limited (e.g. h_std or gradients), you must:
   - Keep trying to improve it within allowed knobs.
   - If still unsatisfied, clearly explain *why* it appears structurally limited
     (e.g. normal quality, lighting model), but you may NOT change the target.

Always be explicit, deterministic, and surgical in your edits. Do not make broad
changes if a smaller, more targeted change could solve the specific deviation.


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

Here’s a cleaner, self-contained markdown you can drop straight into the repo as your terrain-matching milestone spec.

````markdown
# forge3d – Gore Range Terrain Matching Milestones

Target: make `terrain_csm.png` statistically and visually match  
`Gore_Range_Albers_1m.png` (ignoring water) under tight, testable constraints.

Milestones build on each other:

- **R0–R8**: Global base requirements (always in force).
- **P1**: Baseline reproduction.
- **P2**: Luminance & dynamic range lock.
- **P3**: Gradient structure & spatial variation.
- **P4**: Valley structure, midtones & hue variation.

---

## R0–R8 – Global Base Requirements (Always On)

### R0 – Non-Water Mask

- Build a `nonwater` mask from DEM/land mask + blue dominance; exclude water.
- All metrics below are computed only on `nonwater`.

### R1 – Luminance & Tone Curve

- Luminance: `L = 0.2126R + 0.7152G + 0.0722B` (consistent space).
- Bounds (terrain):
  - `0.04 ≤ L_min ≤ 0.08`
  - `0.80 ≤ L_max ≤ 0.90`
- Reference quantiles (terrain):

  | q   | value |
  |-----|-------|
  | 1%  | 0.077 |
  | 5%  | 0.100 |
  | 25% | 0.209 |
  | 50% | 0.348 |
  | 75% | 0.494 |
  | 95% | 0.648 |
  | 99% | 0.769 |

- Crushed/blown limits:
  - `< 0.5%` pixels with `L < 0.04`.
  - `< 0.5%` pixels with `L > 0.85`.

### R2 – Local Contrast & Gradient

- Gradient on L:

  ```python
  gx[:,1:] = L[:,1:] - L[:,:-1]
  gy[1:,:] = L[1:,:] - L[:-1,:]
  g = np.sqrt(gx*gx + gy*gy)
````

* Reference (terrain):

  * `mean(g_ref) ≈ 0.050`
  * `median(g_ref) ≈ 0.032`
  * `q90(g_ref) ≈ 0.118`
  * `q99(g_ref) ≈ 0.272`

### R3 – Global Color (HSV)

* Reference (terrain):

  * `h_mean_ref ≈ 0.087` (warm tan)
  * `h_std_ref ≈ 0.100`
  * `s_mean_ref ≈ 0.387`
  * `s_std_ref ≈ 0.130`
* Strong magenta suppressed:
  fraction with `h ∈ [0.70,0.95]` & `s > 0.20` ≤ 1%.

### R4 – Color vs Luminance Bands

Bands:

* A: `0.05 ≤ L < 0.20` (shadows)
* B: `0.20 ≤ L < 0.50` (midtones)
* C: `0.50 ≤ L < 0.80` (highlights)

Constraints:

* Saturation decreases from A → C.
* Mean hue increases from A → C:

  `mean(h_A) < mean(h_B) < mean(h_C)`.

### R5 – Hypsometric Ramp

* Single continuous 1D RGB ramp; no visible steps.

* For elevation step `Δz = 1/2048`:

  `max(|ΔR|,|ΔG|,|ΔB|) ≤ 5/255`.

* Anchor colors (ramp, not shaded):

  * `z ≈ 0.2`: olive-tan.
  * `z ≈ 0.5`: muted rose-brown.
  * `z ≈ 0.9`: pale warm tan (not white).

### R6 – Lighting Composition

* Sun: azimuth `315° ± 5°`, elevation `37° ± 5°`.
* `lambert = max(dot(N, L_dir), 0.0)`.
* Ambient floor in shader: `0.22 ≤ ambient_floor ≤ 0.38`.
* AO clamp: `ao = max(ao, 0.65)`.
* Shadow factor: `shadow = clamp(shadow, 0.30, 1.0)`.
* Specular for terrain ≤ 25% of total RGB.

Canonical lighting structure:

```glsl
float diffuse_raw = mix(ambient_floor, 1.0, lambert);
float diffuse_lit = diffuse_raw * max(ao, 0.65) * clamp(shadow, 0.30, 1.0);
float ibl_term    = ibl_diffuse_factor * ambient_floor;
float lighting_factor = diffuse_lit + ibl_term;
```

### R7 – CSM Quality

* No pure-black shadows: terrain `L ≥ 0.08`, `< 0.5%` below `0.04`.
* Penumbra width: 2–6 px transition.
* No visible cascade seams (large, straight shadow jumps).
* Depth bias tuned: no acne, ≤1 px “Peter Panning”.

### R8 – Camera / Projection

* Vertical exaggeration ±10% of reference.
* Perspective FOV ≤ 25° (or orthographic).

---

## Milestone P1 – Baseline Reproduction

**Goal:** Reproduce a plausible Gore Range render (framing, basic hillshade, color ramp) that satisfies structural R0–R8.

### Tasks

* Wire DEM, camera, z-scale, and FOV to match reference framing.
* Implement sun + ambient + AO + CSM shadows in the form above.
* Implement an earthy tan–brown–beige hypsometric ramp.
* Export correct sRGB PNG with no banding or obvious artifacts.
* Implement non-water mask for all metrics.

### Acceptance

* P1 validation script passes:

  * Correct extent & camera (within tolerance).
  * No crushed/blown terrain.
  * Sun direction visually from NW.
  * R0–R8 structural checks OK.

---

## Milestone P2 – Luminance & Dynamic Range Lock

**Goal:** Match terrain luminance distribution and dynamic range to reference.

### Extra Constraints (on top of R0–R8)

**P2-L – Quantiles & Dynamic Ratio**

For each `q ∈ {1,5,25,50,75,95,99}` with reference `q_ref`:

```text
|q - q_ref| ≤ 0.030
```

Dynamic ratio:

```python
q10 = quantile(L, 0.10)
q90 = quantile(L, 0.90)
ratio = mean(L[L>=q90]) / mean(L[L<=q10])
```

* `5.5 ≤ ratio ≤ 8.0`.

Crushed/blown terrain:

* `< 0.1%` pixels with `L < 0.05`.
* `< 0.1%` pixels with `L > 0.85`.

**P2-Band – Luminance Bands**

Band occupancy:

* `0.18 ≤ pA ≤ 0.30`
* `0.45 ≤ pB ≤ 0.60`
* `0.10 ≤ pC ≤ 0.25`

**P2-G – Gradients (Global)**

* `0.045 ≤ mean(g)   ≤ 0.055`
* `0.027 ≤ median(g) ≤ 0.040`
* `0.10 ≤ q90(g)     ≤ 0.14`
* `0.23 ≤ q99(g)     ≤ 0.30`

### Acceptance

* All P1 tests pass.
* All P2-L, P2-Band, P2-G tests pass.

---

## Milestone P3 – Gradient Structure, Spatial Variation & Global Hue

**Goal:** Match terrain texture (ridges/valleys), spatial brightness patterns, and global color stats.

### Extra Constraints

**P3-L – Tightened Quantiles & Bands**

* Dynamic ratio: `6.4 ≤ ratio ≤ 7.2`.
* Bands:

  * `0.18 ≤ pA ≤ 0.30`
  * `0.45 ≤ pB ≤ 0.60`
  * `0.18 ≤ pC ≤ 0.30`

**P3-G – Gradients**

* Global:

  * `0.047 ≤ mean(g)   ≤ 0.053`
  * `0.029 ≤ median(g) ≤ 0.035`
  * `0.112 ≤ q90(g)    ≤ 0.125`
  * `0.245 ≤ q99(g)    ≤ 0.300`

* Band-wise:

  * `0.045 ≤ mean(g_B) ≤ 0.055`
  * `0.044 ≤ mean(g_A) ≤ 0.055`
  * `0.035 ≤ mean(g_C) ≤ 0.050`
  * `mean(g_C) ≤ mean(g_B) + 0.005`

**P3-T – 3×3 Tile Spatial Stats**

Split terrain into 3×3 tiles; let `m_i` be median `L` per tile.

* `0.32 ≤ mean(m_i) ≤ 0.40`
* `0.07 ≤ std(m_i)  ≤ 0.12`
* `max(m_i) - min(m_i) ≥ 0.20`

**P3-C – Global Hue & Saturation**

* `0.075 ≤ h_mean ≤ 0.095`
* `0.055 ≤ h_std  ≤ 0.120`
* `0.36 ≤ s_mean ≤ 0.41`
* `0.10 ≤ s_std  ≤ 0.15`

Plus R4 bandwise hue progression.

### Acceptance

* P1 + P2 tests pass.
* All P3-L, P3-G, P3-T, P3-C tests pass.

---

## Milestone P4 – Valleys, Midtones & Hue Variation

**Goal:** Fix remaining mismatches: valley gradients, midtone occupancy, and hue variation (avoid “single warm tint”).

### Extra Constraints

**P4-L – Band Occupancy & Mid Quantiles (Strict)**

Bands:

* `0.22 ≤ pA ≤ 0.26`
* `0.49 ≤ pB ≤ 0.57`
* `0.20 ≤ pC ≤ 0.26`

Mid quantiles:

* `0.20 ≤ q25 ≤ 0.22`
* `0.34 ≤ q50 ≤ 0.37`
* `0.48 ≤ q75 ≤ 0.51`

**P4-G – Valley Gradients**

On Band A (`0.05 ≤ L < 0.20`):

* `0.024 ≤ mean(g_A) ≤ 0.034`
* `q90(g_A) ≥ 0.060`

Band relationships:

* `mean(g_A) ≤ mean(g_B)`
* `mean(g_B) ≤ mean(g_C) + 0.010`

**P4-C – Hue Variation & Progression**

Global:

* `0.075 ≤ h_mean ≤ 0.095`
* `0.060 ≤ h_std  ≤ 0.130`

Band B (midtones):

* `0.08 ≤ mean(h_B) ≤ 0.11`
* `0.08 ≤ std(h_B)  ≤ 0.15`

Bandwise means:

* `0.02 ≤ mean(h_A) ≤ 0.07`
* `0.06 ≤ mean(h_B) ≤ 0.11`
* `0.09 ≤ mean(h_C) ≤ 0.13`
* `mean(h_A) < mean(h_B) < mean(h_C)`

### Acceptance

* All P1–P3 tests pass (no regressions).
* All P4-L, P4-G, P4-C tests pass.

---

## Suggested File Layout

* `tests/validate_R.py`     – R0–R8 base checks.
* `tests/validate_P2.py`    – luminance & dynamic range.
* `tests/validate_P3.py`    – gradients, tiles, global hue.
* `tests/validate_P4.py`    – valleys, midtones, hue variation.
* `docs/terrain_gore_range_milestones.md` – this spec.