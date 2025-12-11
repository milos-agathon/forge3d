# prompt


You must fully read AGENTS.md to get familiar with my codebase rules. You must then rigorously implement the following changes to make the output of this code more similar to the attached image. Here are the specs:

Below is a “spec sheet” of **hard requirements** to push terrain_csm.png toward Gore_Range_Albers_1m.png in look & feel. Think of Gore_Range_Albers_1m.png as the *ground-truth reference render*.

I’ve broken things down into categories with strict, testable conditions.

---

## A. Global luminance & contrast

Use Gore_Range_Albers_1m.png’s luminance distribution as the target.

* **[L-01] Luminance range (non-water):**

  * Target min luminance ≥ **0.05** (no pure black terrain).
  * Target max luminance ≤ **0.85** (no blown highlights).
* **[L-02] Luminance quantiles (non-water):** for the final PNG of terrain_csm:

  * 5th percentile: **0.10 ± 0.02**
  * 25th percentile: **0.21 ± 0.03**
  * Median: **0.35 ± 0.03**
  * 75th percentile: **0.49 ± 0.03**
  * 95th percentile: **0.65 ± 0.03**
* **[L-03] Global brightness shift:**
  terrain_csm is currently much darker (median ~0.12 vs ~0.35). Raise exposure so its luminance quantiles match the ranges above.
* **[L-04] Gamma / tone curve:**

  * Apply a **single monotonic tone curve** (e.g. filmic or simple power-gamma) to terrain color *after* lighting.
  * Do **not** clip more than **0.5 %** of pixels at 0 or 1 in any RGB channel.

---

## B. Saturation & color variability

Current terrain_csm is too uniformly saturated & magenta.

* **[C-01] Mean saturation (non-water):**

  * Target mean saturation ≈ **0.25 ± 0.03** (HLS space).
  * Target saturation standard deviation ≥ **0.09** (to avoid flat, posterized color).
* **[C-02] Reduce magenta bias:**

  * Red channel must not exceed green by > **0.08** (normalized 0–1) for more than **20 %** of terrain pixels.
  * Hypsometric ramp should sit in an **earthy tan–brown–rose** band, not magenta.
* **[C-03] Preserve neutral shadows:**

  * For pixels with luminance < **0.20**, chroma must be reduced by at least **30 %** relative to midtones so shadows are neutral, not purple.

---

## C. Hypsometric colormap (terrain tint)

Visually match the reference ramp:

* **[H-01] Color stops (approx.):**

  * Low elevations: light olive-tan (H ≈ 40°, S ≈ 0.25, L ≈ 0.55).
  * Mid elevations: muted warm brown / rose (H ≈ 20°, S ≈ 0.25–0.30, L ≈ 0.45).
  * High elevations: pale beige / very light tan, **not pure white** (H ≈ 35°, S ≤ 0.18, L ≈ 0.75).
* **[H-02] Elevation mapping:**

  * Map DEM linearly into this ramp; avoid strong non-linear jumps that cause banding.
  * Color step between adjacent 1-pixel elevation levels must be < **5/255** in any channel.
* **[H-03] Separation of shading vs tint:**

  * Compute base hypsometric color from elevation **before** lighting.
  * Apply lighting (N·L, AO, SSGI) by modulating brightness, **not** re-indexing the colormap.

---

## D. Sun / key light

Match the reference hillshade feel.

* **[S-01] Direction:**

  * Use a single directional sun with azimuth ≈ **315°** (from NW) and elevation ≈ **35–40°** above the horizon.
  * In world space, a plausible direction is dir = normalize(vec3(-1.0, -1.0, 1.1)).
* **[S-02] Intensity:**

  * Direct light intensity scaled so that fully lit, medium-slope pixels (no AO) reach luminance ≈ **0.55–0.65** after tone-mapping when base color L ≈ 0.45.
* **[S-03] Specular:**

  * Terrain should appear mostly diffuse. Specular contribution at grazing angles must be ≤ **20 %** of total luminance for rock/soil; no obvious glints on ridges.

---

## E. Ambient light, AO, and GI composition

terrain_csm currently crushes valleys into near-black. You want a more “hillshade-like” softness.

* **[A-01] Ambient floor:**

  * Add a hemispherical or constant ambient term so that even fully shadowed terrain has **L ≥ 0.10** after tone mapping.
* **[A-02] AO usage:**

  * AO must **multiply diffuse only**, never specular.
  * AO strength (darkening) should not exceed **35 %** in the deepest occlusion regions.
* **[A-03] SSGI / indirect light:**

  * SSGI should brighten shadowed regions slightly, not over-darken them.
  * Clamp SSGI contribution so that diffuse + SSGI never exceeds 1.2 × the unshadowed diffuse for that pixel.
* **[A-04] Balance:**

  * For pixels in deep valleys (local slope high, SDF high), the luminance ratio between fully lit ridge tops and valley floors should be **between 2:1 and 3.5:1**.
  * Reject any configuration where this ratio exceeds **4.5:1** (too crushed) or is below **1.5:1** (too flat).

---

## F. CSM shadows (core of terrain_csm)

We need crisp, *smoothly fading* shadows without cascade artifacts.

* **[CSM-01] No pure-black shadows:**

  * In shadowed terrain areas, luminance must remain ≥ **0.08**.
  * Hard fail if > **0.5 %** of terrain pixels fall below **0.03**.
* **[CSM-02] Penumbra softness:**

  * Apply PCF or similar filtering so shadow edges transition from fully lit to fully shadowed over **2–5 pixels** at this output resolution.
* **[CSM-03] Cascade transitions:**

  * CSM cascade boundaries must be invisible:

    * No straight lines or sharp steps in shadow intensity over distances > **16 px**.
    * Shadow depth and bias interpolation between cascades must ensure a maximum depth discontinuity of **< 0.5 % of total scene depth range**.
* **[CSM-04] Depth bias:**

  * Choose slope-scale + constant bias so that:

    * Self-shadow acne is absent on gentle slopes (no speckled dark noise).
    * Shadow detachment (“Peter Panning”) from steep cliffs is < **1 px** at this resolution.
* **[CSM-05] Shadow opacity:**

  * Combine CSM’s shadow factor with ambient term as:
    L_final = L_direct * shadow + L_ambient where shadow ∈ [0.25, 1.0].

    * Never allow shadow to be 0 for terrain.

---

## G. Terrain detail & filtering

Gore_Range_Albers_1m shows crisp micro-relief; terrain_csm loses it in dark mush.

* **[D-01] Normal map fidelity:**

  * Compute normals from DEM at **1–2× the DEM texel frequency**, not from a heavily blurred height field.
  * Normal variance across a 64×64 valley patch should be within **10 %** of the reference.
* **[D-02] Anisotropic sampling:**

  * Use anisotropic filtering or high-quality trilinear sampling for DEM/normal textures to avoid stair-stepping or moiré on oblique slopes.
* **[D-03] Preserve micro-contrast:**

  * Do **not** apply large-radius blurs or bilateral filters after shading.
  * Edge contrast (difference in luminance across a 3-px ridge cross-section) should be within **±15 %** of the reference image.

---

## I. Composition & view

Even though the extents differ, the local “read” of ridges and valleys must remain similar.

* **[V-01] Vertical exaggeration:**

  * Match reference apparent z-scale:

    * Height difference between main ridge tops and valley floors in screen space should be visually comparable (no more than **±10 %** change in perceived relief).
  * Avoid additional exaggeration that introduces unnaturally steep walls.
* **[V-02] Camera:**

  * Use similar oblique orthographic / low-FOV perspective.
  * If using perspective, keep FOV ≤ **25°** to avoid wide-angle distortion of ridges.

---

## J. Export quality

* **[X-01] Bit depth:**

  * Output 16-bit or high-quality 8-bit PNG, but only after tone mapping; no intermediate quantization.
* **[X-02] Color space:**

  * Work in linear space internally; convert to sRGB only at the very end.
* **[X-03] Consistency:**

  * For any small crop taken from similar terrain regions in both renders, mean luminance and mean hue must agree within **ΔL ≤ 0.05** and **ΔH ≤ 8°** in HSL space.
 
Here is the code:
  
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

You must assess the changes needed in both the backend and frontend. After every edit you make, immediately run the code. If the output doesn't meet the expectations: keep iterating (edit → run tests) until you meet them fully. Do not stop early. Do not claim success without a green test run. Do not stop until you are confident that your results meet the expectations.
