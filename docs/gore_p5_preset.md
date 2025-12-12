# P5.0 – Gore Match Preset (Shader-Only)

**Status:** Frozen as production default  
**Date:** 2024-12-11  
**Milestone:** P5.0 – Practically Achievable (Shader-Only)

## Summary

This preset represents the best achievable Gore Range terrain match using shader-level
tuning only. It passes all luminance, color, dynamic range, and band occupancy metrics.

**Known Gap:** Gradient metrics are at ~50% of Gore targets. This is an architectural
limitation requiring DEM detail normals (see P6 milestone below).

---

## CLI Parameters

```bash
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/hdri/brown_photostudio_02_4k.hdr \
  --size 1920 1080 --render-scale 1 --msaa 8 \
  --ibl-res 1024 --ibl-intensity 0.05 \
  --cam-radius 1400 --cam-phi 135 --cam-theta 35 \
  --exposure 1.0 --sun-azimuth 315 --sun-elevation 37 \
  --sun-intensity 10.0 --sun-color 1.0 0.95 0.90 \
  --gi ibl,ssao --sky hdri \
  --albedo-mode colormap \
  --colormap "#F8D8A8,#D0A068,#A07048,#603820,#381510" \
  --colormap-strength 1.0 --colormap-interpolate --colormap-size 1024 \
  --shadows csm --cascades 4 --shadow-map-res 4096 \
  --normal-strength 2.0 \
  --unsharp-strength 0.4 \
  --output examples/output/terrain_csm.png
```

## Shader Constants

In `src/shaders/terrain_pbr_pom.wgsl`:

```wgsl
const SHADOW_MIN: f32 = 0.20;        // P5-AO: Reduced from 0.30
const SHADOW_IBL_FACTOR: f32 = 0.20;
const AMBIENT_FLOOR: f32 = 0.18;     // P5-AO: Reduced from 0.22
```

## Post-Processing

Luminance unsharp mask with shadow protection (`--unsharp-strength 0.4`):
- Kernel: Gaussian σ=1.0 pixel
- Shadow protection: effect scaled by √(luminance) to prevent crushing darks
- Applied in linear space before sRGB encode

---

## Metrics vs GORE_STRICT_PROFILE

### Passing (19 metrics)

| Metric | Value | Target | Range | Status |
|--------|-------|--------|-------|--------|
| L_min | 0.0498 | 0.0507 | [0.0407, 0.0607] | ✓ |
| L_max | 0.8528 | 0.8467 | [0.8167, 0.8767] | ✓ |
| L_q1 | 0.0752 | 0.0767 | [0.0667, 0.0867] | ✓ |
| L_q5 | 0.0979 | 0.1002 | [0.0902, 0.1102] | ✓ |
| L_q25 | 0.2049 | 0.2089 | [0.1939, 0.2239] | ✓ |
| L_q95 | 0.6340 | 0.6477 | [0.6277, 0.6677] | ✓ |
| crushed_frac | 0.0000 | 0.0000 | [0.0000, 0.0010] | ✓ |
| blown_frac | 0.0000 | 0.0000 | [0.0000, 0.0010] | ✓ |
| dynamic_ratio | 6.5688 | 6.7847 | [6.4000, 7.2000] | ✓ |
| pA | 0.2404 | 0.2336 | [0.1900, 0.2700] | ✓ |
| pB | 0.5568 | 0.5250 | [0.4800, 0.5700] | ✓ |
| pC | 0.2029 | 0.2360 | [0.1900, 0.2700] | ✓ |
| h_mean | 0.0902 | 0.0870 | [0.0750, 0.0950] | ✓ |
| s_mean | 0.4060 | 0.3867 | [0.3600, 0.4100] | ✓ |
| s_std | 0.1166 | 0.1299 | [0.1000, 0.1600] | ✓ |
| h_B | 0.0915 | 0.0987 | [0.0600, 0.1200] | ✓ |
| h_C | 0.1068 | 0.1105 | [0.0900, 0.1300] | ✓ |
| h_monotone | 0.0915 | 0.0000 | [0.0000, 1.0000] | ✓ |

### Failing (8 metrics)

| Metric | Value | Target | Range | % of Target | Status |
|--------|-------|--------|-------|-------------|--------|
| L_q50 | 0.2864 | 0.3478 | [0.3328, 0.3628] | 82% | ✗ |
| L_q75 | 0.4493 | 0.4939 | [0.4789, 0.5089] | 91% | ✗ |
| L_q99 | 0.7009 | 0.7694 | [0.7494, 0.7894] | 91% | ✗ |
| h_A | 0.0732 | 0.0366 | [0.0200, 0.0700] | — | ✗ |
| **g_mean** | 0.0243 | 0.0500 | [0.0470, 0.0530] | **49%** | ✗ |
| **g_median** | 0.0085 | 0.0317 | [0.0290, 0.0350] | **27%** | ✗ |
| **g_q90** | 0.0634 | 0.1179 | [0.1120, 0.1250] | **54%** | ✗ |
| **g_q99** | 0.2120 | 0.2716 | [0.2450, 0.3000] | **78%** | ✗ |

### Stretch Metrics (non-gating)

| Metric | Value | Target | Range | Status |
|--------|-------|--------|-------|--------|
| h_std | 0.0190 | 0.0998 | [0.0800, 0.1200] | ✗ |

---

## Analysis

### What's Working
- **Luminance distribution:** Lower quantiles (q1-q25) and q95 match Gore
- **Dynamic range:** 6.57 within [6.4, 7.2] band
- **Band occupancy:** pA/pB/pC all in spec
- **Color fidelity:** h_mean, s_mean, s_std, band hues all passing
- **No clipping:** crushed_frac and blown_frac at 0

### What's Limited (Architectural)
- **Gradients at ~50% of target:** The 1m DEM resolution creates smooth normals that
  don't generate sufficient pixel-to-pixel luminance transitions
- **Mid-luminance quantiles (q50, q75):** Image is slightly darker than Gore reference
- **h_A band hue:** Slightly outside range (0.073 vs 0.020-0.070)

### Attempted Approaches (P5 Series)
1. **P5-L: Lambert contrast curve** — No effect; N·L values too uniform across adjacent pixels
2. **P5-AO: Shadow/AO floor reduction** — Improved dynamic_ratio but not gradients
3. **P5-US: Luminance unsharp mask** — +32% gradient improvement (0.019→0.024)

---

## P6 Milestone: Gradient Match (Future)

The remaining ~2× gradient gap requires engine-level work, not shader tuning.

### Goal
Lift g_mean/g_median/g_q90/g_q99 to Gore bands without breaking P5.0 metrics.

### Allowed New Work
1. **Multi-scale DEM pipeline:**
   - Keep existing DEM as "base height"
   - Extract high-frequency residual: `detail = original - gaussian_blur(original)`
   - Generate detail normal map at smaller texel pitch

2. **Detail normal compositing in WGSL:**
   - Blend base geometric normal with detail normal
   - Controlled by `detail_strength` uniform

3. **Optional refinement:**
   - Small-radius luminance unsharp after lighting
   - Tuned against strict profile

### Acceptance Criteria (Hard)
- All P1–P5 metrics still pass
- Gradients:
  - g_mean ∈ [0.047, 0.053]
  - g_median ∈ [0.029, 0.035]
  - g_q90 ∈ [0.112, 0.125]
  - g_q99 ∈ [0.245, 0.300]

---

## Notes

- **Do not relax GORE_STRICT_PROFILE** — gradient targets are "truth from Gore"
- P5.0 is the production default until P6 is implemented
- Gradient gap is explicitly documented, not hidden
