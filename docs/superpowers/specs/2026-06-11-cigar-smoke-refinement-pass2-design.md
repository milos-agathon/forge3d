# Cigar Smoke Aesthetic Refinement Pass 2

**Date:** 2026-06-11  
**Scope:** Parameter tuning for `examples/california_cigar_smoke_demo.py`  
**Goal:** Reduce late-frame blanket effect, preserve atmospheric haze presence, add minor near-source streamer variation

## Problem Statement

After pass 1, the smoke aesthetic is good but has two issues observed at 6.5-7.8s:

1. **Main issue:** The downwind plume becomes a broad, fairly uniform sheet along the lower/right side. It reads as atmospheric but loses internal ribbon structure - appears as a soft blanket.

2. **Secondary issue:** The near-source plume neck is smooth and beam-like in some frames. Could use slightly more lateral streamer variation close to the fire complex.

## Approach

**Approach A with light touch from C** (approved):
- Reduce haze accumulation coefficients
- Widen texture modulation range for more internal contrast
- Modest blur reduction and faster decay for sharper haze structure

**Explicitly avoided:**
- Age-aware hole amplification (risk of procedural carving look)
- Changes to `streamer_width`, plume-tail decay, or hole thresholds (affect silhouette and mass balance)

## Changes

### 1. Haze Feed Reduction

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_update_residual_haze()` line 2152

Reduce accumulation coefficients by ~12-17% to prevent blanket buildup:

```python
# Before (pass 1)
haze_feed = np.clip(old_smoke * 0.015 + high_slab * 0.0058 + self.density * 0.0032, 0.0, 1.0)

# After (pass 2)
haze_feed = np.clip(old_smoke * 0.0125 + high_slab * 0.0050 + self.density * 0.0028, 0.0, 1.0)
```

| Coefficient | Before | After | Change |
|-------------|--------|-------|--------|
| old_smoke | 0.015 | 0.0125 | -17% |
| high_slab | 0.0058 | 0.0050 | -14% |
| density | 0.0032 | 0.0028 | -12% |

### 2. Texture Modulation Widening

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_update_residual_haze()` line 2159

Increase texture influence range for more internal tonal variation:

```python
# Before (pass 1)
injected = (broad_feed + regional_feed) * np.clip(0.66 + 0.34 * texture, 0.42, 1.06)

# After (pass 2)
injected = (broad_feed + regional_feed) * np.clip(0.62 + 0.46 * texture, 0.34, 1.12)
```

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| Base | 0.66 | 0.62 | Lower floor |
| Multiplier | 0.34 | 0.46 | More texture sensitivity |
| Lower bound | 0.42 | 0.34 | Darker possible |
| Upper bound | 1.06 | 1.12 | Brighter possible |

### 3. Haze Decay and Blur Reduction

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_update_residual_haze()` lines 2160, 2162

Faster decay and sharper edges:

```python
# Before (pass 1)
residual = advected * 0.993 + injected
residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.65)

# After (pass 2)
residual = advected * 0.990 + injected
residual = _pil_blur_float(np.clip(residual, 0.0, 1.15), 1.40)
```

| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| Decay | 0.993 | 0.990 | Faster thinning |
| Blur radius | 1.65 | 1.40 | Sharper structure |

**Note:** Do not go below 1.40 blur in this pass. If 6.5s/7.8s frames still look flat, that would be a pass 3 consideration.

### 4. Near-Source Curl Amplitude

**File:** `examples/california_cigar_smoke_demo.py`  
**Function:** `_inject_hybrid_sources()` line 1912

Minor increase to first curl term for near-source lateral variation:

```python
# Before (pass 1)
curl_offset = radius * (
    2.8 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
    + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + source.seed * 0.009)
    + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + source.seed * 0.005 + frame_index * 0.012)
)

# After (pass 2)
curl_offset = radius * (
    3.1 * np.sin(along / max(radius * 8.0, 1.0) + source.seed * 0.017 + frame_index * 0.025)
    + 4.5 * along_frac * np.sin(along / max(radius * 15.0, 1.0) + source.seed * 0.009)
    + 2.0 * (along_frac ** 0.7) * np.sin(along / max(radius * 28.0, 1.0) + source.seed * 0.005 + frame_index * 0.012)
)
```

Only the first amplitude changes: `2.8 -> 3.1` (+11%)

## Validation

1. Run existing test suite to ensure no regressions
2. Render 8s clip
3. Compare frames at 6.5s and 7.8s against pass 1 render

**Success criteria:**
- Downwind sheet retains atmospheric haze presence
- Shows more internal tonal variation (ribbons visible within haze)
- Less uniform opacity in broad areas
- Near-source plume has subtle lateral variation (not beam-like)

## Not Changed This Pass

These parameters are explicitly preserved to maintain smoke mass balance:

- `streamer_width` (affects main silhouette)
- `plume_tail` exponential decay (affects mass distribution)
- `holes` thresholds (risk of procedural carving)
- All color parameters (pass 1 values are good)
- All edge softness parameters (pass 1 values are good)

## Rollback

If any change causes visual degradation:
1. Revert that specific parameter to pass 1 value
2. Re-render and compare
3. Document for pass 3 consideration
