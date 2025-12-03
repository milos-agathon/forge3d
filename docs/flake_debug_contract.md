# Flake Debug Contract

This document defines the canonical encoding for debug mode outputs in the terrain shader.

## Normal Encoding (C0)

All normal debug outputs (modes 24, 25, and internal normal visualizations) use:

```
RGB = encode(n) = n * 0.5 + 0.5
```

Where `n` is the normalized surface normal in [-1, 1] range per component.

### Decoding

To recover the normal vector from RGB:

```
n = RGB * 2.0 - 1.0
```

### Alpha Channel Validity Mask

The alpha channel encodes validity:

| Value | Meaning |
|-------|---------|
| `A = 1.0` (255) | Valid normal: finite, `|n| > 0.5` before normalization, finite after |
| `A = 0.0` (0)   | Invalid: NaN, degenerate, or zero-length normal |

### Shader Reference

See `src/shaders/terrain_pbr_pom.wgsl`:
- `DBG_FLAKE_NO_HEIGHT_NORMAL` (mode 24)
- `DBG_FLAKE_DDXDDY_NORMAL` (mode 25)

## Grayscale Encoding (Modes 26, 27)

Modes 26 and 27 output grayscale values:

| Mode | Output | Range |
|------|--------|-------|
| 26 (Height LOD) | `LOD / maxLOD` | [0, 1] → grayscale |
| 27 (Normal Blend) | `blend factor` | [0, 1] → grayscale |

Where:
- `blend = 1.0` means full height-normal contribution
- `blend = 0.0` means geometric normal only (far field)

## Luma Conversion (Rec.709)

For consistency, all luma calculations use Rec.709 coefficients:

```
Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
```

## Thresholds

### Non-Uniformity (B2)

For modes 26 and 27 to be valid:
- `mean ∈ [0.05, 0.95]` (not clipped)
- `p95 - p05 ≥ 0.25` (sufficient gradient)
- `unique_bins ≥ 64` (real gradient, not banding)

### Angular Error (C2)

For near-field (LOD ≤ 1.5):
- `θ_p50 ≤ 3°`
- `θ_p95 ≤ 12°`
- `θ_max ≤ 35°`

For mid-field (1.5 < LOD ≤ 4.0):
- `θ_p50 ≤ 6°`
- `θ_p95 ≤ 18°`
- `θ_max ≤ 45°`

## Bandlimit Fade (D1)

The normal blend function is:

```
blend = 1.0 - smoothstep(lod_lo, lod_hi, lod)
```

With defaults:
- `lod_lo = 1.0`
- `lod_hi = 4.0`

Properties:
- Monotonic decreasing
- `blend(lod ≤ lod_lo) = 1.0`
- `blend(lod ≥ lod_hi) = 0.0`
- `blend(lod = 2.5) = 0.5` (midpoint for symmetric range)
