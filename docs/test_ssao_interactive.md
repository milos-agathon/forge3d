# SSAO/GTAO Interactive Testing Guide

## Launch Viewer
```bash
cargo run --release --example interactive_viewer
```

## Test Sequence

### 1. Enable SSAO
```
:gi ssao on
```
You should see the HUD update with SSAO parameters (orange bars):
- SSAO/GTAO mode indicator (0)
- RAD (radius) ~0.50
- INT (intensity) ~1.00
- AO-T (temporal alpha) ~0.20

### 2. Switch to GTAO
```
:ssao-technique gtao
```
HUD should show mode = 1 instead of 0

### 3. Adjust Parameters
```
:ssao-radius 0.8
:ssao-intensity 1.5
```
Watch HUD bars update in real-time

### 4. Test Temporal Accumulation
```
:ssao-temporal on
:ssao-alpha 0.5
```
Move camera slightly - AO should be more stable but show slight ghosting

```
:ssao-temporal off
```
AO should be noisier but respond immediately to camera motion

### 5. Compare Techniques
```
:ssao-technique ssao
```
(wait a moment, orbit camera)
```
:ssao-technique gtao
```
GTAO should show better horizon-based occlusion

### 6. Test Composite Controls
```
:ssao-composite on
:ssao-mul 0.8
```
Adjust multiplier to control AO strength in final image

### 7. Capture Snapshots
```
:snapshot ssao_default.png
```
(change parameters)
```
:snapshot gtao_comparison.png
```

## Expected Behavior

**SSAO Mode (technique=0)**:
- Hemisphere sampling (16 samples default)
- Softer, more diffuse occlusion
- Temporal dithering reduces flicker

**GTAO Mode (technique=1)**:
- Horizon-based angular integration
- Sharper contact shadows
- Better crevice darkening

**Temporal Resolve**:
- Alpha=0.2: Fast response, some noise
- Alpha=0.5: Balanced stability/ghosting
- Alpha=0.8: Very stable, noticeable trails

**HUD Display**:
- Orange bars for SSAO parameters
- Mode indicator shows 0 (SSAO) or 1 (GTAO)
- Bars scale: radius 0-2m, intensity 0-2x, temporal 0-1

## Visual Checks

1. **Contact shadows**: Dark bands where surfaces meet
2. **Cavity occlusion**: Darkening in crevices and corners
3. **No haloing**: AO should not bleed over edges (bilateral blur prevents this)
4. **Temporal stability**: With temporal on, AO should be smooth when static
5. **Edge preservation**: Sharp geometric edges should remain crisp
