# General HDR Terrain Lighting and Mood Design

**Date:** 2026-07-13
**Status:** Approved design
**Scope:** Hybrid terrain path tracing, reusable image colour grading, and IBL environment decoding

## Problem

The Italy forest poster exposed three separate package concerns:

1. `hybrid_render_terrain_reference` hard-codes a warm-white directional sun in both its direct-light and ReSTIR setup, while the standard terrain API already accepts `sun_color`.
2. Physically correct environment lighting does not necessarily express the artistic mood associated with an HDR. For example, an up-facing surface can receive blue sky irradiance from a sunset environment whose warm colour is concentrated near the horizon.
3. `IBL.from_hdr` documents `.hdr` and `.exr` input, but its loader only decodes Radiance HDR.

Terrain raster formats are outside this bug: terrain reaches the path tracer as arrays, so GeoTIFF, DEM, PNG, and other source formats do not affect these lighting and colour mechanisms.

## Goals

- Give the hybrid terrain reference path the same explicit sun-colour control as other terrain paths.
- Provide an opt-in, renderer-independent way to derive and apply artistic environment mood without changing physical lighting.
- Make the documented `.hdr`/`.exr` IBL file contract true.
- Preserve current output byte-for-byte when callers use defaults.

## Non-goals

- Do not change the shader's physical environment sampling or terrain-raster loading.
- Do not infer a sun colour from an HDR.
- Do not apply a mood grade automatically.
- Do not add a new renderer configuration hierarchy, file-format dependency, or example-specific compositing path.

## Design

### 1. Hybrid terrain sun-colour parity

Add `sun_color` to the complete `hybrid_render_terrain_reference` call chain:

- Python wrapper and type stubs
- PyO3 function signature and validation
- `TerrainReferenceDesc`
- direct terrain-light uniform construction
- ReSTIR directional-light construction

The default is `(1.0, 0.97, 0.92)`, exactly matching today's two hard-coded values. The direct and ReSTIR paths both read `TerrainReferenceDesc.sun_color`; neither retains a duplicate literal.

The Python and native trust boundaries reject values that are not three finite, non-negative numbers. Values above one remain valid because colour and intensity are separate artistic/physical controls.

No WGSL binding or shader-layout change is required because both paths already receive a resolved light colour.

### 2. Opt-in environment mood grade

Add two NumPy utilities to `forge3d.colors`:

```python
environment_mood_tint(environment, *, horizon_fraction=10 / 64, max_gain=1.25)
apply_luminance_preserving_tint(image, tint, *, strength=0.0)
```

`environment_mood_tint` accepts a linear RGB environment array, averages the central equirectangular horizon band with a float64 accumulator, normalizes the result to Rec.709 luminance, and bounds each channel multiplier to `[1 / max_gain, max_gain]`. It deliberately does not use saturation or peak-radiance weighting, which gives the wrong sign for valid environments such as the brown studio HDR.

`apply_luminance_preserving_tint` accepts RGB or RGBA arrays. It blends the supplied tint with identity using `strength`, multiplies RGB, then restores the original Rec.709 luminance by adding the per-pixel luminance delta. Alpha is copied unchanged. A strength of zero returns values identical to the input. Callers grade a completed composition when map and legend colours must remain matched.

Both functions validate shapes, finite inputs, `0 <= strength <= 1`, `0 < horizon_fraction <= 1`, and `max_gain >= 1`. They preserve floating input dtype; integer input is calculated in float and rounded/clipped back to the original integer range.

This is a post-render artistic operation. It never modifies the environment used by the path tracer and never causes a retrace.

### 3. HDR and EXR environment decoding

Keep the compatible public name `IBL.from_hdr`, but make its internal loader dispatch by extension:

- `.hdr` / `.rgbe`: existing Radiance RGBE decoder
- `.exr`: existing optional `exr` crate, already enabled by the default `images` feature
- anything else: clear unsupported-format error

EXR decoding reads the first flat RGB layer into the existing `HdrImage { width, height, data }` representation. Missing RGB channels, invalid dimensions, and decode errors become contextual `RenderError`/Python `IOError` messages. Builds without the `images` feature continue to compile and return an explicit feature-required error for `.exr`.

No new dependency or public loader class is introduced.

## Data flow

Physical rendering remains:

```text
terrain arrays + environment array + sun direction/intensity/colour
    -> hybrid terrain path tracer
    -> linear rendered image
```

Optional mood treatment is separate:

```text
environment array -> horizon tint
rendered/composed RGB(A) + tint + nonzero strength
    -> luminance-preserving final RGB(A)
```

File-backed IBL remains:

```text
.hdr or .exr -> HdrImage -> existing IBL preparation
```

## Verification

Use the smallest tests that lock each contract:

- Source/API tests confirm `sun_color` exists in the wrapper and stubs, defaults to the legacy warm-white value, rejects malformed values, and feeds both direct and ReSTIR lights from one descriptor field.
- Existing default hybrid-terrain golden output must remain unchanged; one custom-colour render check confirms the control is live when GPU evidence is available.
- CPU NumPy tests confirm horizon extraction gives the expected warmth ordering for cached snow, Venice, and brown environments; strength zero is exact; luminance is preserved before output clipping; alpha is unchanged; and gains remain bounded.
- Rust decoder tests create minimal deterministic Radiance HDR and EXR inputs and compare dimensions and RGB values. A Python contract test confirms both extensions are accepted and unsupported extensions fail clearly.

## Compatibility and rollout

All new behaviour is opt-in. Existing callers that omit `sun_color`, do not call the mood utilities, and load Radiance HDR receive the current behavior. The implementation is delivered as three independently testable slices, but no slice changes terrain-raster APIs or existing example defaults.
