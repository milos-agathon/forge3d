# General HDR Terrain Lighting and Mood Design

**Date:** 2026-07-13
**Status:** Revised after readiness review
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

- Python wrapper in `python/forge3d/path_tracing.py`
- both type stubs: `python/forge3d/path_tracing.pyi` and the native re-export in `python/forge3d/__init__.pyi`
- PyO3 function signature and validation
- `TerrainReferenceDesc`
- direct terrain-light uniform construction
- ReSTIR directional-light construction

The standard terrain API reference is `make_terrain_params_config` in `python/forge3d/terrain_params.py`. It treats `sun_color=None` as `[1.0, 1.0, 1.0]`; the hybrid path instead keeps its existing hard warm-white default `(1.0, 0.97, 0.92)` to preserve byte-identical default output. Parity means both APIs expose an explicit colour control, not that their defaults change.

Let `I = TerrainReferenceDesc.sun_intensity` and `c = TerrainReferenceDesc.sun_color`. The two existing factoring conventions must remain asymmetric:

- direct-light uniform: `light_color = [I * c[0], I * c[1], I * c[2]]`
- ReSTIR directional light: pass `I.max(1e-6)` as intensity and `[c[0], c[1], c[2]]` unchanged as colour

Both sites read the descriptor field and retain no warm-white literal. Multiplying the ReSTIR colour by intensity would apply intensity twice; omitting the direct multiply would drop it there.

The Python wrapper and native PyO3 boundary both reject anything other than exactly three finite, non-negative numbers with `ValueError`. `(0.0, 0.0, 0.0)` is intentionally valid and disables the sun through colour even though the ReSTIR intensity is clamped to `1e-6`. Values above one remain valid because colour and intensity are separate artistic/physical controls. Existing `sun_intensity` validation is unchanged.

This is a new keyword argument, not a new exported symbol: no `__all__`, `EXPECTED_FUNCTIONS`, or `m.add_function` change is required.

No WGSL binding or shader-layout change is required because both paths already receive a resolved light colour.

### 2. Opt-in environment mood grade

Add two NumPy utilities to `forge3d.colors`:

```python
environment_mood_tint(environment, *, horizon_fraction=10 / 64, max_gain=1.25)
apply_luminance_preserving_tint(image, tint, *, strength=0.0)
```

`environment_mood_tint` accepts a linear RGB environment array with height `H`. Its horizon rows are fixed as follows:

```python
band_height = max(1, min(H, round(horizon_fraction * H)))
start = (H - band_height) // 2
stop = start + band_height
band = environment[start:stop, :, :3]
```

`round` is Python's built-in ties-to-even rounding and `stop` is exclusive. Average the band over both spatial axes with a float64 accumulator to obtain `mean_rgb`. With Rec.709 weights `w = [0.2126, 0.7152, 0.0722]`, compute `L = dot(mean_rgb, w)`. If `L <= 1e-12`, return identity tint `[1.0, 1.0, 1.0]`; otherwise compute `mean_rgb / L` and clamp each multiplier to `[1 / max_gain, max_gain]`. This division is the luminance normalization. The utility deliberately does not use saturation or peak-radiance weighting, which gives the wrong sign for valid brown environments.

`apply_luminance_preserving_tint` accepts RGB or RGBA arrays. It blends the supplied tint with identity using `strength`, multiplies RGB, then restores the original Rec.709 luminance by adding the per-pixel luminance delta. Alpha is copied unchanged. A strength of zero returns values identical to the input. Callers grade a completed composition when map and legend colours must remain matched.

Both functions validate shapes, finite inputs, `0 <= strength <= 1`, `0 < horizon_fraction <= 1`, and `max_gain >= 1`. They preserve floating input dtype; integer input is calculated in float and rounded/clipped back to the original integer range.

Mood-ordering tests use small synthetic linear-RGB environment arrays with controlled cool, warm, and brown horizon rows. They do not depend on git-ignored HDR files, local caches, or machine-specific paths.

This is a post-render artistic operation. It never modifies the environment used by the path tracer and never causes a retrace.

### 3. HDR and EXR environment decoding

Keep the compatible public name `IBL.from_hdr`, but make its internal loader dispatch by extension:

- `.hdr` / `.rgbe`: existing Radiance RGBE decoder
- `.exr`: existing optional `exr` crate, already enabled by the default `images` feature
- anything else: clear unsupported-format error

EXR decoding reads the first flat layer and maps channels by the exact OpenEXR channel names `R`, `G`, and `B`, independent of their storage order; other channels are ignored. All three channels are required. `FlatSamples::F16` values use `to_f32()`, `FlatSamples::F32` values are copied, and `FlatSamples::U32` values use numeric `as f32` conversion. The three converted channels are interleaved in row-major RGB order into the existing `HdrImage { width, height, data }` representation.

Missing RGB channels, invalid dimensions, unsupported sample data, and decode errors become contextual `RenderError` values in the internal loader. `IBL.from_hdr` must continue to wrap loader failures manually with `PyIOError`, as it does today; using `?` through the generic `From<RenderError> for PyErr` mapping would incorrectly expose I/O failures as `PyRuntimeError`. Builds without the `images` feature continue to compile and return an explicit feature-required error for `.exr`.

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

- Source/API tests confirm `sun_color` exists in the wrapper and both stubs, defaults to the legacy warm-white value, and rejects malformed values with `ValueError` while accepting `(0.0, 0.0, 0.0)`.
- A direct Rust assertion independent of image goldens locks the factoring contract: direct colour equals `I * sun_color`, while ReSTIR colour equals raw `sun_color` and intensity is passed separately. The existing default hybrid-terrain golden remains supplemental evidence because it is not committed and can self-generate.
- One custom-colour render check confirms the control is live when GPU evidence is available.
- CPU NumPy tests use synthetic environments to lock the exact horizon rows, identity fallback for a near-black band, cool/warm/brown ordering, strength-zero identity, luminance preservation before output clipping, unchanged alpha, and bounded gains. A separate integer-dtype check locks rounding, clipping, and return dtype.
- Rust decoder tests create minimal deterministic Radiance HDR and EXR inputs, including F16 and U32 EXR samples with shuffled channel storage order, and compare dimensions and interleaved f32 RGB values.
- A Python contract test confirms `.hdr`, `.rgbe`, and `.exr` are accepted and unsupported extensions fail with `IOError`. A focused `cargo test --no-default-features` check locks the explicit feature-required `.exr` error; the normal CI matrix always enables `images` and cannot cover this branch.

## Scope consequence

These three slices do not change the Italy forest poster. That example renders through `open_viewer_async`, not `hybrid_render_terrain_reference`, `forge3d.colors`, or `IBL.from_hdr`. Making the poster visibly warmer still requires a separate caller-side change that opts into the mood grade for the completed overlay and legend composition; that example wiring remains a non-goal here.

## Compatibility and rollout

All new behaviour is opt-in. Existing callers that omit `sun_color`, do not call the mood utilities, and load Radiance HDR receive the current behavior. The implementation is delivered as three independently testable slices, but no slice changes terrain-raster APIs or existing example defaults.
