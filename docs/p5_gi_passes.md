# P5 GI Passes: G-Buffer, HZB, AO/SSGI/SSR, and Composition

This page documents the P5 screen-space GI pipeline in forge3d:

- **G-Buffer + HZB** (P5.0)
- **SSAO/GTAO** (P5.1)
- **SSGI** (P5.2)
- **SSR** (P5.3)
- **GI composition + performance HUD** (P5.4–P5.8)

It is intended as a companion to the automated tests and golden artifacts
under `reports/p5/` and the Rust examples in `examples/`.

> All behavior is defined by `tests/test_p5_*.py` and the P5 examples.
> This document explains the intent, parameter ranges, and common
> troubleshooting paths (light leaks, speckle, banding).

## 1. Pipeline Overview

The P5 GI stack runs after direct lighting and before tone mapping:

```text
          +------------------------------+
          | Forward/Deferred Lighting    |
          |  (terrain, meshes, sky, etc) |
          +------------------------------+
                         |
                         v
                 G-Buffer + Depth
                         |
                         v
                 HZB (Hierarchical Z)
                         |
                         v
              Screen-Space GI Passes
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
     SSAO/GTAO         SSGI             SSR
  (diffuse occl.)  (diffuse bounce) (specular refl.)
        \               |                /
         \              |               /
          \             |              /
           v            v             v
            GI Composite (compute)
                         |
                         v
                 Tone Map / PostFX
```

The GI passes read from the shared G-Buffer and HZB and write into
separate AO/SSGI/SSR AOVs. The final compute kernel in
`src/shaders/gi/composite.wgsl` blends them into the lighting buffer
according to the physically-based rules from `todo-5.md`:

- **AO** multiplies diffuse only.
- **SSGI** adds to diffuse (clamped to an energy budget).
- **SSR** lerps specular according to roughness and Fresnel.

The `P5.4` example `p5_gi_stack_ablation.rs` captures all four stages in
`reports/p5/p5_gi_stack_ablation.png`.

## 2. G-Buffer and HZB

### 2.1 G-Buffer targets

The G-Buffer is produced by `src/passes/gbuffer.rs` using
`src/shaders/gbuffer/pack.wgsl` and `src/shaders/gbuffer/common.wgsl`.
The key targets are:

- **Normal**: `RG16F` view-space normals, oct-encoded.
- **Material**: `R8G8` roughness/metallic + optional albedo.
- **Depth**: default depth attachment (`Depth32Float`).

The helper `p5_dump_gbuffer.rs` exports sanity-check PNGs and metadata
under `reports/p5/`:

- `p5_gbuffer_normals.png`
- `p5_gbuffer_material.png`
- `p5_gbuffer_depth_mips.png`
- `p5_meta.json` (formats, sizes, shader hashes, depth convention).

### 2.2 HZB (Hierarchical Z)

`src/passes/hzb.rs` and `src/shaders/hzb_build.wgsl` build a min-depth
mip pyramid over the depth buffer. The HZB is used by SSGI ray march
and (optionally) SSR.

Key properties:

- **Mip 0** matches the full-resolution depth.
- **Mips 1–N** store min-depth over progressively larger tiles.
- **Reversed-Z detection** is derived from a center sample in
  `p5_dump_gbuffer.rs` and recorded in `p5_meta.json`.

When debugging SSGI/SSR misses or leaks, check:

- HZB mip 0..4 in `p5_gbuffer_depth_mips.png`.
- `reversed_z` flag and `depth_sample_center` in `p5_meta.json`.

## 3. SSAO / GTAO (P5.1)

The AO pass is implemented in:

- `src/passes/ssao.rs`
- `src/shaders/ssao/ssao.wgsl`
- `src/shaders/ssao/gtao.wgsl`
- `src/shaders/filters/bilateral_separable.wgsl`

It produces three internal AOVs:

- **Raw AO**: unfiltered hemisphere occlusion.
- **Blurred AO**: bilateral blur pass.
- **Resolved AO**: optional temporal accumulation.

CLI / viewer knobs (via `GiCliConfig` and `:ssao-*` commands):

| Parameter             | Range          | Effect                             |
|-----------------------|----------------|-------------------------------------|
| `ssao_radius`         | `>= 0.0`       | Sample radius in view-space units. |
| `ssao_intensity`      | `>= 0.0`       | AO strength multiplier.            |
| `ssao_technique`      | `ssao`/`gtao`  | Kernel type.                       |
| `ssao_samples`        | `>= 1`         | Hemisphere sample count.           |
| `ssao_directions`     | `>= 1`         | Azimuthal directions (GTAO).       |
| `ssao_bias`           | any `f32`      | Self-occlusion bias.               |
| `ssao_temporal_alpha` | `[0, 1]`       | History weight.                    |
| `ssao_composite`      | on/off         | Whether AO affects lighting.       |
| `ssao_mul`            | `[0, 1]`       | Scalar AO weight in composite.     |

**Before/after**: `reports/p5/p5_ssao_cornell.png` shows a Cornell
split with AO disabled vs enabled.

### 3.1 AO troubleshooting

- **Over-dark creases / banding**:
  - Reduce `ssao_intensity`.
  - Increase `ssao_radius` slightly and ensure enough samples.
  - Verify `ssao_bias` is not too small (self-shadowing).
- **Speckle / high-frequency noise**:
  - Increase `ssao_samples` and/or `ssao_directions`.
  - Ensure bilateral blur is enabled (`--ao-blur on`).
  - Enable temporal accumulation (`ssao_temporal_alpha` ~ `0.1–0.3`).
- **Light leaks across depth edges**:
  - Check normal/depth correctness via `p5_gbuffer_normals.png` and
    `p5_gbuffer_depth_mips.png`.
  - Tighten bilateral filter sigmas (in filters shader) if needed.

## 4. SSGI (P5.2)

SSGI runs at half resolution with an HZB-assisted ray march:

- `src/passes/ssgi.rs`
- `src/shaders/ssgi/trace.wgsl`
- `src/shaders/ssgi/shade.wgsl`
- `src/shaders/ssgi/resolve_temporal.wgsl`
- `src/shaders/filters/edge_aware_upsample.wgsl`

Parameters (CLI / viewer `:ssgi-*`):

| Parameter                  | Range        | Effect                                      |
|----------------------------|--------------|--------------------------------------------|
| `ssgi_steps`               | `>= 0`       | Ray-march steps (0 = IBL-only fallback).    |
| `ssgi_radius`              | `>= 0.0`     | Max ray distance in view space.            |
| `ssgi_half`                | on/off       | Half-res rendering toggle.                  |
| `ssgi_temporal_alpha`      | `[0, 1]`     | History weight.                             |
| `ssgi_temporal`            | on/off       | Enable/disable temporal accumulation.       |
| `ssgi_edges`               | on/off       | Edge-aware upsample vs box upsample.        |
| `ssgi_upsample_sigma_depth`| `> 0.0`      | Depth sigma for upsample weights.           |
| `ssgi_upsample_sigma_normal`| `> 0.0`     | Normal sigma (radians) for upsample.        |

**Before/after**:

- `reports/p5/p5_ssgi_cornell.png` – red/green wall bounce with SSGI on.
- `reports/p5/p5_ssgi_temporal_compare.png` – single-frame vs 16-frame
  accumulation.

### 4.1 SSGI troubleshooting

- **Light leaks / bleeding through walls**:
  - Check HZB correctness (`p5_gbuffer_depth_mips.png`).
  - Reduce `ssgi_radius`.
  - Consider fewer steps with a tighter cone.
- **Speckle / temporal flicker**:
  - Increase `ssgi_steps` moderately.
  - Use temporal accumulation with `ssgi_temporal_alpha` in `0.05–0.2`.
  - Ensure `ssgi_edges on` and reasonable upsample sigmas.
- **Banding along ray direction**:
  - Increase step count.
  - Jitter rays per frame using the GI seed if available.

## 5. SSR (P5.3)

SSR reflects specular lighting using a screen-space ray march with
optional thickness support:

- `src/passes/ssr.rs`
- `src/shaders/ssr/trace.wgsl`
- `src/shaders/ssr/shade.wgsl`
- `src/shaders/ssr/fallback_env.wgsl`

Parameters (via `SsrParams` and `:ssr-*` commands):

| Parameter        | Range           | Effect                                     |
|------------------|-----------------|-------------------------------------------|
| `ssr_enable`     | bool            | Master on/off for SSR.                    |
| `ssr_max_steps`  | `[1, 512]`      | Max march steps along reflection.         |
| `ssr_thickness`  | `[0.0, 1.0]`    | Thickness tolerance for hit acceptance.   |

**Before/after**:

- `reports/p5/p5_ssr_glossy_spheres.png` – highlight contrast across
  roughness.
- `reports/p5/p5_ssr_thickness_ablation.png` – undershoot artifacts vs
  thickness-corrected reflections.

### 5.1 SSR troubleshooting

- **Black holes / missing reflections**:
  - Ensure `ssr_enable` is true.
  - Increase `ssr_max_steps` gradually.
  - Relax `ssr_thickness` slightly (but keep within spec to avoid leaks).
- **Light leaks / reflections “through” geometry**:
  - Reduce `ssr_thickness`.
  - Double-check depth convention (reversed vs regular Z).
- **Noisy reflections on rough surfaces**:
  - Verify roughness remapping in `ssr/shade.wgsl` (cone filtering).
  - Let SSGI handle low-frequency indirect light; SSR should focus on
    sharper features.

## 6. GI Composition and P5.4 Stack

The composition pass lives in:

- `src/passes/gi.rs`
- `src/shaders/gi/composite.wgsl`

It blends the baseline lighting, AO, SSGI, and SSR using
`GiCompositeParams` and the rules from P5.4.

The example `examples/p5_gi_stack_ablation.rs` uses the viewer to
produce `reports/p5/p5_gi_stack_ablation.png` with four columns:

1. Baseline (no GI)
2. +AO
3. +AO+SSGI
4. +AO+SSGI+SSR

This image is the primary visual reference for verifying GI ordering and
energy behavior.

## 7. Performance HUD (P5.6 + P5.8)

The interactive viewer samples both **CPU-side timings** and optional
**GPU timestamp queries** for each P5 pass:

- HZB build
- SSAO/GTAO
- SSGI (trace / shade / temporal / upsample)
- SSR (trace / shade / fallback)
- GI composite

When GI is active, the HUD overlay in the viewer renders small numeric
indicators (milliseconds) for each component. The `gi_showcase` example
starts with AO, SSGI, and SSR enabled so you can immediately see:

- Per-pass timings vs the budgets from `todo-5.md`.
- The impact of changing `--ssao-*`, `--ssgi-*`, and `--ssr-*` flags.

P5.6 budgets (RTX 3060 @ 1080p) are summarized in `p5_meta.json` under
`gi_composition.perf_budgets`:

- HZB: `<= 0.5 ms`
- SSAO/GTAO: `<= 1.6 ms`
- SSGI (half-res): `<= 2.8 ms`
- SSR (HZB): `<= 2.2 ms`
- Bilateral + temporal + composite: `<= 1.2 ms`

If any budget is exceeded, `p5_meta.json` records a `p56_status` of
`"REGRESSION"`.

## 8. GI Showcase Example (P5.8)

`examples/gi_showcase.rs` provides a minimal Rust entrypoint that:

- Parses GI-related CLI flags via `GiCliConfig`.
- Seeds the interactive viewer with `:gi` / `:ssao-*` / `:ssgi-*` /
  `:ssr-*` commands.
- Enables SSAO, SSGI, and SSR by default when no explicit GI flags are
  given.
- Selects a default IBL (`assets/snow_field_4k.hdr`) and lit view mode.
- Relies on the viewer’s on-screen HUD for P5 performance readouts.

Example usage:

```bash
# Default GI showcase (AO + SSGI + SSR)
cargo run --release --example gi_showcase -- \
  --ibl assets/snow_field_4k.hdr

# Tweak AO and SSGI radius/steps
cargo run --release --example gi_showcase -- \
  --gi ssao:on --gi ssgi:on --gi ssr:on \
  --ssao-radius 0.6 --ssao-intensity 1.2 \
  --ssgi-steps 16 --ssgi-radius 0.9

# Turn off SSR to inspect diffuse-only GI
cargo run --release --example gi_showcase -- \
  --gi ssao:on --gi ssgi:on --gi ssr:off
```

The example is cross-platform and relies only on the main forge3d
viewer stack, so it builds and runs on **Windows**, **macOS**, and
**Linux** with a compatible GPU and Rust toolchain.
