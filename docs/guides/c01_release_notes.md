# C01 release notes (unreleased)

Milos accepted these compatibility decisions on 2026-10-16.

## Building output

Buildings use native 3D geometry at every MapScene quality setting. Previously, output with more than one sample or HDR used a display composite after tonemapping. Those scenes now have the geometry's consistent size and silhouette, and HDR carries real building radiance. Their appearance can change. Denoising preserves accumulated building samples and excludes their sampling footprint from terrain neighbourhoods.

## Scaled frame edges

Direct terrain renders at a resolution scale other than 1 clamp beauty/AOV frame edges. The top/bottom and left/right edges no longer sample each other's colour. This is a framebuffer correction; material texture tiling is unchanged. The standard unscaled recipe images retain their output.

## Deprecated address fields

`SamplingSettings.address_u`, `address_v` and `address_w` remain accepted and serialized unchanged, but have no effect. Framebuffer resolves always clamp. Their historical default is `Repeat` on every axis. Choosing a non-default `ClampToEdge` or `MirrorRepeat` emits `DeprecationWarning`, both on construction and later assignment. Remove non-default overrides. The direct `SamplingSettings` constructor still requires address arguments; use `Repeat` there until the fields are removed in a future release. They have not been rewired to material textures.

## MapScene recipe sampling

MapScene keeps native MSAA at 1 and resolution scale at 1. `lighting.settings["cli_params"]["msaa_samples"]` and `["render_scale"]` are ignored with `UserWarning` whenever present, including explicit default values. The recipe retains the keys and values; the warning does not rewrite saved recipes. Removing these ignored keys gives the same pixels. Native provenance works under the existing one-shot restrictions; offline/HDR provenance restrictions are unchanged.

`output.samples` still controls offline accumulation. To choose a native MSAA setting or resolution scale, use `TerrainRenderParams` and the direct renderer. Native offline accumulation itself forces MSAA 1; requested 1/4 combinations are compatibility controls. Recipe-controlled sampling is deferred scope, requiring measured quality benefit and provenance support before introduction.

The accepted building-output and frame-edge changes are settled. C01 remains partial for Milos's assessment of the prepared renders, F6 hardware-counter and exclusive-host qualification, and integration. Other adapters and backends remain unqualified. See [current validation](c01_material_validation.md) and [the changelog](../../CHANGELOG.md).
