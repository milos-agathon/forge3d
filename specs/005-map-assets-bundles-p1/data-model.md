# Data Model: Map Assets and Bundle Round-Trip P1

## LabelLayer

- `layer_id`
- `features`
- `geometry_type`
- `properties`
- `crs`
- `text_expression`
- `terrain_sampling_policy`
- `typography`
- `diagnostics`

## LabelTextExpression

Supported forms: `{name}`, `get`, `concat`, `coalesce`, casing transforms. Missing fields produce `missing_label_field`.

## FontAtlas

- `atlas_id`
- `source_font`
- `coverage_ranges`
- `generation_status`
- `diagnostics`

## FontFallbackRange

- `range_start`
- `range_end`
- `font_or_atlas`
- `priority`

## TypographySettings

Kerning, tracking, line-height, multiline, callout behavior, and support status.

## BuildingLayer

- `layer_id`
- `source`
- `source_format`
- `support_level`
- `geometry_count`
- `bounds`
- `material_status`
- `diagnostics`

## Tiles3DLayer

- `layer_id`
- `tileset`
- `supported_format_status`
- `cache_stats`
- `lod_config`
- `diagnostics`

## BundleManifest

Versioned deterministic manifest containing terrain metadata, layers, camera, lighting, output spec, source labels, compiled label plan payloads where available, diagnostics, supported export settings, external asset references, and review metadata.

## MissingAssetDiagnostic

Diagnostic for bundle references unavailable at load or validation time.
