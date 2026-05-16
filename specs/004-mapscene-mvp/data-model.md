# Data Model: MapScene MVP

## MapScene

- `recipe`
- `last_validation_report`
- `compiled_label_plans`
- `render_policy`
- `reproducibility_profile`

Methods: `validate()`, `render(path)`, `save_bundle(path)`.

## SceneRecipe

- `terrain`
- `camera`
- `lighting`
- `layers`
- `output`
- `map_furniture`
- `render_policy`
- `diagnostics_policy`

## TerrainSource

- `path`
- `crs`
- `metadata`
- `elevation_sampling_available`

## RasterOverlay

- `layer_id`
- `path`
- `crs`
- `opacity`
- `metadata`

## VectorOverlay

- `layer_id`
- `path_or_features`
- `crs`
- `style`
- `style_support`

## LabelLayer

- `layer_id`
- `labels`
- `glyph_atlas`
- `typography`
- `priority_rules`
- `plan`

## PointCloudLayer

- `layer_id`
- `path`
- `crs`
- `point_count`
- `metadata`

## BuildingLayer

- `layer_id`
- `source`
- `support_level`
- `geometry_count`
- `bounds`
- `material_status`

## MapFurnitureLayer

- `title`
- `legend`
- `scale_bar`
- `north_arrow`
- `keepouts`

## Camera, Lighting, OutputSpec

Typed reproducibility settings for view, illumination, PNG size, format, and output intent.

## ReproducibilityProfile

- `seed`
- `camera`
- `output_size`
- `terrain_transform`
- `style_hashes`
- `asset_hashes_or_ids`
- `renderer_backend`
- `pixel_tolerance`: default exact unless recorded before verification.

## ReviewBundle

Deterministic scene recipe, diagnostics, compiled label plan plus source references, supported layer metadata, camera, lighting, output, and review metadata.
