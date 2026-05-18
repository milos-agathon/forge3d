# Data Model: Material, VT, and Large-Scene P2

## VirtualTextureFamilyRequest

- `family`: albedo, normal, mask.
- `source`
- `support_level`
- `diagnostics`

## VTFamilySupportReport

- `requested_families`
- `supported_families`
- `unsupported_families`
- `runtime_status`
- `diagnostics`

## TexturedBuildingMaterial

- `material_id`
- `albedo_texture`
- `uv_available`
- `texture_format`
- `scalar_fallback`
- `support_level`
- `diagnostics`

## BuildingTextureDiagnostic

Feature-local diagnostic for `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, Pro-gated path, or explicit fallback.

## AdvancedLabelRuleSet

- `repeat_distance`
- `curved_text`
- `road_rules`
- `river_rules`
- `leader_lines`
- `priority_presets`
- `shaping_policy`

## AdvancedLabelPlanResult

Deterministic accepted/rejected label output for advanced modes plus diagnostics and reason codes.

## LargeSceneResourceSummary

- `memory_estimates`
- `cache_stats`
- `lod_stats`
- `instancing_status`
- `bottleneck_layer_types`
- `unavailable_stats`
- `diagnostics`
