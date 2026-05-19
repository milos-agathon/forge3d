# Building Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Native GeoJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Native CityJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Python fallback geometry | `placeholder/fallback` | Current fallback paths can create zero-geometry building records. | `placeholder_fallback`. |
| `MapSceneBuildingLayer` supported source intent | `underdeveloped` | Typed MapScene recipes record building source identity, support level, geometry count, bounds, and material status. | `missing_external_asset`, `unsupported_asset_format`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, or `unsupported_feature`. |
| Scalar PBR materials | `underdeveloped` | Scalar material values exist and MapScene can preserve material status, but production building material rendering is not fully integrated. | Support matrix entry plus validation diagnostics. |
| Textured PBR buildings | `unsupported` | `MapScene` records albedo texture path intent, UV coordinates prerequisites, texture format, and affected material/object IDs, but does not render textured PBR buildings end to end. | `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `placeholder_fallback`, `pro_gated_path`. |

Building workflows must not treat zero-geometry fallback output as successful
ingestion.

Textured building material intent requires an albedo texture path, readable local
asset metadata, supported image suffix, and UV coordinates for the affected
geometry. scalar fallback is not textured PBR support; it must be reported as a
`placeholder/fallback` diagnostic and diagnosed before render. This
non-MVP-blocking textured-material deferral keeps the current offline map workflow
honest while avoiding claims of material parity with DCC or game engines.
