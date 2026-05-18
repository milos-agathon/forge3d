# Building Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Native GeoJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Native CityJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Python fallback geometry | `placeholder/fallback` | Current fallback paths can create zero-geometry building records. | `placeholder_fallback`. |
| `MapSceneBuildingLayer` supported source intent | `underdeveloped` | Typed MapScene recipes record building source identity, support level, geometry count, bounds, and material status. | `missing_external_asset`, `unsupported_asset_format`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, or `unsupported_feature`. |
| Scalar PBR materials | `underdeveloped` | Scalar material values exist and MapScene can preserve material status, but production building material rendering is not fully integrated. | Support matrix entry plus validation diagnostics. |
| Textured PBR buildings | `missing` | End-to-end textured building workflow is not implemented here. | Future material/texture diagnostics. |

Building workflows must not treat zero-geometry fallback output as successful
ingestion.
