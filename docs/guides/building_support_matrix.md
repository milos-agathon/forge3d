# Building Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Native GeoJSON building import | `Pro-gated` | `MapSceneBuildingLayer` reports native/Pro asset paths when unavailable. | `pro_gated_path` and `missing_external_asset` when unavailable. |
| Native CityJSON building import | `Pro-gated` | `MapSceneBuildingLayer` preserves typed intent for native/Pro import. | `pro_gated_path` and `unsupported_asset_format` when unavailable. |
| Python fallback geometry | `placeholder/fallback` | Current fallback paths can create zero-geometry building records. | `placeholder_fallback` and `missing_renderable_data`. |
| Scalar PBR materials | `underdeveloped` | Scalar material values exist on `MapSceneBuildingLayer`; validation remains diagnostic-bearing. | `unsupported_feature` for unsupported material paths. |
| Textured PBR buildings | `missing` | End-to-end textured building workflow is not implemented here. | Future material/texture diagnostics. |

Building workflows must not treat zero-geometry fallback output as successful
ingestion.
