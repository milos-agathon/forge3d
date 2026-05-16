# Building Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Native GeoJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Native CityJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Python fallback geometry | `placeholder/fallback` | Current fallback paths can create zero-geometry building records. | `placeholder_fallback`. |
| Scalar PBR materials | `underdeveloped` | Scalar material values exist; typed MapScene workflow is later work. | Support matrix entry plus later validation diagnostics. |
| Textured PBR buildings | `missing` | End-to-end textured building workflow is not implemented here. | Future material/texture diagnostics. |

Building workflows must not treat zero-geometry fallback output as successful
ingestion.
