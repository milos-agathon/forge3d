# Building Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Native GeoJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Native CityJSON building import | `Pro-gated` | Available only when the native/Pro path is present. | `pro_gated_path` when unavailable. |
| Python fallback geometry | `placeholder/fallback` | Fallback paths can still create zero-geometry building records when native rendering is unavailable. | `placeholder_fallback`. |
| `MapSceneBuildingLayer` supported source intent | `supported` | Inline typed MapScene building features with local polygon geometry render as terrain scatter instanced meshes in the GPU terrain path, preserve source identity, geometry count, bounds, roof type, material status, per-building batch ids, and review-bundle metadata. External asset imports still use the diagnostics in this row when unavailable. | `missing_external_asset`, `unsupported_asset_format`, `pro_gated_path`, `placeholder_fallback`, `experimental_feature`, or `unsupported_feature` for unsupported sources. |
| Scalar PBR materials | `supported` | Scalar material colors feed terrain scatter instanced building meshes for the MapScene scalar tier; this is not textured facade support. | Textured material requests still use texture/UV diagnostics. |
| Flat/gabled/hipped/pyramidal roof geometry | `supported` | Roofs are generated as native mesh geometry. Ridge placement is bounding-box based, so rectangular footprints are the supported high-confidence case; irregular footprints keep their footprint for walls but use approximate ridge placement. | Future footprint-aware ridge generation must add explicit tests before claiming exact irregular-roof placement. |
| Building CSM shadows | `supported` | In the non-offline GPU terrain path, MapScene submits building meshes through `TerrainRenderer.set_scatter_batches()`, renders those meshes into terrain cascaded shadow-map passes, and samples cascades in mesh shading (`building_shadow_model="terrain_csm_mesh_cast_receive"`). The projected native compositor remains the fallback for paths that cannot submit terrain scatter, including offline accumulation. | Offline accumulation still uses the projected compositor until the HDR/offline scatter pass is added. |
| Building batch IDs | `supported` | Render metadata preserves per-feature native mesh batch IDs as `building_batch_ids` for review/debugging. These are not picking IDs and no MapScene building pick map is claimed. | Use viewer/vector picking APIs for interactive picking until a building pick-map contract exists. |
| Textured PBR buildings | `unsupported` | `MapScene` records albedo texture path intent, UV coordinates prerequisites, texture format, and affected material/object IDs, but does not render textured PBR buildings end to end. | `missing_texture_path`, `missing_uvs`, `unsupported_texture_format`, `placeholder_fallback`, `pro_gated_path`. |

Building workflows must not treat zero-geometry fallback output as successful
ingestion.

Textured building material intent requires an albedo texture path, readable local
asset metadata, supported image suffix, and UV coordinates for the affected
geometry. scalar fallback is not textured PBR support; it must be reported as a
`placeholder/fallback` diagnostic and diagnosed before render. This
non-MVP-blocking textured-material deferral keeps the current offline map workflow
honest while avoiding claims of material parity with DCC or game engines.
