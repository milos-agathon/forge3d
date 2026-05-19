# Large-Scene Support

forge3d large-scene work is scoped to offline map-production diagnostics for
`MapScene.validate`, not live globe streaming or hosted tile delivery.

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Memory budget estimates | `supported` | Validation estimates known output, terrain/raster dimensions, point counts, building geometry counts, and tile metadata where available. | `estimated_gpu_memory` when the configured budget is exceeded. |
| Cache/LOD stat availability | `underdeveloped` | Terrain, point cloud, building, and tile layers can preserve cache/LOD metadata when supplied; unavailable stats are reported instead of invented. | `unavailable_cache_lod_stats`. |
| Instancing status | `unsupported` | MapScene records requested instancing workflows and blocks unsupported product paths before render. | `unsupported_instancing_path`. |
| Bottleneck layer types | `underdeveloped` | Validation can summarize bottleneck layer types from known memory, count, cache, LOD, or timing metadata with deterministic ordering. | Summary data in validation reports. |
| Live globe streaming | `non-goal` | forge3d does not host or stream a browser globe. | Documentation boundary. |
| Hosted tile-provider parity | `non-goal` | Hosted Mapbox/Cesium-style provider ecosystems are outside this offline workflow. | Documentation boundary. |
| Blender parity | `non-goal` | General DCC scene authoring is outside the map-production contract. | Documentation boundary. |
| Unreal parity | `non-goal` | Game/editor runtime behavior is outside the map-production contract. | Documentation boundary. |
| General DCC | `non-goal` | Non-map rendering and cinematic/general content tooling are not the P2 goal. | Documentation boundary. |

Large-scene P2 gaps are non-MVP-blocking when they are diagnosed before render.
Validation must preserve known memory budget estimates, cache/LOD stat
availability, instancing status, and bottleneck layer types without claiming
hosted streaming capability or parity with Blender, Unreal, Cesium, Mapbox GL, or
browser-hosted engines.
The phrase cache/LOD stat availability is the product boundary: known stats are
reported, unknown stats are diagnosed, and precision is not invented.

The PRD classification `missing` remains available for capabilities with no
product feature at all; the Milestone 5 items in this guide prefer diagnosed
`unsupported`, `underdeveloped`, or `experimental` status where substrate exists.
