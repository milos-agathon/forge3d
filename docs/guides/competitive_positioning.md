# Competitive Positioning

forge3d is an offline, Python-native 3D map-production engine. It should not be
described as a replacement for every capability in Blender, Mapbox GL, Cesium,
three.js, or Unreal.

| Comparison pressure | forge3d position | Support level |
| --- | --- | --- |
| Offline map plates and reproducible terrain renders | Product focus, with exact workflow status documented per API and guide. | `underdeveloped` |
| Streamed browser map delivery | Out of scope for the MVP. | `non-goal` |
| Complete Mapbox style parity | Out of scope for P0; local feature styling has its own limited support matrix. | `unsupported` |
| Cesium-grade global runtime parity | Out of scope for P0/P1. | `non-goal` |
| General DCC or game-editor workflows | Outside the product boundary. | `non-goal` |
| Live globe streaming | Offline validation and rendering only; browser globe hosting is outside the product boundary. | `non-goal` |
| Hosted tile-provider parity | Local fixtures and review bundles are supported where documented; hosted provider parity is not a goal. | `non-goal` |
| Blender parity | forge3d is not a general DCC package. | `non-goal` |
| Unreal parity | forge3d is not a game/editor runtime. | `non-goal` |
| Pro/native-only geospatial import paths | Must be called out honestly. | `Pro-gated` |
| Textured PBR buildings | End-to-end textured city material rendering is diagnosed before render, not treated as scalar fallback success. | `unsupported` |
| VT normal/mask runtime | Native runtime pages albedo only; normal and mask families must diagnose before render. | `unsupported` |
| Advanced labels and shaping | Deterministic product paths are used where available; incomplete shaping or curved rendering paths stay diagnostic-bearing. | `experimental` |
| Large-scene diagnostics | Memory, cache/LOD, instancing, and bottleneck summaries are offline diagnostics, not hosted streaming capability. | `underdeveloped` |

Docs and diagnostics should use the exact support classification instead of
umbrella wording.
The term `missing` is reserved for scoped map features with no product path or
diagnostic-bearing substrate.
