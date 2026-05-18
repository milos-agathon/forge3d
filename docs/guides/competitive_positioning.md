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
| Pro/native-only geospatial import paths | Must be called out honestly. | `Pro-gated` |
| Textured PBR buildings | End-to-end textured city material rendering is not implemented in the MVP. | `missing` |
| VT normal/mask runtime | Native runtime pages albedo only; normal and mask families must diagnose before render. | `missing` |

Docs and diagnostics should use the exact support classification instead of
umbrella wording.
