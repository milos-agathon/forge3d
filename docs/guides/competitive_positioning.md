# Competitive Positioning

forge3d is an offline, Python-native 3D map-production engine. It should not be
described as a replacement for every capability in Blender, Mapbox GL, Cesium,
three.js, or Unreal.

| Comparison pressure | forge3d position | Support level wording |
| --- | --- | --- |
| Offline map plates and reproducible terrain renders | Product focus. | `supported` |
| Streamed browser map delivery | Out of scope for the MVP. | `non-goal` |
| Complete Mapbox style parity | Out of scope for P0. | `unsupported` |
| Cesium-grade global runtime parity | Out of scope for P0/P1. | `non-goal` |
| General DCC or game-editor workflows | Outside the product boundary. | `non-goal` |
| Textured PBR buildings | Texture-backed building material import is not implemented end to end. | `missing` |
| VT normal/mask runtime | Runtime virtual texturing pages albedo only. | `missing` |
| Pro/native-only geospatial import paths | Must be called out honestly. | `Pro-gated` |

Docs and diagnostics should use the exact support classification instead of
umbrella wording.
