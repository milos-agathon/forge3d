# 3D Tiles Support Matrix

| Capability | Support level | Scope | Diagnostics |
| --- | --- | --- | --- |
| Rust B3DM and tileset substrate | `underdeveloped` | Parser and renderer substrate exists, but public Python MapScene workflow is not complete. | `python_public_3dtiles_incomplete`. |
| Public Python scene integration | `underdeveloped` | `MapScene` preserves typed 3D Tiles intent, but render preparation remains incomplete. | `python_public_3dtiles_incomplete`. |
| Unsupported tile features | `unsupported` | Unsupported B3DM/GLB features must be diagnosed before render. | Future tile-format diagnostics. |
| Large production tile hierarchies | `non-goal` | This feature does not claim Cesium runtime parity. | Documentation boundary. |

3D Tiles support must distinguish Rust infrastructure from a public, renderable
Python product workflow.
