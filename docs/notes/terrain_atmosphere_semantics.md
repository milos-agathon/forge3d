# Terrain Atmosphere Semantics

This note is the authoritative contract for terrain fog and sky behavior.

## Ownership

- `FogSettings` owns height fog only: density, height falloff, base height, and inscatter tint.
- `SkySettings` owns aerial perspective for terrain: turbidity, aerial density, sun intensity, sun size, and sky exposure.
- The terrain renderer does not interpret a separate fog-side aerial-perspective control.

## Runtime Behavior

- When `sky.enabled = false`, terrain uses the fog inscatter tint directly and the sky path contributes nothing.
- When `sky.enabled = true`, terrain samples the sky atmosphere output for distant tint and aerial perspective.
- Height fog remains enabled independently of sky; sky settings only affect the atmosphere tint and distance haze.

## Compatibility

- The legacy `FogSettings.aerial_perspective` terrain knob was removed in TV1.2.
- Existing terrain code should move haze tuning to `SkySettings.aerial_perspective` and `SkySettings.aerial_density`.
- If a scene needs plain fog only, configure `FogSettings` without enabling sky.
