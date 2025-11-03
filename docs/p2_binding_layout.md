# P2 BRDF Module: Binding Layout Documentation

This document enumerates the group/binding indices for the P2 BRDF module to avoid binding collisions across pipelines.

## Mesh PBR Pipeline (`src/shaders/pbr.wgsl`)

### @group(0) - Per-draw Uniforms
- `@binding(0)`: `uniform<Uniforms>` - Model, view, projection matrices
- `@binding(1)`: `uniform<PbrLighting>` - Light direction, color, intensity, camera position, IBL settings, exposure, gamma
- `@binding(2)`: `uniform<ShadingParamsGPU>` - BRDF dispatch parameters (brdf model index, metallic, roughness, sheen, clearcoat, subsurface, anisotropy)

### @group(1) - Material Textures/Samplers
- `@binding(0)`: `uniform<PbrMaterial>` - Base color, metallic, roughness, normal scale, occlusion strength, emissive, alpha cutoff, texture flags
- `@binding(1)`: `texture_2d<f32>` - Base color texture
- `@binding(2)`: `texture_2d<f32>` - Metallic-roughness texture (B=metallic, G=roughness)
- `@binding(3)`: `texture_2d<f32>` - Normal map texture
- `@binding(4)`: `texture_2d<f32>` - Ambient occlusion texture
- `@binding(5)`: `texture_2d<f32>` - Emissive texture
- `@binding(6)`: `sampler` - Material sampler

### @group(2) - IBL Textures (Optional)
- `@binding(0)`: `texture_2d<f32>` - Irradiance texture (equirectangular)
- `@binding(1)`: `sampler` - Irradiance sampler
- `@binding(2)`: `texture_2d<f32>` - Prefiltered environment map
- `@binding(3)`: `sampler` - Prefilter sampler
- `@binding(4)`: `texture_2d<f32>` - BRDF LUT texture
- `@binding(5)`: `sampler` - BRDF LUT sampler

**Note**: The PBR pipeline does NOT conflict with lights defined in `lighting.wgsl` (@group(2) bindings 0-5 for IBL env/irradiance/sampler/lut) since lights buffer uses @group(1) bindings in the lighting module.

## Terrain PBR Pipeline (`src/shaders/terrain_pbr_pom.wgsl`)

### @group(0) - Terrain Uniforms
- `@binding(0)`: `uniform<TerrainUniforms>` - View, projection, sun exposure, spacing, height exaggeration
- `@binding(1)`: `texture_2d<f32>` - Height texture
- `@binding(2)`: `sampler` - Height sampler
- `@binding(3)`: `texture_2d_array<f32>` - Material albedo texture array
- `@binding(4)`: `sampler` - Material sampler
- `@binding(5)`: `uniform<TerrainShadingUniforms>` - Triplanar params, POM steps, layer heights/roughness/metallic, light params, clamps
- `@binding(6)`: `texture_2d<f32>` - Colormap texture
- `@binding(7)`: `sampler` - Colormap sampler
- `@binding(8)`: `uniform<OverlayUniforms>` - Overlay domain, blend mode, colormap strength, gamma

### @group(1) - Light Buffer (P1-06)
- `@binding(0)`: `storage<array<Light>>` - Light array
- `@binding(1)`: `uniform<LightCount>` - Active light count

**Note**: Terrain shader has its own `TerrainShadingUniforms` at @group(0) @binding(5) for terrain-specific knobs. P2-05 (optional) will add a bridging mechanism to map a subset to `ShadingParamsGPU` for BRDF evaluation when enabled via feature gate.

## Lighting Module (`src/shaders/lighting.wgsl`)

The centralized lighting module defines:
- `ShadingParamsGPU` struct (used in @group(0) @binding(2) for mesh PBR)
- BRDF model constants (BRDF_LAMBERT, BRDF_PHONG, BRDF_COOK_TORRANCE_GGX, etc.)
- Shadow and GI settings structs
- IBL environment textures (@group(2) bindings 0-3 for cube maps in unified lighting)

### @group(2) - IBL Environment (Unified Lighting)
- `@binding(0)`: `texture_cube<f32>` - IBL specular environment
- `@binding(1)`: `texture_cube<f32>` - IBL irradiance cube
- `@binding(2)`: `sampler` - IBL environment sampler
- `@binding(3)`: `texture_2d<f32>` - IBL BRDF LUT

**Important**: The mesh PBR pipeline uses equirectangular textures in @group(2), while the unified lighting module uses cube maps. These are separate pipelines and do not conflict.

## Binding Collision Prevention

1. **Mesh PBR vs Lights**: Mesh PBR uses @group(0-2), lights use @group(1) in their own pipeline. No conflicts.
2. **Mesh PBR vs IBL**: IBL textures are in @group(2) with distinct bindings per pipeline.
3. **Terrain vs Lights**: Terrain uses @group(0) for uniforms/textures, @group(1) for light buffer. No conflicts.
4. **ShadingParamsGPU**: Defined once in `lighting.wgsl`, imported by `pbr.wgsl`. No duplicate definitions.

## BRDF Module Structure

All BRDF implementations are in `src/shaders/brdf/`:
- `common.wgsl` - Shared math (Fresnel, GGX/Beckmann NDFs, geometry terms, orthonormal basis)
- `lambert.wgsl` - Lambertian diffuse
- `phong.wgsl` - Blinn-Phong specular
- `oren_nayar.wgsl` - Rough diffuse
- `cook_torrance.wgsl` - Microfacet GGX and Beckmann
- `disney_principled.wgsl` - Disney BRDF subset
- `ashikhmin_shirley.wgsl` - Anisotropic BRDF
- `ward.wgsl` - Ward anisotropic
- `toon.wgsl` - Cel-shaded stylized
- `minnaert.wgsl` - Dark-backscattering diffuse
- `dispatch.wgsl` - Centralized `eval_brdf()` function with switch dispatch

The `dispatch.wgsl` file is included by `lighting.wgsl` after defining `ShadingParamsGPU` and BRDF constants, ensuring all symbols are available.

## Exit Criteria (P2-01 & P2-02)

✅ All BRDF module files exist and implement required functions
✅ `ShadingParamsGPU` defined centrally in `lighting.wgsl`
✅ BRDF constants aligned with Rust `BrdfModel` enum in `src/lighting/types.rs`
✅ `dispatch.wgsl` includes all model files and implements `eval_brdf()`
✅ `pbr.wgsl` imports `lighting.wgsl` to access BRDF definitions (no duplicate symbols)
✅ Binding layouts documented per pipeline
✅ No binding collisions across pipelines

## Next Steps (P2-03)

Refactor `pbr.wgsl` fragment shader to call `eval_brdf()` instead of inline BRDF computation. This requires updating the fragment shader to construct `ShadingParamsGPU` from material parameters and pass it to `eval_brdf()`.
