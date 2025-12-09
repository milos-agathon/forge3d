# WGSL Bind Group Layouts Reference

This document provides a comprehensive reference for all bind group layouts used across forge3d's WGSL shaders. Each pipeline has clearly defined resource binding patterns that must match between Rust pipeline creation and WGSL shader declarations.

## Overview

All shaders follow a consistent binding pattern:
- **Group 0**: Global uniforms (camera, transforms, lighting)
- **Group 1**: Pipeline-specific uniforms and textures
- **Group 2**: Additional resources (IBL textures, shadow maps, storage buffers)

## Pipeline Bind Group Layouts

### 1. Terrain Pipeline (`terrain.wgsl`)

**Purpose**: Height-field terrain rendering with colormap lookup

```wgsl
// Group 0: Global uniforms (176 bytes, std140-compatible)
@group(0) @binding(0) var<uniform> globals : Globals;
struct Globals {
    view: mat4x4<f32>,                    // 64 B - View matrix
    proj: mat4x4<f32>,                    // 64 B - Projection matrix  
    sun_exposure: vec4<f32>,              // 16 B - xyz=sun_dir, w=exposure
    spacing_h_exag_pad: vec4<f32>,        // 16 B - x=dx, y=dy, z=height_exaggeration, w=palette_index
    _pad_tail: vec4<f32>,                 // 16 B - Tail padding
};

// Group 1: Height data
@group(1) @binding(0) var height_tex  : texture_2d<f32>;  // R32Float, non-filterable
@group(1) @binding(1) var height_samp : sampler;          // NonFiltering sampler

// Group 2: Color lookup table
@group(2) @binding(0) var lut_tex  : texture_2d<f32>;     // RGBA8 (sRGB/UNORM), 256×N multi-palette
@group(2) @binding(1) var lut_samp : sampler;             // Filtering sampler
```

**Notes**: 
- Height texture must use non-filtering sampler (Nearest) for R32Float compatibility
- LUT supports multiple palettes via palette_index in globals
- Analytic height fallback ensures variation even with 1×1 dummy height texture

### 2. PBR Material Pipeline (`pbr.wgsl`)

**Purpose**: Physically-based rendering with IBL support

```wgsl
// Group 0: Camera and lighting uniforms
@group(0) @binding(0) var<uniform> uniforms: Uniforms;      // Camera transforms
@group(0) @binding(1) var<uniform> lighting: PbrLighting;   // Light parameters

// Group 1: Material textures and properties
@group(1) @binding(0) var<uniform> material: PbrMaterial;         // Material parameters
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;    // Albedo/diffuse
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>; // Metallic (B) + Roughness (G)
@group(1) @binding(3) var normal_texture: texture_2d<f32>;        // Tangent-space normals
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;     // Ambient occlusion
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;      // Emissive color
@group(1) @binding(6) var material_sampler: sampler;             // Shared material sampler

// Group 2: Image-based lighting (IBL)
@group(2) @binding(0) var irradiance_texture: texture_2d<f32>;    // Diffuse irradiance map
@group(2) @binding(1) var irradiance_sampler: sampler;
@group(2) @binding(2) var prefilter_texture: texture_2d<f32>;     // Pre-filtered environment map
@group(2) @binding(3) var prefilter_sampler: sampler;
@group(2) @binding(4) var brdf_lut_texture: texture_2d<f32>;      // BRDF integration LUT
@group(2) @binding(5) var brdf_lut_sampler: sampler;
```

**Notes**:
- Follows glTF 2.0 PBR specification
- Metallic-roughness workflow (not specular-glossiness)
- IBL provides realistic environment lighting

### 3. Shadow Mapping Pipeline (`shadows.wgsl`)

**Purpose**: Cascaded shadow maps with PCF filtering

```wgsl
// Groups 0-1: Inherited from base pipeline (e.g., PBR uniforms)

// Group 2: Shadow mapping resources
@group(2) @binding(0) var<uniform> csm_uniforms: CsmUniforms;     // CSM parameters
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;    // Cascade depth maps
@group(2) @binding(2) var shadow_sampler: sampler_comparison;     // PCF comparison sampler

struct CsmUniforms {
    light_direction: vec4<f32>,           // Light direction in world space
    light_view: mat4x4<f32>,              // Light view matrix
    cascades: array<ShadowCascade, 4>,    // Up to 4 shadow cascades
    cascade_count: u32,                   // Number of active cascades
    pcf_kernel_size: u32,                 // PCF kernel size (1, 3, 5, or 7)
    depth_bias: f32,                      // Depth bias to prevent acne
    slope_bias: f32,                      // Slope-scaled bias
    shadow_map_size: f32,                 // Shadow map resolution
    debug_mode: u32,                      // Debug visualization mode
    _padding: vec2<f32>,
};
```

**Notes**:
- Memory constraint: cascade_count × shadow_map_size² × 4 bytes ≤ 256 MiB
- PCF kernel sizes: 1 (no PCF), 3 (3×3), 5 (5×5), 7 (7×7)
- Depth array texture supports up to 4 cascades

### 4. Environment Mapping Pipeline (`env_map.wgsl`)

**Purpose**: Environment/skybox rendering

```wgsl
// Group 0: Transform uniforms
@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Group 1: Environment textures  
@group(1) @binding(0) var env_texture: texture_2d<f32>;           // Environment map
@group(1) @binding(1) var env_sampler: sampler;
@group(1) @binding(2) var irradiance_texture: texture_2d<f32>;    // Irradiance map
@group(1) @binding(3) var irradiance_sampler: sampler;
```

### 5. Render Bundles Pipeline

**Purpose**: Efficient multi-object rendering with shared state (uses instanced shaders)

#### Standard Bundle Layout
```wgsl
// Group 0: Camera uniforms
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Group 1: Bundle-specific uniforms
@group(1) @binding(0) var<uniform> bundle_uniforms: BundleUniforms;

// Group 2: Bundle textures
@group(2) @binding(0) var bundle_texture: texture_2d<f32>;
@group(2) @binding(1) var bundle_sampler: sampler;
```

#### Instanced Rendering Layout (Extended)
```wgsl
// Group 0: Camera (same as above)
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Group 1: Instance data and materials
@group(1) @binding(1) var<uniform> particle_uniforms: ParticleUniforms;
@group(1) @binding(2) var<storage, read> transforms: array<mat4x4<f32>>;      // Instance transforms
@group(1) @binding(3) var<storage, read> materials: array<vec4<f32>>;         // Material properties

// Group 2: Texture arrays
@group(2) @binding(2) var material_textures: texture_2d_array<f32>;           // Texture array
```

### 6. Vector Graphics Pipelines

#### Line Anti-aliased (`line_aa.wgsl`)
```wgsl
// Group 0: Global uniforms
@group(0) @binding(0) var<uniform> uniforms: LineUniforms;
```

#### Point Instanced (`point_instanced.wgsl`)
```wgsl
// Group 0: Rendering uniforms  
@group(0) @binding(0) var<uniform> uniforms: PointUniforms;
@group(0) @binding(1) var<storage, read> point_data: array<PointInstance>;
@group(0) @binding(2) var<uniform> viewport: ViewportUniforms;
```

#### Polygon Fill (`polygon_fill.wgsl`)
```wgsl
// Group 0: Transform uniforms
@group(0) @binding(0) var<uniform> uniforms: PolygonUniforms;
```

### 7. Compute Pipelines

#### Mipmap Downsample (`mipmap_downsample.wgsl`)
```wgsl
// Group 0: Compute resources (all in one group)
@group(0) @binding(0) var<uniform> uniforms: MipmapUniforms;
@group(0) @binding(1) var src_texture: texture_2d<f32>;           // Source mip level
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_texture: texture_storage_2d<rgba32float, write>; // Destination
```

#### Culling Compute (`culling_compute.wgsl`)
```wgsl
// Group 0: Culling resources
@group(0) @binding(0) var<uniform> cull_uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> instances: array<InstanceData>;
@group(0) @binding(2) var<storage, read_write> visibility: array<u32>;
@group(0) @binding(3) var<storage, read_write> draw_commands: array<DrawCommand>;
```

### 8. Post-processing Pipelines

#### Tonemap (`tonemap.wgsl`, `postprocess_tonemap.wgsl`)
```wgsl
// Group 0: HDR input and controls
@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var hdr_sampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: TonemapUniforms;
```

#### OIT Compose (`oit_compose.wgsl`)
```wgsl
// Group 0: OIT buffers
@group(0) @binding(0) var<storage, read> fragments: array<OitFragment>;
@group(0) @binding(1) var<storage, read> head_pointers: array<u32>;
@group(0) @binding(2) var<uniform> oit_uniforms: OitUniforms;
```

## Binding Guidelines

### Memory Layout Rules
- **Uniform buffers**: Must follow std140 layout rules
  - `mat4x4<f32>`: 64-byte alignment
  - `vec4<f32>`: 16-byte alignment  
  - Struct padding: Multiple of 16 bytes
- **Storage buffers**: Follow std430 layout rules (tighter packing)

### Sampler Requirements
- **R32Float textures**: Must use non-filtering samplers (Nearest)
- **Depth textures**: Use `sampler_comparison` for PCF
- **sRGB textures**: Automatic gamma correction on sampling

### Resource Constraints
- **Shadow atlas memory**: cascade_count × shadow_map_size² × 4 ≤ 256 MiB
- **Texture arrays**: Limited by GPU max array layers (typically 256-2048)
- **Uniform buffer size**: Limited to 64 KiB on some backends

### Backend Compatibility
- **Vulkan**: Full feature support
- **Metal**: May require format adaptations for R32Float
- **DirectX 12**: Full support with validation layers
- **OpenGL**: May have reduced functionality on older versions

## Validation

Each pipeline's bind group layout is validated at runtime against the WGSL shader requirements. Mismatched layouts will result in pipeline creation failures with descriptive error messages indicating the expected vs actual binding configuration.

For debugging bind group issues:
1. Enable wgpu validation features in debug builds  
2. Use labeled resources for clearer error messages
3. Check bind group compatibility with `wgpu::Device::create_render_pipeline`
4. Verify uniform buffer alignments match WGSL struct layouts