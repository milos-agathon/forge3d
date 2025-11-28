# Terrain PBR+POM Shader - Quick Reference

## File Location
`src/shaders/terrain_pbr_pom.wgsl`

## Overview
Advanced terrain rendering shader combining:
- **PBR:** Physical-Based Rendering (Cook-Torrance BRDF)
- **POM:** Parallax Occlusion Mapping for surface detail
- **Triplanar:** Distortion-free texture mapping
- **IBL:** Image-Based Lighting for realistic ambient
- **Shadows:** Shadow mapping with PCF

## Bind Group Layout

### Group 0: Global Uniforms
```wgsl
struct Globals {
    view: mat4x4<f32>,                // Camera view matrix
    proj: mat4x4<f32>,                // Projection matrix
    sun_exposure: vec4<f32>,          // xyz=sun_dir, w=exposure
    spacing_h_exag_pad: vec4<f32>,    // x=dx, y=dy, z=exag, w=palette_idx
    camera_pos: vec4<f32>,            // xyz=position, w=time
    z_scale: f32,                     // Vertical exaggeration
    render_scale: f32,                // Resolution scale
    fov_y: f32,                       // Field of view
}

@group(0) @binding(0) var<uniform> globals: Globals;
```

### Group 1: Height Map
```wgsl
@group(1) @binding(0) var height_tex: texture_2d<f32>;   // R32Float
@group(1) @binding(1) var height_samp: sampler;          // NonFiltering
```

### Group 2: Colormap LUT
```wgsl
@group(2) @binding(0) var lut_tex: texture_2d<f32>;      // RGBA8 (256×N)
@group(2) @binding(1) var lut_samp: sampler;
```

### Group 3: Material Textures
```wgsl
@group(3) @binding(0) var material_albedo_tex: texture_2d<f32>;
@group(3) @binding(1) var material_normal_tex: texture_2d<f32>;
@group(3) @binding(2) var material_roughness_tex: texture_2d<f32>;
@group(3) @binding(3) var material_height_tex: texture_2d<f32>;   // For POM
@group(3) @binding(4) var material_samp: sampler;
```

### Group 4: Rendering Parameters
```wgsl
struct TriplanarParams {
    scale: f32,              // Texture tiling scale (e.g., 6.0)
    blend_sharpness: f32,    // Blend weight power (e.g., 4.0)
    normal_strength: f32,    // Normal map intensity (e.g., 1.0)
}

struct PomParams {
    enabled: u32,            // 1 = enabled, 0 = disabled
    mode: u32,               // 0=Parallax, 1=Relief, 2=Occlusion
    scale: f32,              // Height displacement (e.g., 0.04)
    min_steps: u32,          // Min ray march steps (e.g., 12)
    max_steps: u32,          // Max ray march steps (e.g., 40)
    refine_steps: u32,       // Binary refinement (e.g., 4)
    shadow_enabled: u32,     // POM self-shadowing
    occlusion_enabled: u32,  // POM occlusion
}

struct PbrParams {
    metallic: f32,           // Material metallic [0,1]
    roughness: f32,          // Material roughness [0.04,1.0]
    ao_strength: f32,        // Ambient occlusion strength
    albedo_mode: u32,        // 0=colormap, 1=mix, 2=material
    colormap_strength: f32,  // Colormap blend [0,1]
}

@group(4) @binding(0) var<uniform> triplanar: TriplanarParams;
@group(4) @binding(1) var<uniform> pom: PomParams;
@group(4) @binding(2) var<uniform> pbr: PbrParams;
```

### Post-Processing Parameters

Current builds pack the renderer-side post processing controls into ``TerrainShadingUniforms``:

```wgsl
struct TerrainShadingUniforms {
    triplanar_params : vec4<f32>; // x=scale, y=blend_sharpness, z=normal_strength, w=pom_scale
    pom_steps        : vec4<f32>; // x=min_steps, y=max_steps, z=refine_steps, w=flags
    layer_heights    : vec4<f32>;
    layer_roughness  : vec4<f32>;
    layer_metallic   : vec4<f32>;
    layer_control    : vec4<f32>;
    light_params     : vec4<f32>;
    clamp0           : vec4<f32>;
    clamp1           : vec4<f32>;
    clamp2           : vec4<f32>;
    post_params      : vec4<f32>; // x=exposure, y=gamma, z=colormap_strength, w=unused
};
```

The defaults mirror ``TerrainRenderParamsConfig``:

- ``exposure = 1.0`` keeps rendered luminance in camera space and feeds the ACES tonemapper.
- ``gamma = 2.2`` applies the final display transform since the render target is `Rgba8Unorm`.
- ``colormap_strength = 0.5`` blends material albedo with the overlay LUT (multiplied by overlay strength).

When adjusting tone-mapping or colormap blending from Python, set the corresponding dataclass fields so the uniforms stay in sync with the shader.

### Group 5: IBL Environment Maps
```wgsl
@group(5) @binding(0) var ibl_diffuse_tex: texture_2d<f32>;   // Irradiance
@group(5) @binding(1) var ibl_specular_tex: texture_2d<f32>;  // Prefiltered
@group(5) @binding(2) var ibl_brdf_lut: texture_2d<f32>;      // BRDF LUT
@group(5) @binding(3) var ibl_samp: sampler;

struct IblParams {
    enabled: u32,
    intensity: f32,
    rotation_deg: f32,
}

@group(5) @binding(4) var<uniform> ibl: IblParams;
```

### Group 6: Shadow Map
```wgsl
@group(6) @binding(0) var shadow_tex: texture_depth_2d;
@group(6) @binding(1) var shadow_samp: sampler_comparison;

struct ShadowParams {
    enabled: u32,
    technique: u32,          // 0=PCF, 1=PCSS, 2=ESM, 3=EVSM
    resolution: u32,
    cascades: u32,
    light_view_proj: mat4x4<f32>,
    softness: f32,
    intensity: f32,
    bias: f32,
}

@group(6) @binding(2) var<uniform> shadow: ShadowParams;
```

## Entry Points

### Vertex Shader
```wgsl
@vertex
fn vs_main(in: VsIn) -> VsOut
```

**Input:**
- `pos_xy: vec2<f32>` - Terrain grid position
- `uv: vec2<f32>` - UV coordinates [0,1]

**Output:**
- `clip_pos: vec4<f32>` - Clip-space position
- `world_pos: vec3<f32>` - World-space position
- `world_normal: vec3<f32>` - Surface normal (from Sobel)
- `height: f32` - Terrain height
- `view_dir: vec3<f32>` - View direction

### Fragment Shader
```wgsl
@fragment
fn fs_main(in: VsOut) -> FsOut
```

**Output:**
- `color: vec4<f32>` - Final RGBA (tone-mapped + gamma)
- `normal_depth: vec4<f32>` - Encoded normal + linear depth

## Key Functions

### Task 4.1: Normal Calculation
```wgsl
fn calculate_normal(
    uv: vec2<f32>,
    texel_size: vec2<f32>,
    spacing: f32,
    exaggeration: f32
) -> vec3<f32>
```
Calculates surface normal from height map using Sobel filter.

**Usage:**
```wgsl
let texel = vec2(1.0) / tex_dims;
let normal = calculate_normal(in.uv, texel, spacing, exaggeration);
```

### Task 4.2: Triplanar Sampling
```wgsl
fn sample_triplanar(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    scale: f32,
    blend_sharpness: f32,
    tex: texture_2d<f32>,
    samp: sampler
) -> vec4<f32>
```
Samples texture using triplanar projection to avoid UV distortion.

**Usage:**
```wgsl
let albedo = sample_triplanar(
    in.world_pos,
    normal,
    triplanar.scale,
    triplanar.blend_sharpness,
    material_albedo_tex,
    material_samp
);
```

```wgsl
fn sample_triplanar_normal(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    scale: f32,
    blend_sharpness: f32,
    strength: f32
) -> vec3<f32>
```
Samples normal map using triplanar projection.

### Task 4.3: Parallax Occlusion Mapping
```wgsl
fn parallax_occlusion_mapping(
    uv: vec2<f32>,
    view_dir_tangent: vec3<f32>,
    height_scale: f32,
    min_steps: u32,
    max_steps: u32,
    refine_steps: u32
) -> vec2<f32>
```
Displaces UVs based on height map for parallax effect.

**Usage:**
```wgsl
// Build TBN for tangent space
let T = normalize(cross(N, vec3(0.0, 0.0, 1.0)));
let B = normalize(cross(N, T));
let TBN_inv = transpose(mat3x3(T, B, N));
let view_tangent = TBN_inv * view_dir;

let displaced_uv = parallax_occlusion_mapping(
    in.uv,
    view_tangent,
    pom.scale,
    pom.min_steps,
    pom.max_steps,
    pom.refine_steps
);
```

```wgsl
fn pom_self_shadow(
    uv: vec2<f32>,
    light_dir_tangent: vec3<f32>,
    height: f32,
    num_shadow_steps: u32
) -> f32
```
Calculates POM self-shadowing factor [0,1].

### Task 4.4: PBR BRDF
```wgsl
fn calculate_pbr_brdf(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
    f0: vec3<f32>
) -> vec3<f32>
```
Computes Cook-Torrance BRDF (diffuse + specular).

**Usage:**
```wgsl
// Calculate F0
let dielectric_f0 = vec3(0.04);
let f0 = mix(dielectric_f0, albedo, metallic);

// Direct lighting
let direct = calculate_pbr_brdf(
    normal,
    view_dir,
    light_dir,
    albedo,
    roughness,
    metallic,
    f0
);
```

**Component Functions:**
```wgsl
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32>
fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32>
```

## IBL Functions
```wgsl
fn sample_ibl_irradiance(normal: vec3<f32>) -> vec3<f32>
fn sample_ibl_specular(reflection: vec3<f32>, roughness: f32) -> vec3<f32>
fn sample_brdf_lut(n_dot_v: f32, roughness: f32) -> vec2<f32>
```

## Shadow Functions
```wgsl
fn sample_shadow_pcf(shadow_pos: vec4<f32>, bias: f32) -> f32
```

## Tone Mapping
```wgsl
fn reinhard(x: vec3<f32>) -> vec3<f32>
fn gamma_correct(x: vec3<f32>, gamma: f32) -> vec3<f32>
```

## Typical Parameter Values

### Triplanar Settings
```rust
TriplanarParams {
    scale: 6.0,              // Texture tiling frequency
    blend_sharpness: 4.0,    // Higher = sharper transitions
    normal_strength: 1.0,    // Normal map intensity
}
```

### POM Settings
```rust
// High Quality
PomParams {
    enabled: 1,
    mode: 2,                 // Occlusion mapping
    scale: 0.04,             // Subtle displacement
    min_steps: 12,
    max_steps: 40,
    refine_steps: 4,
    shadow_enabled: 1,
    occlusion_enabled: 1,
}

// Performance
PomParams {
    enabled: 1,
    mode: 0,                 // Basic parallax
    scale: 0.02,
    min_steps: 8,
    max_steps: 16,
    refine_steps: 0,
    shadow_enabled: 0,
    occlusion_enabled: 0,
}
```

### PBR Settings
```rust
PbrParams {
    metallic: 0.0,           // Non-metallic terrain
    roughness: 0.7,          // Slightly rough surface
    ao_strength: 1.0,
    albedo_mode: 1,          // Mix colormap + material
    colormap_strength: 0.5,  // 50/50 blend
}
```

### IBL Settings
```rust
IblParams {
    enabled: 1,
    intensity: 1.0,
    rotation_deg: 0.0,
}
```

### Shadow Settings
```rust
ShadowParams {
    enabled: 1,
    technique: 0,            // PCF
    resolution: 4096,
    cascades: 3,
    softness: 1.5,
    intensity: 0.8,          // 80% shadow opacity
    bias: 0.002,
}
```

## IBL On/Off Comparison (Terrain Demo)

The easiest way to visually validate that terrain PBR is correctly picking up IBL is to render
two frames with identical geometry, camera, and direct light, but different GI modes.

Use the public CLI in `examples/terrain_demo.py`:

```bash
# IBL ON: full PBR + IBL, material albedo only (no elevation colormap)
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr \
  --size 1600 900 \
  --render-scale 1.0 \
  --msaa 4 \
  --z-scale 2.0 \
  --albedo-mode material \
  --colormap-strength 0.0 \
  --gi ibl \
  --output examples/out/terrain_ibl_on.png

# IBL OFF: same setup, but GI excludes IBL so only the directional light remains
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr \
  --size 1600 900 \
  --render-scale 1.0 \
  --msaa 4 \
  --z-scale 2.0 \
  --albedo-mode material \
  --colormap-strength 0.0 \
  --gi "" \
  --output examples/out/terrain_ibl_off.png
```

**Expected differences:**

- **terrain_ibl_on.png**
  - Steep rock and snow faces show view-dependent specular from the HDR environment.
  - Overall tint and fill light match the sky colors from `snow_field_4k.hdr`.
  - The lowered snow roughness (see `MaterialSet::terrain_default`) should make high-altitude
    regions visibly glossier under grazing angles.

- **terrain_ibl_off.png**
  - Lighting is dominated by the directional sun; shadows and contrast remain, but ambient
    fill is flatter.
  - Specular reflections from the environment vanish; surfaces read more "matte" overall.

When tuning exposure or HDR choices, keep these two renders in sync and adjust only
camera/light/IBL parameters from the Python side so that the uniform packing into
`TerrainShadingUniforms` and `IblUniforms` remains consistent with this reference.

## POM On/Off Grazing Debug View

To verify that **Parallax Occlusion Mapping (POM)** is correctly wired and
visibly contributing relief at grazing angles, you can render a close-up pair of
frames that differ only by the POM enable flag. These use the public
`examples/terrain_demo.py` CLI and the `--pom-disabled` switch.

```bash
# POM ON: low-angle close-up, material albedo only
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr \
  --size 1600 900 \
  --render-scale 1.0 \
  --msaa 4 \
  --z-scale 2.0 \
  --cam-radius 250 \
  --cam-phi 135 \
  --cam-theta 20 \
  --albedo-mode material \
  --colormap-strength 0.0 \
  --gi ibl \
  --ibl-intensity 1.0 \
  --sun-azimuth 135 \
  --sun-elevation 25 \
  --sun-intensity 3.0 \
  --output examples/out/terrain_pom_on.png --overwrite

# POM OFF: same setup, but POM disabled
python examples/terrain_demo.py \
  --dem assets/Gore_Range_Albers_1m.tif \
  --hdr assets/snow_field_4k.hdr \
  --size 1600 900 \
  --render-scale 1.0 \
  --msaa 4 \
  --z-scale 2.0 \
  --cam-radius 250 \
  --cam-phi 135 \
  --cam-theta 20 \
  --albedo-mode material \
  --colormap-strength 0.0 \
  --gi ibl \
  --ibl-intensity 1.0 \
  --sun-azimuth 135 \
  --sun-elevation 25 \
  --sun-intensity 3.0 \
  --pom-disabled \
  --output examples/out/terrain_pom_off.png --overwrite
```

**Expected differences:**

- **terrain_pom_on.png**
  - At grazing angles, small-scale height detail appears to slide relative to
    the camera, giving a stronger sense of surface relief.
  - Ridges and micro-features exhibit self-occlusion consistent with the
    `PomSettings` scale and step counts.

- **terrain_pom_off.png**
  - Geometry is identical (same DEM), but fine detail looks flatter; textures
    behave more like simple triplanar mapping without parallax.
  - The apparent depth of cracks and ridges no longer varies with view
    direction; silhouettes remain the same, but interior detail loses its
    “3D” pop.

These commands rely on the existing `PomSettings` defaults from
`make_terrain_params_config`. The `--pom-disabled` flag only flips the
`pom.enabled` bit in the Python configuration before it is bridged to Rust via
`TerrainRenderParams`.

## Performance Tuning

**High Quality (≥60 fps @ 1080p):**
- POM: max_steps=40, refine_steps=4
- Shadow: resolution=4096, PCF 3×3
- Triplanar: blend_sharpness=4.0

**Medium Quality (≥120 fps @ 1080p):**
- POM: max_steps=24, refine_steps=2
- Shadow: resolution=2048, PCF 3×3
- Triplanar: blend_sharpness=2.0

**Low Quality (≥240 fps @ 1080p):**
- POM: max_steps=16, refine_steps=0
- Shadow: resolution=1024, simple PCF
- Triplanar: blend_sharpness=1.0

## Common Issues

### POM artifacts at edges
- Increase `min_steps` to 16+
- Add refinement steps (4+)
- Reduce `height_scale`

### Triplanar seams visible
- Increase `blend_sharpness` (4.0-8.0)
- Ensure textures tile seamlessly
- Check normal map sampling

### Dark/bright lighting
- Adjust `globals.sun_exposure.w` (exposure)
- Check IBL intensity
- Verify F0 calculation for metallics

### Shadow acne
- Increase shadow bias (0.002-0.005)
- Use slope-scale bias
- Check depth precision

## Example Integration (Rust)

```rust
// Create bind group layouts
let globals_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
    entries: &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(std::mem::size_of::<Globals>() as u64).unwrap()),
            },
            count: None,
        },
    ],
    label: Some("Globals Layout"),
});

// Create uniform buffers
let globals = Globals {
    view: view_matrix,
    proj: proj_matrix,
    sun_exposure: [sun_dir.x, sun_dir.y, sun_dir.z, exposure],
    // ... etc
};

let globals_buffer = device.create_buffer_init(&BufferInitDescriptor {
    label: Some("Globals Buffer"),
    contents: bytemuck::cast_slice(&[globals]),
    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
});

// Create pipeline
let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
    vertex: VertexState {
        module: &shader_module,
        entry_point: "vs_main",
        // ...
    },
    fragment: Some(FragmentState {
        module: &shader_module,
        entry_point: "fs_main",
        // ...
    }),
    // ...
});
```

## References

- **Task Breakdown:** `terrain_demo_task_breakdown.md`
- **Completion Report:** `MILESTONE_4_COMPLETE.md`
- **Verification:** `tests/verify_terrain_pbr_pom_shader.py`

---

**Last Updated:** 2025-10-22
**Shader Version:** 1.0
**Total Functions:** 15+
**Total Lines:** 706
