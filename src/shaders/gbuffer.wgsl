// GBuffer pass for screen-space effects (P5)
// Outputs per-fragment data consumed by SSAO/GTAO, SSGI, and SSR:
//   - depth         : linear view-space depth (>0 for geometry, 0.0 for background)
//   - normal.xyz    : view-space normal encoded into [0,1]
//   - normal.w      : perceptual roughness in [0,1]
//   - albedo_metal.rgb : diffuse/base color in linear space (texture * material)
//   - albedo_metal.a   : metallic factor in [0,1]

struct GBufferOutput {
    @location(0) depth: f32,
    @location(1) normal: vec4<f32>,      // xyz = view-space normal in [0,1], w = roughness [0,1]
    @location(2) albedo_metal: vec4<f32>, // rgb = diffuse/base color, a = metallic [0,1]
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) view_position: vec3<f32>,
    @location(3) view_normal: vec3<f32>,
    @location(4) uv: vec2<f32>,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    eye_position: vec3<f32>,
    _pad0: f32,
    view_dir: vec3<f32>,
    _pad1: f32,
}

struct ModelUniforms {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct MaterialUniforms {
    base_color: vec4<f32>,
    roughness: f32,
    metallic: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var base_color_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = model.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;

    // Transform normal to world space
    out.world_normal = normalize((model.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);

    // Transform to view space for screen-space techniques
    let view_pos = camera.view * world_pos;
    out.view_position = view_pos.xyz;
    out.view_normal = normalize((camera.view * vec4<f32>(out.world_normal, 0.0)).xyz);

    out.uv = in.uv;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;

    // Store linear depth (view-space Z)
    out.depth = -in.view_position.z;

    // Store view-space normal (normalized) and roughness
    let normal = normalize(in.view_normal);
    out.normal = vec4<f32>(normal * 0.5 + 0.5, material.roughness);

    // Sample albedo and store with metallic
    let albedo = textureSample(base_color_texture, base_color_sampler, in.uv);
    out.albedo_metal = vec4<f32>(albedo.rgb * material.base_color.rgb, material.metallic);

    return out;
}

// Depth reconstruction utilities for screen-space techniques

fn reconstruct_view_position(uv: vec2<f32>, depth: f32, inv_proj: mat4x4<f32>) -> vec3<f32> {
    // Convert UV [0,1] to NDC [-1,1]
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Reconstruct clip-space position
    let clip_pos = vec4<f32>(ndc, depth, 1.0);

    // Transform to view space
    let view_pos = inv_proj * clip_pos;
    return view_pos.xyz / view_pos.w;
}

fn reconstruct_view_position_from_depth(uv: vec2<f32>, linear_depth: f32, inv_proj: mat4x4<f32>) -> vec3<f32> {
    // Fast reconstruction from linear depth (assumes perspective projection)
    let ndc_xy = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    // Extract inverse projection terms for faster reconstruction
    let focal = vec2<f32>(inv_proj[0][0], inv_proj[1][1]);
    let center = vec2<f32>(inv_proj[2][0], inv_proj[2][1]);

    let view_xy = (ndc_xy - center) / focal;
    return vec3<f32>(view_xy * linear_depth, -linear_depth);
}

fn unpack_normal(packed: vec4<f32>) -> vec3<f32> {
    return normalize(packed.xyz * 2.0 - 1.0);
}

fn get_roughness(packed: vec4<f32>) -> f32 {
    return packed.w;
}
