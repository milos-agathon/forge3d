// Cascaded Shadow Maps - Depth Pass Generation
// CSM depth pass for cascaded shadow map generation

// Shadow cascade data
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // Light-space projection matrix
    near_distance: f32,             // Near plane distance
    far_distance: f32,              // Far plane distance
    texel_size: f32,                // Texel size in world space
    _padding: f32,
}

// CSM uniform data for shadow generation
struct CsmUniforms {
    light_direction: vec4<f32>,     // Light direction in world space
    light_view: mat4x4<f32>,        // Light view matrix
    cascades: array<ShadowCascade, 4>, // Shadow cascades
    cascade_count: u32,             // Number of active cascades
    pcf_kernel_size: u32,           // PCF kernel size (1, 3, 5, or 7)
    depth_bias: f32,                // Depth bias to prevent acne
    slope_bias: f32,                // Slope-scaled bias
    shadow_map_size: f32,           // Shadow map resolution
    debug_mode: u32,                // Debug visualization mode
    _padding: vec2<f32>,
}

// Bind group for shadow generation resources
@group(2) @binding(0) var<uniform> csm_uniforms: CsmUniforms;

// Convert world position to light space for cascade
fn world_to_light_space(world_pos: vec3<f32>, cascade_idx: u32) -> vec4<f32> {
    let light_space_pos = csm_uniforms.cascades[cascade_idx].light_projection * vec4<f32>(world_pos, 1.0);
    return light_space_pos;
}

// Vertex shader for shadow map rendering
struct ShadowVertexInput {
    @location(0) position: vec3<f32>,
}

struct ShadowVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn shadow_vs_main(input: ShadowVertexInput, @builtin(instance_index) cascade_idx: u32) -> ShadowVertexOutput {
    var out: ShadowVertexOutput;
    
    // Transform vertex to light space for current cascade
    out.clip_position = csm_uniforms.cascades[cascade_idx].light_projection * vec4<f32>(input.position, 1.0);
    
    return out;
}

// Fragment shader for shadow map rendering
@fragment
fn shadow_fs_main() -> @location(0) vec4<f32> {
    // Depth is written automatically, just return dummy color
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}