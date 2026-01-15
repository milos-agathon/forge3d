// P1.4: Terrain velocity buffer generation for TAA
// Computes screen-space motion vectors from depth + camera matrices

struct VelocityUniforms {
    current_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    resolution: vec2<f32>,
    jitter_current: vec2<f32>,
    jitter_prev: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: VelocityUniforms;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var velocity_out: texture_storage_2d<rg16float, write>;

// Reconstruct world position from depth
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC [-1, 1]
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    let clip_pos = vec4<f32>(ndc, depth, 1.0);
    
    // Transform to world space
    let world_pos = uniforms.inv_view_proj * clip_pos;
    return world_pos.xyz / world_pos.w;
}

@compute @workgroup_size(8, 8, 1)
fn generate_velocity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = vec2<i32>(global_id.xy);
    let dims = vec2<i32>(uniforms.resolution);
    
    if (pixel.x >= dims.x || pixel.y >= dims.y) {
        return;
    }
    
    // Sample depth
    let depth = textureLoad(depth_tex, pixel, 0).r;
    
    // Skip sky pixels (depth = 1.0)
    if (depth >= 0.9999) {
        textureStore(velocity_out, pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }
    
    // Current UV (with jitter removed for accurate reprojection)
    let uv = (vec2<f32>(pixel) + 0.5) / uniforms.resolution;
    let uv_unjittered = uv - uniforms.jitter_current / uniforms.resolution;
    
    // Reconstruct world position
    let world_pos = reconstruct_world_pos(uv_unjittered, depth);
    
    // Project to previous frame
    let prev_clip = uniforms.prev_view_proj * vec4<f32>(world_pos, 1.0);
    let prev_ndc = prev_clip.xyz / prev_clip.w;
    
    // Convert previous NDC to UV
    let prev_uv = vec2<f32>(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    
    // Add previous jitter offset
    let prev_uv_jittered = prev_uv + uniforms.jitter_prev / uniforms.resolution;
    
    // Compute velocity (current - previous) in UV space
    let velocity = uv - prev_uv_jittered;
    
    // Store velocity (scale to reasonable range for Rg16Float)
    textureStore(velocity_out, pixel, vec4<f32>(velocity, 0.0, 0.0));
}
