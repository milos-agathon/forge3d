// Shadow pass shader: Depth-only rendering from light's perspective
// Used to generate shadow map for terrain draping

// ============================================================================
// Bind Groups
// ============================================================================

// Group 0: Light view-projection matrix + UV transform
struct ShadowGlobals {
    light_view_proj: mat4x4<f32>,  // 64 bytes
    z_dir: f32,                     // 4 bytes
    zscale: f32,                    // 4 bytes
    uv_scale_x: f32,                // 4 bytes
    uv_scale_y: f32,                // 4 bytes
    uv_offset_x: f32,               // 4 bytes
    uv_offset_y: f32,               // 4 bytes
    y_flip: u32,                    // 4 bytes
    _pad: u32,                      // 4 bytes (padding)
}

@group(0) @binding(0) var<uniform> shadow_globals: ShadowGlobals;

// Group 1: DEM height texture (same as main pass)
@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var height_samp: sampler;

// ============================================================================
// Vertex Shader
// ============================================================================

struct VertexInput {
    @location(0) position: vec2<f32>,  // XZ plane position
    @location(1) uv: vec2<f32>,        // Texture coordinates
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

// UV transformation (using shadow_globals UV parameters)
fn apply_uv_transform(uv: vec2<f32>) -> vec2<f32> {
    let uv_y_adjusted = select(uv.y, 1.0 - uv.y, shadow_globals.y_flip != 0u);
    let uv_flipped = vec2<f32>(uv.x, uv_y_adjusted);
    let scale = vec2<f32>(shadow_globals.uv_scale_x, shadow_globals.uv_scale_y);
    let offset = vec2<f32>(shadow_globals.uv_offset_x, shadow_globals.uv_offset_y);
    return clamp(uv_flipped * scale + offset, vec2<f32>(0.0), vec2<f32>(1.0));
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform UV coordinates
    let uv_transformed = apply_uv_transform(in.uv);
    
    // Sample height from DEM
    let height = textureSampleLevel(height_tex, height_samp, uv_transformed, 0.0).r;
    
    // Build world position (Y-up coordinate system)
    let world_pos = vec3<f32>(
        in.position.x,
        height * shadow_globals.zscale * shadow_globals.z_dir,
        in.position.y
    );
    
    // Transform to light clip space
    out.clip_position = shadow_globals.light_view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

// ============================================================================
// Fragment Shader
// ============================================================================

// Depth-only pass - no fragment shader needed (depth writes automatically)
// But we include a minimal one for compatibility
@fragment
fn fs_main() {
    // No-op: depth is written automatically
}
