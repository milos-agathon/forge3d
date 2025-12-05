// terrain_shadow_depth.wgsl
// Depth-only terrain rendering for CSM shadow passes
// Renders terrain heightmap as a tessellated grid from light's perspective

// Shadow pass uniforms (per-cascade)
// Size: 112 bytes - must match Rust struct exactly
struct ShadowPassUniforms {
    // Light view-projection matrix for this cascade (64 bytes)
    light_view_proj: mat4x4<f32>,
    // Terrain params: (spacing, height_exag, height_min, height_max) (16 bytes)
    terrain_params: vec4<f32>,
    // Grid params: (grid_resolution, _pad, _pad, _pad) (16 bytes)
    grid_params: vec4<f32>,
    // Height curve params: (mode, strength, power, _pad) (16 bytes)
    // mode: 0=linear, 1=pow, 2=smoothstep, 3=lut (not supported)
    height_curve: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u_shadow: ShadowPassUniforms;

@group(0) @binding(1)
var height_tex: texture_2d<f32>;

@group(0) @binding(2)
var height_samp: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

/// Apply height curve to normalized height value (matching main shader)
/// t: input normalized height [0, 1]
/// Returns: curved normalized height [0, 1]
fn apply_height_curve(t: f32) -> f32 {
    let mode = u32(u_shadow.height_curve.x + 0.5);
    let strength = clamp(u_shadow.height_curve.y, 0.0, 1.0);
    let power = max(u_shadow.height_curve.z, 0.01);
    
    if (strength <= 0.0) {
        return t;
    }
    
    var curved = t;
    if (mode == 1u) { // pow
        curved = pow(t, power);
    } else if (mode == 2u) { // smoothstep
        curved = t * t * (3.0 - 2.0 * t);
    }
    // mode 3 (lut) not supported in shadow pass, falls back to linear
    
    return mix(t, curved, strength);
}

/// Vertex shader for shadow depth pass
/// Uses vertex_index to generate a grid of vertices covering the terrain
@vertex
fn vs_shadow(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Extract parameters from packed vec4s
    let terrain_spacing = u_shadow.terrain_params.x;
    let height_exag = u_shadow.terrain_params.y;
    let height_min = u_shadow.terrain_params.z;
    let height_max = u_shadow.terrain_params.w;
    
    // DEBUG: Verify parameters are sane
    // If height_min is ~0 instead of ~3363, there's a uniform mismatch
    // Expected values: terrain_spacing~1, height_exag~2, height_min~3363, height_max~4055
    let grid_res = u32(u_shadow.grid_params.x);
    
    // Decode vertex position from index
    // We're rendering as triangles, 6 vertices per quad, (grid_res-1)^2 quads
    let quads_per_row = grid_res - 1u;
    
    // Which quad and which vertex within the quad
    let triangle_idx = vertex_id / 3u;
    let vertex_in_tri = vertex_id % 3u;
    let quad_idx = triangle_idx / 2u;
    let tri_in_quad = triangle_idx % 2u;
    
    let quad_x = quad_idx % quads_per_row;
    let quad_y = quad_idx / quads_per_row;
    
    // Vertex offsets for the two triangles in a quad
    // Triangle 0: (0,0), (1,0), (0,1)
    // Triangle 1: (1,0), (1,1), (0,1)
    var dx: u32;
    var dy: u32;
    if (tri_in_quad == 0u) {
        // First triangle
        if (vertex_in_tri == 0u) { dx = 0u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 0u; }
        else { dx = 0u; dy = 1u; }
    } else {
        // Second triangle
        if (vertex_in_tri == 0u) { dx = 1u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 1u; }
        else { dx = 0u; dy = 1u; }
    }
    
    let grid_x = quad_x + dx;
    let grid_y = quad_y + dy;
    
    // Convert grid position to UV [0,1]
    let uv = vec2<f32>(
        f32(grid_x) / f32(grid_res - 1u),
        f32(grid_y) / f32(grid_res - 1u)
    );
    
    // Sample height from heightmap
    let h_raw = textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
    
    // Normalize height to [0,1] range (same as main shader's get_height_geom_t)
    let h_range = max(height_max - height_min, 1e-6);
    let h_norm = clamp((h_raw - height_min) / h_range, 0.0, 1.0);
    
    // Apply height curve (MUST match main shader for shadow consistency)
    let h_curved = apply_height_curve(h_norm);
    
    // SHADOW-NORMALIZED height calculation:
    // For shadow mapping, we normalize Z to [0, height_exag] range to match XY scale [-0.5, 0.5]
    // This ensures the terrain aspect ratio is reasonable for shadow calculations.
    // Note: This differs from the main shader's world_z which uses raw meters.
    let world_z = h_curved * height_exag;  // Normalized: [0, height_exag]
    
    // Map UV to world XY (terrain centered at origin)
    let world_xy = (uv - vec2<f32>(0.5, 0.5)) * terrain_spacing;
    let world_pos = vec3<f32>(world_xy.x, world_xy.y, world_z);
    
    // Transform to light clip space
    out.clip_position = u_shadow.light_view_proj * vec4<f32>(world_pos, 1.0);
    
    return out;
}

/// Fragment shader for shadow depth pass
/// No color output - depth is written automatically
@fragment
fn fs_shadow() {
    // No-op: depth is written automatically by the rasterizer
}
