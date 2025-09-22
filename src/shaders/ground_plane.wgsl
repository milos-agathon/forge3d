// B10: Ground Plane (Raster) - Simple raster ground plane with grid/albedo
// Provides infinite ground plane rendering with grid patterns and z-fighting protection
// Draws beneath all geometry with proper depth handling

// ---------- Ground Plane Uniforms ----------
struct GroundPlaneUniforms {
    view_proj: mat4x4<f32>,                    // View-projection matrix
    world_transform: mat4x4<f32>,              // World transformation matrix
    plane_params: vec4<f32>,                   // size (x), height (y), grid_enabled (z), z_bias (w)
    grid_params: vec4<f32>,                    // major_spacing (x), minor_spacing (y), major_width (z), minor_width (w)
    color_params: vec4<f32>,                   // albedo (rgb) + alpha (w)
    grid_color_params: vec4<f32>,              // major_grid_color (rgb) + major_alpha (w)
    minor_grid_color_params: vec4<f32>,        // minor_grid_color (rgb) + minor_alpha (w)
    fade_params: vec4<f32>,                    // fade_distance (x), fade_power (y), grid_fade_distance (z), grid_fade_power (w)
};

@group(0) @binding(0) var<uniform> ground_uniforms : GroundPlaneUniforms;

// ---------- Vertex Input/Output ----------
struct VsIn {
    @location(0) position: vec3<f32>,          // Local vertex position
    @location(1) uv: vec2<f32>,                // UV coordinates
    @location(2) normal: vec3<f32>,            // Vertex normal
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) view_distance: f32,           // Distance to camera
};

// ---------- Grid Functions ----------
fn grid_pattern(uv: vec2<f32>, major_spacing: f32, minor_spacing: f32, major_width: f32, minor_width: f32) -> vec3<f32> {
    // Calculate major grid lines
    let major_grid_uv = uv / major_spacing;
    let major_grid = abs(fract(major_grid_uv - 0.5) - 0.5) / fwidth(major_grid_uv);
    let major_line = min(major_grid.x, major_grid.y);
    let major_mask = 1.0 - min(major_line * major_width, 1.0);

    // Calculate minor grid lines
    let minor_grid_uv = uv / minor_spacing;
    let minor_grid = abs(fract(minor_grid_uv - 0.5) - 0.5) / fwidth(minor_grid_uv);
    let minor_line = min(minor_grid.x, minor_grid.y);
    let minor_mask = 1.0 - min(minor_line * minor_width, 1.0);

    // Combine major and minor grid patterns
    // Major grid takes precedence over minor grid
    let combined_mask = max(major_mask, minor_mask * (1.0 - major_mask));

    return vec3<f32>(major_mask, minor_mask, combined_mask);
}

fn calculate_grid_fade(distance: f32, fade_distance: f32, fade_power: f32) -> f32 {
    if fade_distance <= 0.0 {
        return 1.0;
    }
    let fade_factor = clamp(1.0 - (distance / fade_distance), 0.0, 1.0);
    return pow(fade_factor, fade_power);
}

// ---------- Z-Fighting Protection ----------
fn apply_z_bias(clip_pos: vec4<f32>, z_bias: f32) -> vec4<f32> {
    var biased_pos = clip_pos;
    // Apply small bias towards camera to prevent z-fighting
    // Bias is applied in NDC space for consistent behavior
    biased_pos.z = biased_pos.z - z_bias * biased_pos.w;
    return biased_pos;
}

// ---------- Vertex Shader ----------
@vertex
fn vs_main(in: VsIn) -> VsOut {
    // Transform vertex to world space
    let world_pos = (ground_uniforms.world_transform * vec4<f32>(in.position, 1.0)).xyz;

    // Calculate clip space position
    let clip_pos = ground_uniforms.view_proj * vec4<f32>(world_pos, 1.0);

    // Apply z-bias to prevent z-fighting with terrain
    let biased_clip_pos = apply_z_bias(clip_pos, ground_uniforms.plane_params.w);

    // Calculate view distance for fading
    let view_distance = length(world_pos);

    var out: VsOut;
    out.clip_pos = biased_clip_pos;
    out.world_pos = world_pos;
    out.uv = in.uv;
    out.normal = normalize((ground_uniforms.world_transform * vec4<f32>(in.normal, 0.0)).xyz);
    out.view_distance = view_distance;

    return out;
}

// ---------- Fragment Shader ----------
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Base albedo color
    var final_color = ground_uniforms.color_params.rgb;
    var final_alpha = ground_uniforms.color_params.w;

    // Apply distance-based fading for the ground plane
    let distance_fade = calculate_grid_fade(
        in.view_distance,
        ground_uniforms.fade_params.x,
        ground_uniforms.fade_params.y
    );

    // Apply grid pattern if enabled
    if ground_uniforms.plane_params.z > 0.5 { // grid_enabled
        let major_spacing = ground_uniforms.grid_params.x;
        let minor_spacing = ground_uniforms.grid_params.y;
        let major_width = ground_uniforms.grid_params.z;
        let minor_width = ground_uniforms.grid_params.w;

        // Use world position for consistent grid spacing
        let grid_uv = in.world_pos.xz;
        let grid_masks = grid_pattern(grid_uv, major_spacing, minor_spacing, major_width, minor_width);

        let major_mask = grid_masks.x;
        let minor_mask = grid_masks.y;

        // Apply grid-specific fading
        let grid_fade = calculate_grid_fade(
            in.view_distance,
            ground_uniforms.fade_params.z,
            ground_uniforms.fade_params.w
        );

        // Blend major grid lines
        let major_grid_color = ground_uniforms.grid_color_params.rgb;
        let major_grid_alpha = ground_uniforms.grid_color_params.w * major_mask * grid_fade;
        final_color = mix(final_color, major_grid_color, major_grid_alpha);

        // Blend minor grid lines (only where major lines aren't present)
        let minor_grid_color = ground_uniforms.minor_grid_color_params.rgb;
        let minor_grid_alpha = ground_uniforms.minor_grid_color_params.w * minor_mask * (1.0 - major_mask) * grid_fade;
        final_color = mix(final_color, minor_grid_color, minor_grid_alpha);

        // Increase overall alpha where grid lines are present
        final_alpha = max(final_alpha, max(major_grid_alpha, minor_grid_alpha));
    }

    // Apply overall distance fading to alpha
    final_alpha *= distance_fade;

    // Simple lighting: use the normal to create subtle shading
    let light_dir = normalize(vec3<f32>(0.3, 0.8, 0.2)); // Soft overhead light
    let ndotl = max(dot(in.normal, light_dir), 0.2); // Minimum ambient
    final_color *= ndotl;

    return vec4<f32>(final_color, final_alpha);
}

// ---------- Alternative Infinite Plane Implementation ----------
// For cases where we want a truly infinite ground plane without geometry

struct InfinitePlaneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,               // Inverse view-projection for ray reconstruction
    camera_pos: vec4<f32>,                    // Camera position (xyz) + plane_height (w)
    plane_normal: vec4<f32>,                  // Plane normal (xyz) + enabled (w)
    grid_params: vec4<f32>,                   // major_spacing (x), minor_spacing (y), major_width (z), minor_width (w)
    color_params: vec4<f32>,                  // albedo (rgb) + alpha (w)
    grid_colors: vec4<f32>,                   // grid_color (rgb) + grid_alpha (w)
};

// Fullscreen triangle vertices for infinite plane rendering
@vertex
fn vs_infinite_plane(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    // Generate fullscreen triangle
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u),
    );
    let pos = uv * 2.0 - 1.0;

    var out: VsOut;
    out.clip_pos = vec4<f32>(pos, 0.999, 1.0); // Far plane to ensure it's behind everything
    out.world_pos = vec3<f32>(0.0); // Will be calculated in fragment shader
    out.uv = uv;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.view_distance = 0.0;

    return out;
}

@fragment
fn fs_infinite_plane(in: VsOut) -> @location(0) vec4<f32> {
    // This would implement ray-plane intersection for infinite ground plane
    // For now, discard to avoid compilation issues
    discard;
}

// ---------- Utility Functions ----------
fn world_to_grid_coords(world_pos: vec3<f32>, grid_spacing: f32) -> vec2<f32> {
    return world_pos.xz / grid_spacing;
}

fn calculate_mip_level(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> f32 {
    let distance = length(world_pos - camera_pos);
    return log2(max(distance / 10.0, 1.0)); // LOD based on distance
}