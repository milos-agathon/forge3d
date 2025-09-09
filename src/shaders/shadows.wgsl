// Cascaded Shadow Maps with PCF filtering
// Bind Groups and Layouts:
// - @group(2) Shadow resources
//   - @binding(0): uniform buffer `CsmUniforms`
//   - @binding(1): texture_depth_2d_array `shadow_maps`
//   - @binding(2): sampler_comparison `shadow_sampler`
// Formats:
// - Depth maps: D32Float
// Address Space: `uniform`, `fragment`
// Provides high-quality shadows for directional lights

// Shadow cascade data
struct ShadowCascade {
    light_projection: mat4x4<f32>,  // Light-space projection matrix
    near_distance: f32,             // Near plane distance
    far_distance: f32,              // Far plane distance
    texel_size: f32,                // Texel size in world space
    _padding: f32,
}

// CSM uniform data
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

// Bind group for shadow resources
@group(2) @binding(0) var<uniform> csm_uniforms: CsmUniforms;
@group(2) @binding(1) var shadow_maps: texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

// Convert world position to light space for cascade
fn world_to_light_space(world_pos: vec3<f32>, cascade_idx: u32) -> vec4<f32> {
    let light_space_pos = csm_uniforms.cascades[cascade_idx].light_projection * vec4<f32>(world_pos, 1.0);
    return light_space_pos;
}

// Select appropriate shadow cascade based on view depth
fn select_cascade(view_depth: f32) -> u32 {
    var cascade_idx = csm_uniforms.cascade_count - 1u;
    
    for (var i = 0u; i < csm_uniforms.cascade_count; i++) {
        if (view_depth <= csm_uniforms.cascades[i].far_distance) {
            cascade_idx = i;
            break;
        }
    }
    
    return cascade_idx;
}

// Basic shadow sampling (single sample)
fn sample_shadow_basic(light_space_pos: vec4<f32>, cascade_idx: u32) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check if position is within shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0; // Outside shadow map bounds - not in shadow
    }
    
    // Apply depth bias
    let biased_depth = shadow_coords.z - csm_uniforms.depth_bias;
    
    // Sample shadow map with comparison
    return textureSampleCompare(shadow_maps, shadow_sampler, 
                               shadow_coords.xy, cascade_idx, biased_depth);
}

// PCF (Percentage-Closer Filtering) implementation
fn sample_shadow_pcf(light_space_pos: vec4<f32>, cascade_idx: u32, world_normal: vec3<f32>) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check if position is within shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0; // Outside shadow map bounds
    }
    
    // Calculate slope-scaled bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = dot(world_normal, -light_dir);
    let slope_scale = sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l;
    let bias = csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale;
    let biased_depth = shadow_coords.z - bias;
    
    // PCF kernel size
    let kernel_size = i32(csm_uniforms.pcf_kernel_size);
    let half_kernel = kernel_size / 2;
    
    // Texel size for this cascade
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    
    // Accumulate shadow samples
    var shadow_factor = 0.0;
    var sample_count = 0.0;
    
    for (var x = -half_kernel; x <= half_kernel; x++) {
        for (var y = -half_kernel; y <= half_kernel; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords.xy + offset;
            
            // Bounds check for each sample
            if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 && 
                sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
                
                shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler, 
                                                    sample_coords, cascade_idx, biased_depth);
                sample_count += 1.0;
            }
        }
    }
    
    return shadow_factor / sample_count;
}

// Advanced PCF with Poisson disk sampling for better quality
fn sample_shadow_poisson_pcf(light_space_pos: vec4<f32>, cascade_idx: u32, world_normal: vec3<f32>) -> f32 {
    // Perspective divide and convert to texture coordinates
    let proj_coords = light_space_pos.xyz / light_space_pos.w;
    let shadow_coords = proj_coords * 0.5 + 0.5;
    
    // Check bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 || 
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0;
    }
    
    // Calculate bias
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = dot(world_normal, -light_dir);
    let slope_scale = sqrt(1.0 - n_dot_l * n_dot_l) / n_dot_l;
    let bias = csm_uniforms.depth_bias + csm_uniforms.slope_bias * slope_scale;
    let biased_depth = shadow_coords.z - bias;
    
    // Poisson disk samples for better distribution
    var poisson_disk = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216),
        vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870),
        vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432),
        vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845),
        vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554),
        vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023),
        vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507),
        vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367),
        vec2<f32>(0.14383161, -0.14100790)
    );
    
    // Sample using Poisson disk
    let texel_size = 1.0 / csm_uniforms.shadow_map_size;
    let filter_radius = f32(csm_uniforms.pcf_kernel_size) * texel_size * 0.5;
    
    var shadow_factor = 0.0;
    let sample_count = 16.0; // Using 16 Poisson samples
    
    for (var i = 0; i < 16; i++) {
        let offset = poisson_disk[i] * filter_radius;
        let sample_coords = shadow_coords.xy + offset;
        
        if (sample_coords.x >= 0.0 && sample_coords.x <= 1.0 && 
            sample_coords.y >= 0.0 && sample_coords.y <= 1.0) {
            
            shadow_factor += textureSampleCompare(shadow_maps, shadow_sampler, 
                                                sample_coords, cascade_idx, biased_depth);
        } else {
            shadow_factor += 1.0; // Outside bounds - not shadowed
        }
    }
    
    return shadow_factor / sample_count;
}

// Main shadow calculation function
fn calculate_shadow(world_pos: vec3<f32>, view_depth: f32, world_normal: vec3<f32>) -> f32 {
    // Skip shadow calculation if light doesn't cast shadows
    if (csm_uniforms.cascade_count == 0u) {
        return 1.0;
    }
    
    // Select appropriate cascade
    let cascade_idx = select_cascade(view_depth);
    
    // Transform to light space
    let light_space_pos = world_to_light_space(world_pos, cascade_idx);
    
    // Choose filtering method based on kernel size
    var shadow_factor: f32;
    if (csm_uniforms.pcf_kernel_size <= 1u) {
        // No filtering
        shadow_factor = sample_shadow_basic(light_space_pos, cascade_idx);
    } else if (csm_uniforms.pcf_kernel_size <= 5u) {
        // Standard PCF
        shadow_factor = sample_shadow_pcf(light_space_pos, cascade_idx, world_normal);
    } else {
        // High-quality Poisson PCF
        shadow_factor = sample_shadow_poisson_pcf(light_space_pos, cascade_idx, world_normal);
    }
    
    return shadow_factor;
}

// Debug visualization colors for cascades
fn get_cascade_debug_color(cascade_idx: u32) -> vec3<f32> {
    switch (cascade_idx) {
        case 0u: { return vec3<f32>(1.0, 0.0, 0.0); } // Red
        case 1u: { return vec3<f32>(0.0, 1.0, 0.0); } // Green  
        case 2u: { return vec3<f32>(0.0, 0.0, 1.0); } // Blue
        case 3u: { return vec3<f32>(1.0, 1.0, 0.0); } // Yellow
        default: { return vec3<f32>(1.0, 0.0, 1.0); } // Magenta
    }
}

// Apply debug cascade visualization
fn apply_debug_visualization(base_color: vec3<f32>, world_pos: vec3<f32>, view_depth: f32) -> vec3<f32> {
    if (csm_uniforms.debug_mode == 0u) {
        return base_color;
    }
    
    let cascade_idx = select_cascade(view_depth);
    let debug_color = get_cascade_debug_color(cascade_idx);
    
    // Blend base color with cascade debug color
    return mix(base_color, debug_color, 0.3);
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

// Standard vertex shader that includes shadow calculation
struct StandardVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct StandardVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_depth: f32,
}

// Requires camera uniforms to be bound at group(0) binding(0)
struct CameraUniforms {
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    view_projection: mat4x4<f32>,
    position: vec3<f32>,
    _padding: f32,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@vertex
fn standard_vs_main(input: StandardVertexInput) -> StandardVertexOutput {
    var out: StandardVertexOutput;
    
    // Transform to world space
    out.world_position = input.position; // Assuming input is already in world space
    out.world_normal = normalize(input.normal);
    out.uv = input.uv;
    
    // Transform to clip space
    out.clip_position = camera.view_projection * vec4<f32>(input.position, 1.0);
    
    // Calculate view depth for cascade selection
    let view_pos = camera.view * vec4<f32>(input.position, 1.0);
    out.view_depth = -view_pos.z; // Negative Z in view space
    
    return out;
}

// Standard fragment shader with shadow calculation
struct StandardFragmentOutput {
    @location(0) color: vec4<f32>,
}

@fragment
fn standard_fs_main(input: StandardVertexOutput) -> StandardFragmentOutput {
    var out: StandardFragmentOutput;
    
    // Base material color (white for demonstration)
    var base_color = vec3<f32>(0.8, 0.8, 0.8);
    
    // Calculate lighting
    let light_dir = normalize(csm_uniforms.light_direction.xyz);
    let n_dot_l = max(dot(input.world_normal, -light_dir), 0.0);
    
    // Calculate shadow
    let shadow_factor = calculate_shadow(input.world_position, input.view_depth, input.world_normal);
    
    // Apply lighting and shadows
    let ambient = vec3<f32>(0.1, 0.1, 0.1);
    let diffuse = base_color * n_dot_l * shadow_factor;
    let final_color = ambient + diffuse;
    
    // Apply debug visualization if enabled
    let debug_color = apply_debug_visualization(final_color, input.world_position, input.view_depth);
    
    out.color = vec4<f32>(debug_color, 1.0);
    return out;
}
