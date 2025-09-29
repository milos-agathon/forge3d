// Planar Reflections - Main Implementation
// Real-time planar reflections with roughness-aware blur and clip plane support (B5)
// RELEVANT FILES: src/core/reflections.rs, python/forge3d/lighting.py, tests/test_b5_reflections.py

// Reflection plane data
struct ReflectionPlane {
    /// Plane equation coefficients (ax + by + cz + d = 0)
    plane_equation: vec4<f32>,
    /// Reflection matrix for transforming geometry
    reflection_matrix: mat4x4<f32>,
    /// View matrix for reflection camera
    reflection_view: mat4x4<f32>,
    /// Projection matrix for reflection camera
    reflection_projection: mat4x4<f32>,
    /// World position of reflection plane center
    plane_center: vec4<f32>,
    /// Plane dimensions (width, height, 0, 0)
    plane_size: vec4<f32>,
}

// Planar reflection configuration
struct PlanarReflectionUniforms {
    /// Reflection plane data
    reflection_plane: ReflectionPlane,
    /// Enable/disable reflection rendering
    enable_reflections: u32,
    /// Reflection intensity [0, 1]
    reflection_intensity: f32,
    /// Fresnel power for reflection falloff
    fresnel_power: f32,
    /// Blur kernel size for roughness
    blur_kernel_size: u32,
    /// Maximum blur radius in texels
    max_blur_radius: f32,
    /// Reflection texture resolution
    reflection_resolution: f32,
    /// Distance fade start/end
    distance_fade_start: f32,
    distance_fade_end: f32,
    /// Debug visualization mode
    debug_mode: u32,
    /// Padding for alignment
    _padding: vec2<f32>,
}

// Bind groups for planar reflections
@group(3) @binding(0) var<uniform> reflection_uniforms: PlanarReflectionUniforms;
@group(3) @binding(1) var reflection_texture: texture_2d<f32>;
@group(3) @binding(2) var reflection_sampler: sampler;
@group(3) @binding(3) var reflection_depth: texture_depth_2d;

// Constants
const PI: f32 = 3.14159265359;
const REFLECTION_EPSILON: f32 = 0.001;
const MAX_REFLECTION_DISTANCE: f32 = 1000.0;

// Calculate distance from point to plane
fn distance_to_plane(world_pos: vec3<f32>, plane: vec4<f32>) -> f32 {
    return dot(vec4<f32>(world_pos, 1.0), plane);
}

// Check if point is above reflection plane
fn is_above_plane(world_pos: vec3<f32>) -> bool {
    let distance = distance_to_plane(world_pos, reflection_uniforms.reflection_plane.plane_equation);
    return distance > REFLECTION_EPSILON;
}

// Calculate reflection vector for a point relative to the plane
fn calculate_reflection_vector(incident: vec3<f32>, plane_normal: vec3<f32>) -> vec3<f32> {
    return incident - 2.0 * dot(incident, plane_normal) * plane_normal;
}

// Transform world position to reflection space
fn world_to_reflection_space(world_pos: vec3<f32>) -> vec4<f32> {
    let reflection_view_pos = reflection_uniforms.reflection_plane.reflection_view * vec4<f32>(world_pos, 1.0);
    return reflection_uniforms.reflection_plane.reflection_projection * reflection_view_pos;
}

// Sample reflection texture at given UV coordinates
fn sample_reflection_basic(uv: vec2<f32>) -> vec4<f32> {
    // Clamp UV to avoid sampling outside texture bounds
    let clamped_uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));
    return textureSample(reflection_texture, reflection_sampler, clamped_uv);
}

// Fresnel calculation for reflection intensity
fn calculate_fresnel(view_dir: vec3<f32>, surface_normal: vec3<f32>, fresnel_power: f32) -> f32 {
    let n_dot_v = max(0.0, dot(surface_normal, view_dir));
    let fresnel = pow(1.0 - n_dot_v, fresnel_power);
    return clamp(fresnel, 0.0, 1.0);
}

// Calculate blur radius based on surface roughness
fn calculate_blur_radius(roughness: f32, distance_to_camera: f32) -> f32 {
    let base_blur = roughness * reflection_uniforms.max_blur_radius;
    let distance_factor = min(1.0, distance_to_camera / 100.0); // Increase blur with distance
    return base_blur * (1.0 + distance_factor);
}

// Sample reflection with roughness-aware blur
fn sample_reflection_with_blur(uv: vec2<f32>, blur_radius: f32) -> vec4<f32> {
    if blur_radius < 0.5 {
        // No blur needed for sharp reflections
        return sample_reflection_basic(uv);
    }

    // Gaussian-like blur using compact kernel (half-kernel extents)
    let texel_size = 1.0 / reflection_uniforms.reflection_resolution;
    let kernel_size = min(reflection_uniforms.blur_kernel_size, 7u);
    let half_kernel: i32 = i32(kernel_size) / 2;

    var blur_result = vec4<f32>(0.0);
    var weight_sum = 0.0;

    // Sample compact square neighborhood for performance
    for (var i = -half_kernel; i <= half_kernel; i++) {
        for (var j = -half_kernel; j <= half_kernel; j++) {
            let offset = vec2<f32>(f32(i), f32(j)) * texel_size * blur_radius;
            let sample_uv = uv + offset;

            // Check if sample is within bounds
            if sample_uv.x >= 0.0 && sample_uv.x <= 1.0 &&
               sample_uv.y >= 0.0 && sample_uv.y <= 1.0 {

                let distance = length(offset);
                let weight = exp(-distance * distance / (blur_radius * blur_radius * 0.5));

                blur_result += textureSample(reflection_texture, reflection_sampler, sample_uv) * weight;
                weight_sum += weight;
            }
        }
    }

    if weight_sum > 0.001 {
        return blur_result / weight_sum;
    } else {
        return sample_reflection_basic(uv);
    }
}

// Advanced Poisson disk blur for high-quality reflections
fn sample_reflection_poisson_blur(uv: vec2<f32>, blur_radius: f32) -> vec4<f32> {
    if blur_radius < 0.5 {
        return sample_reflection_basic(uv);
    }

    // Poisson disk sample pattern
    let poisson_samples = array<vec2<f32>, 16>(
        vec2<f32>(-0.94201624, -0.39906216), vec2<f32>(0.94558609, -0.76890725),
        vec2<f32>(-0.094184101, -0.92938870), vec2<f32>(0.34495938, 0.29387760),
        vec2<f32>(-0.91588581, 0.45771432), vec2<f32>(-0.81544232, -0.87912464),
        vec2<f32>(-0.38277543, 0.27676845), vec2<f32>(0.97484398, 0.75648379),
        vec2<f32>(0.44323325, -0.97511554), vec2<f32>(0.53742981, -0.47373420),
        vec2<f32>(-0.26496911, -0.41893023), vec2<f32>(0.79197514, 0.19090188),
        vec2<f32>(-0.24188840, 0.99706507), vec2<f32>(-0.81409955, 0.91437590),
        vec2<f32>(0.19984126, 0.78641367), vec2<f32>(0.14383161, -0.14100790)
    );

    let texel_size = 1.0 / reflection_uniforms.reflection_resolution;
    let sample_radius = blur_radius * texel_size;

    var blur_result = vec4<f32>(0.0);
    var sample_count = 0.0;

    for (var i = 0; i < 16; i++) {
        let offset = poisson_samples[i] * sample_radius;
        let sample_uv = uv + offset;

        if sample_uv.x >= 0.0 && sample_uv.x <= 1.0 &&
           sample_uv.y >= 0.0 && sample_uv.y <= 1.0 {
            blur_result += textureSample(reflection_texture, reflection_sampler, sample_uv);
            sample_count += 1.0;
        }
    }

    if sample_count > 0.0 {
        return blur_result / sample_count;
    } else {
        return sample_reflection_basic(uv);
    }
}

// Calculate distance-based fade for reflections
fn calculate_distance_fade(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> f32 {
    let distance = length(world_pos - camera_pos);
    let fade_range = reflection_uniforms.distance_fade_end - reflection_uniforms.distance_fade_start;

    if fade_range <= 0.0 {
        return 1.0; // No fade
    }

    let fade_factor = (reflection_uniforms.distance_fade_end - distance) / fade_range;
    return clamp(fade_factor, 0.0, 1.0);
}

// Main reflection calculation function
fn calculate_planar_reflection(
    world_pos: vec3<f32>,
    view_dir: vec3<f32>,
    surface_normal: vec3<f32>,
    roughness: f32,
    camera_pos: vec3<f32>
) -> vec4<f32> {
    // Early exit if reflections are disabled
    if reflection_uniforms.enable_reflections == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Check if surface is facing away from the reflection plane
    if !is_above_plane(world_pos) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Transform world position to reflection clip space
    let reflection_clip_pos = world_to_reflection_space(world_pos);

    // Perspective divide to get NDC coordinates
    if abs(reflection_clip_pos.w) < REFLECTION_EPSILON {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    let reflection_ndc = reflection_clip_pos.xyz / reflection_clip_pos.w;

    // Convert NDC to UV coordinates
    let reflection_uv = reflection_ndc.xy * 0.5 + 0.5;

    // Check if reflection UV is within valid range
    if reflection_uv.x < 0.0 || reflection_uv.x > 1.0 ||
       reflection_uv.y < 0.0 || reflection_uv.y > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }

    // Calculate blur radius based on roughness and distance
    let distance_to_camera = length(world_pos - camera_pos);
    let blur_radius = calculate_blur_radius(roughness, distance_to_camera);

    // Sample reflection texture with appropriate blur
    var reflection_color: vec4<f32>;
    if blur_radius > 3.0 {
        // Use Poisson disk for high-quality blur
        reflection_color = sample_reflection_poisson_blur(reflection_uv, blur_radius);
    } else {
        // Use standard blur for lower roughness
        reflection_color = sample_reflection_with_blur(reflection_uv, blur_radius);
    }

    // Calculate Fresnel factor
    let fresnel = calculate_fresnel(view_dir, surface_normal, reflection_uniforms.fresnel_power);

    // Calculate distance fade
    let distance_fade = calculate_distance_fade(world_pos, camera_pos);

    // Apply reflection intensity, Fresnel, and distance fade
    let final_intensity = reflection_uniforms.reflection_intensity * fresnel * distance_fade;

    return vec4<f32>(reflection_color.rgb, final_intensity);
}

// Screen-space reflection UV calculation (alternative method)
fn calculate_ssr_uv(world_pos: vec3<f32>, view_matrix: mat4x4<f32>, projection_matrix: mat4x4<f32>) -> vec2<f32> {
    let view_pos = view_matrix * vec4<f32>(world_pos, 1.0);
    let clip_pos = projection_matrix * view_pos;
    let ndc = clip_pos.xy / clip_pos.w;
    return ndc * 0.5 + 0.5;
}

// Blend reflection with base color
fn blend_reflection(base_color: vec3<f32>, reflection: vec4<f32>) -> vec3<f32> {
    return mix(base_color, reflection.rgb, reflection.a);
}

// Debug visualization modes
fn apply_reflection_debug(base_color: vec3<f32>, world_pos: vec3<f32>, reflection: vec4<f32>) -> vec3<f32> {
    if reflection_uniforms.debug_mode == 0u {
        return base_color;
    }

    switch reflection_uniforms.debug_mode {
        case 1u: {
            // Show reflection UV coordinates
            let clip_pos = world_to_reflection_space(world_pos);
            let ndc = clip_pos.xyz / clip_pos.w;
            let uv = ndc.xy * 0.5 + 0.5;
            return vec3<f32>(uv, 0.0);
        }
        case 2u: {
            // Show reflection intensity
            return vec3<f32>(reflection.a, reflection.a, reflection.a);
        }
        case 3u: {
            // Show distance to plane
            let distance = abs(distance_to_plane(world_pos, reflection_uniforms.reflection_plane.plane_equation));
            let normalized_distance = min(1.0, distance / 10.0);
            return vec3<f32>(normalized_distance, 0.0, 1.0 - normalized_distance);
        }
        case 4u: {
            // Show blur radius visualization
            let distance_to_camera = length(world_pos); // Assuming camera at origin for debug
            let blur_radius = calculate_blur_radius(0.5, distance_to_camera); // Mid roughness
            let normalized_blur = min(1.0, blur_radius / reflection_uniforms.max_blur_radius);
            return vec3<f32>(normalized_blur, 1.0 - normalized_blur, 0.0);
        }
        default: {
            return base_color;
        }
    }
}

// Vertex shader for reflection rendering (during reflection pass)
struct ReflectionVertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct ReflectionVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) clip_distance: f32,
}

@vertex
fn reflection_vs_main(input: ReflectionVertexInput) -> ReflectionVertexOutput {
    var output: ReflectionVertexOutput;

    // Transform vertex to world space
    output.world_position = input.position;
    output.world_normal = normalize(input.normal);
    output.uv = input.uv;

    // Apply reflection transformation
    let reflected_pos = reflection_uniforms.reflection_plane.reflection_matrix * vec4<f32>(input.position, 1.0);

    // Transform to clip space using reflection view/projection
    let view_pos = reflection_uniforms.reflection_plane.reflection_view * reflected_pos;
    output.clip_position = reflection_uniforms.reflection_plane.reflection_projection * view_pos;

    // Calculate clip distance for plane clipping
    output.clip_distance = distance_to_plane(reflected_pos.xyz, reflection_uniforms.reflection_plane.plane_equation);

    return output;
}

// Fragment shader for reflection rendering
@fragment
fn reflection_fs_main(input: ReflectionVertexOutput) -> @location(0) vec4<f32> {
    // Clip fragments below the reflection plane
    if input.clip_distance < 0.0 {
        discard;
    }

    // Simple shading for reflection pass
    let light_dir = normalize(vec3<f32>(0.0, 1.0, 0.5));
    let n_dot_l = max(0.0, dot(input.world_normal, light_dir));
    let diffuse = vec3<f32>(0.8, 0.8, 0.8) * n_dot_l + vec3<f32>(0.2, 0.2, 0.2);

    return vec4<f32>(diffuse, 1.0);
}

// Fragment shader integration for main rendering pass
fn integrate_planar_reflections(
    base_color: vec3<f32>,
    world_pos: vec3<f32>,
    view_dir: vec3<f32>,
    surface_normal: vec3<f32>,
    roughness: f32,
    camera_pos: vec3<f32>
) -> vec3<f32> {
    // Calculate reflection
    let reflection = calculate_planar_reflection(
        world_pos, view_dir, surface_normal, roughness, camera_pos
    );

    // Apply debug visualization if enabled
    if reflection_uniforms.debug_mode > 0u {
        return apply_reflection_debug(base_color, world_pos, reflection);
    }

    // Blend reflection with base color
    return blend_reflection(base_color, reflection);
}

// Performance optimization: LOD selection for reflections
fn should_render_reflection(world_pos: vec3<f32>, camera_pos: vec3<f32>) -> bool {
    let distance = length(world_pos - camera_pos);
    return distance <= reflection_uniforms.distance_fade_end;
}

// Utility function for reflection matrix calculation (used in CPU setup)
fn create_reflection_matrix(plane_normal: vec3<f32>, plane_distance: f32) -> mat4x4<f32> {
    let n = normalize(plane_normal);
    let d = plane_distance;

    return mat4x4<f32>(
        1.0 - 2.0 * n.x * n.x, -2.0 * n.x * n.y, -2.0 * n.x * n.z, -2.0 * n.x * d,
        -2.0 * n.y * n.x, 1.0 - 2.0 * n.y * n.y, -2.0 * n.y * n.z, -2.0 * n.y * d,
        -2.0 * n.z * n.x, -2.0 * n.z * n.y, 1.0 - 2.0 * n.z * n.z, -2.0 * n.z * d,
        0.0, 0.0, 0.0, 1.0
    );
}