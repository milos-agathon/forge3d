//! B16: Dual-source blending Order Independent Transparency
//! High-quality OIT using dual-source color blending for supported hardware

// Dual-source blending uniforms
struct DualSourceOITUniforms {
    alpha_correction: f32,      // Alpha correction factor
    depth_weight_scale: f32,    // Depth-based weight scaling
    max_fragments: f32,         // Maximum expected fragments per pixel
    premultiply_factor: f32,    // Premultiplication factor
}

@group(0) @binding(0)
var<uniform> dual_oit_uniforms: DualSourceOITUniforms;

// Standard vertex input for transparency
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

// Standard vertex output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) view_depth: f32,
}

// Dual-source fragment output
struct DualSourceOutput {
    @location(0) color0: vec4<f32>,    // Premultiplied color and alpha
    @location(1) color1: vec4<f32>,    // Alpha and weight information
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Transform position to clip space
    out.clip_position = vec4<f32>(input.position, 1.0);
    out.world_position = input.position;
    out.world_normal = normalize(input.normal);
    out.uv = input.uv;

    // Calculate view-space depth for weight computation
    out.view_depth = out.clip_position.z / out.clip_position.w;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> DualSourceOutput {
    var out: DualSourceOutput;

    // Sample material properties (simplified for demo)
    let base_color = vec3<f32>(0.5, 0.7, 0.9);  // Demo blue color
    let alpha = 0.6;  // Demo transparency

    // Calculate depth-based weight for better sorting
    let depth_weight = calculate_dual_source_weight(input.view_depth, alpha);

    // Apply alpha correction for better blending
    let corrected_alpha = pow(alpha, dual_oit_uniforms.alpha_correction);

    // Dual-source blending technique:
    // Color0: Premultiplied color with corrected alpha
    // Color1: Alpha and weight for proper blending

    // Color output 0: Premultiplied color
    let premultiplied_color = base_color * corrected_alpha * dual_oit_uniforms.premultiply_factor;
    out.color0 = vec4<f32>(premultiplied_color, corrected_alpha);

    // Color output 1: Alpha and weight information
    // .r = alpha for blending
    // .g = depth weight
    // .b = coverage (for anti-aliasing)
    // .a = fragment count contribution
    let coverage = 1.0;  // Full coverage for solid fragments
    let fragment_contribution = 1.0 / dual_oit_uniforms.max_fragments;

    out.color1 = vec4<f32>(
        corrected_alpha,
        depth_weight,
        coverage,
        fragment_contribution
    );

    return out;
}

// Calculate depth-based weight for dual-source OIT
fn calculate_dual_source_weight(depth: f32, alpha: f32) -> f32 {
    // Improved weight function for dual-source blending
    // Combines depth and alpha for better visual quality

    // Normalize depth to [0, 1] range
    let normalized_depth = clamp(depth * 0.5 + 0.5, 0.0, 1.0);

    // Calculate base weight using depth
    let depth_factor = 1.0 - normalized_depth;
    let depth_weight = pow(depth_factor, 2.0) * dual_oit_uniforms.depth_weight_scale;

    // Combine with alpha for final weight
    let final_weight = alpha * depth_weight;

    return clamp(final_weight, 0.001, 1000.0);
}

// Alternative fragment shader for terrain transparency
@fragment
fn fs_terrain(input: VertexOutput) -> DualSourceOutput {
    var out: DualSourceOutput;

    // Sample terrain height and calculate alpha based on elevation
    let height_factor = input.world_position.y;
    let terrain_alpha = smoothstep(0.0, 1.0, height_factor) * 0.7;

    // Terrain color based on height
    let terrain_color = mix(
        vec3<f32>(0.2, 0.5, 0.1),  // Low elevation (green)
        vec3<f32>(0.6, 0.4, 0.2),  // High elevation (brown)
        height_factor
    );

    // Calculate depth weight
    let depth_weight = calculate_dual_source_weight(input.view_depth, terrain_alpha);

    // Apply alpha correction
    let corrected_alpha = pow(terrain_alpha, dual_oit_uniforms.alpha_correction);

    // Output dual-source colors
    let premultiplied_color = terrain_color * corrected_alpha * dual_oit_uniforms.premultiply_factor;
    out.color0 = vec4<f32>(premultiplied_color, corrected_alpha);

    let coverage = 1.0;
    let fragment_contribution = 1.0 / dual_oit_uniforms.max_fragments;

    out.color1 = vec4<f32>(
        corrected_alpha,
        depth_weight,
        coverage,
        fragment_contribution
    );

    return out;
}

// Utility functions for dual-source OIT

// Calculate blend factors for dual-source equation
fn calculate_dual_source_blend_factors(alpha0: f32, alpha1: f32, weight: f32) -> vec2<f32> {
    // Advanced blending factors for dual-source OIT
    let src_factor = alpha0 * weight;
    let dst_factor = 1.0 - alpha1;

    return vec2<f32>(src_factor, dst_factor);
}

// Color space conversion for better blending
fn prepare_color_for_blending(color: vec3<f32>, alpha: f32) -> vec3<f32> {
    // Convert to perceptual space for better transparency blending
    let perceptual_color = pow(color, vec3<f32>(2.2));  // Linear to sRGB-like
    return perceptual_color * alpha;
}

// Anti-aliasing support for transparent edges
fn calculate_edge_coverage(distance_to_edge: f32, pixel_size: f32) -> f32 {
    // Smooth coverage based on distance to fragment edge
    let edge_width = pixel_size * 0.5;
    return smoothstep(-edge_width, edge_width, distance_to_edge);
}

// Quality assessment for dual-source OIT
fn dual_source_quality_metric(fragment_count: f32, max_expected: f32) -> f32 {
    // Return quality metric for adaptive behavior
    return clamp(fragment_count / max_expected, 0.0, 1.0);
}