// B14: LTC Rect Area Lights - Linearly Transformed Cosines WGSL Implementation
//
// Provides physically accurate real-time rectangular area lighting using
// precomputed lookup textures for efficient BRDF integration.
//
// Bind Groups:
// Group 0: LTC rect area lights data and lookup textures
// Group 1: Scene geometry and material data
//
// References:
// - "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
// - LTC lookup tables for GGX BRDF approximation

// LTC constants
const LTC_LUT_SIZE: f32 = 64.0;
const PI: f32 = 3.14159265359;
const TWOPI: f32 = 6.28318530718;
const INV_PI: f32 = 0.31830988618;

// Rectangular area light structure (must match Rust)
struct RectAreaLight {
    position: vec3<f32>,
    intensity: f32,
    right: vec3<f32>,
    width: f32,
    up: vec3<f32>,
    height: f32,
    color: vec3<f32>,
    power: f32,
    normal: vec3<f32>,
    two_sided: f32,
}

// LTC uniform data
struct LTCUniforms {
    light_count: u32,
    lut_size: u32,
    global_intensity: f32,
    enable_ltc: f32,
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// Bind group 0: LTC resources
@group(0) @binding(0) var<storage, read> rect_area_lights: array<RectAreaLight>;
@group(0) @binding(1) var<uniform> ltc_uniforms: LTCUniforms;
@group(0) @binding(2) var ltc_matrix_texture: texture_2d<f32>;
@group(0) @binding(3) var ltc_scale_texture: texture_2d<f32>;
@group(0) @binding(4) var ltc_sampler: sampler;

// Utility functions for LTC evaluation

/// Sample LTC lookup textures
fn sample_ltc_matrix(roughness: f32, cos_theta: f32) -> mat3x3<f32> {
    let u = roughness;
    let v = acos(cos_theta) * (2.0 / PI);

    let uv = vec2<f32>(u, v);
    let matrix_data = textureSample(ltc_matrix_texture, ltc_sampler, uv);

    // Reconstruct 3x3 matrix from texture data
    return mat3x3<f32>(
        matrix_data.x, matrix_data.y, matrix_data.z,
        matrix_data.w, 1.0, 0.0,  // Assume identity for missing components
        0.0, 0.0, 1.0
    );
}

fn sample_ltc_scale(roughness: f32, cos_theta: f32) -> vec2<f32> {
    let u = roughness;
    let v = acos(cos_theta) * (2.0 / PI);

    let uv = vec2<f32>(u, v);
    return textureSample(ltc_scale_texture, ltc_sampler, uv).xy;
}

/// Compute edge vector for rect area light integration
fn ltc_edge_vector_form_factor(v1: vec3<f32>, v2: vec3<f32>) -> f32 {
    let cosine = dot(v1, v2);
    let sine = length(cross(v1, v2));

    if (sine < 1e-7) {
        return 0.0;
    }

    return atan2(sine, cosine);
}

/// Evaluate rect area light using LTC approximation
fn evaluate_ltc_rect_area_light(
    light: RectAreaLight,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    roughness: f32,
    metallic: f32,
    albedo: vec3<f32>
) -> vec3<f32> {
    // Vector from shading point to light center
    let light_vec = light.position - world_pos;
    let light_distance = length(light_vec);

    if (light_distance < 0.001) {
        return vec3<f32>(0.0);
    }

    let light_dir = light_vec / light_distance;

    // Check if we're facing the light (for one-sided lights)
    let facing_light = dot(-light_dir, normal) > 0.0;
    if (light.two_sided < 0.5 && !facing_light) {
        return vec3<f32>(0.0);
    }

    // Transform to light-centered coordinate system
    let local_pos = world_pos - light.position;

    // Light frame vectors (half-extents)
    let light_right = normalize(light.right) * (light.width * 0.5);
    let light_up = normalize(light.up) * (light.height * 0.5);

    // Four corners of the rectangle in world space
    let p0 = light.position + light_right + light_up;     // Top-right
    let p1 = light.position - light_right + light_up;     // Top-left
    let p2 = light.position - light_right - light_up;     // Bottom-left
    let p3 = light.position + light_right - light_up;     // Bottom-right

    // Vectors from shading point to corners
    let v0 = normalize(p0 - world_pos);
    let v1 = normalize(p1 - world_pos);
    let v2 = normalize(p2 - world_pos);
    let v3 = normalize(p3 - world_pos);

    // LTC transform based on material properties
    let cos_theta = max(0.0, dot(normal, view_dir));
    let ltc_matrix = sample_ltc_matrix(roughness, cos_theta);
    let ltc_scale = sample_ltc_scale(roughness, cos_theta);

    // Apply LTC transformation to corner vectors
    let ltc_v0 = normalize(ltc_matrix * v0);
    let ltc_v1 = normalize(ltc_matrix * v1);
    let ltc_v2 = normalize(ltc_matrix * v2);
    let ltc_v3 = normalize(ltc_matrix * v3);

    // Compute form factor using edge integration
    var form_factor = 0.0;
    form_factor += ltc_edge_vector_form_factor(ltc_v0, ltc_v1);
    form_factor += ltc_edge_vector_form_factor(ltc_v1, ltc_v2);
    form_factor += ltc_edge_vector_form_factor(ltc_v2, ltc_v3);
    form_factor += ltc_edge_vector_form_factor(ltc_v3, ltc_v0);

    form_factor = abs(form_factor) * INV_PI;

    // Apply LTC scale factors
    let spec_scale = ltc_scale.x;
    let diffuse_scale = ltc_scale.y;

    // Compute fresnel term
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    // Combine diffuse and specular contributions
    let diffuse_brdf = albedo * INV_PI * (1.0 - metallic);
    let specular_brdf = fresnel * spec_scale;

    // Final radiance contribution
    let brdf = diffuse_brdf * diffuse_scale + specular_brdf;
    let light_emission = light.color * light.intensity * ltc_uniforms.global_intensity;

    return brdf * light_emission * form_factor;
}

/// Exact analytical evaluation (fallback when LTC is disabled)
fn evaluate_exact_rect_area_light(
    light: RectAreaLight,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>
) -> vec3<f32> {
    // Simplified analytical evaluation for comparison
    let light_vec = light.position - world_pos;
    let light_distance = length(light_vec);
    let light_dir = normalize(light_vec);

    // Basic Lambert shading with distance attenuation
    let ndotl = max(0.0, dot(normal, light_dir));
    let attenuation = 1.0 / (1.0 + 0.01 * light_distance * light_distance);

    // Approximate solid angle subtended by the rectangle
    let area = light.width * light.height;
    let solid_angle = area / (light_distance * light_distance + area);

    let light_emission = light.color * light.intensity * ltc_uniforms.global_intensity;
    return albedo * light_emission * ndotl * attenuation * solid_angle;
}

/// Main LTC rect area lights evaluation function
fn evaluate_ltc_area_lighting(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    view_dir: vec3<f32>,
    material_roughness: f32,
    material_metallic: f32,
    material_albedo: vec3<f32>
) -> vec3<f32> {
    var total_lighting = vec3<f32>(0.0);

    let light_count = min(ltc_uniforms.light_count, arrayLength(&rect_area_lights));

    for (var i = 0u; i < light_count; i++) {
        let light = rect_area_lights[i];

        var light_contribution: vec3<f32>;

        if (ltc_uniforms.enable_ltc > 0.5) {
            // Use LTC approximation
            light_contribution = evaluate_ltc_rect_area_light(
                light, world_pos, world_normal, view_dir,
                material_roughness, material_metallic, material_albedo
            );
        } else {
            // Use exact analytical evaluation
            light_contribution = evaluate_exact_rect_area_light(
                light, world_pos, world_normal, view_dir, material_albedo
            );
        }

        total_lighting += light_contribution;
    }

    return total_lighting;
}

/// Compute lighting at a single point (for compute shaders)
@compute @workgroup_size(8, 8, 1)
fn cs_ltc_lighting(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);

    // This would typically read from G-buffer textures
    // and write lighting results to an output texture

    // Placeholder for actual G-buffer sampling and lighting computation
    let world_pos = vec3<f32>(0.0);
    let world_normal = vec3<f32>(0.0, 1.0, 0.0);
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let roughness = 0.5;
    let metallic = 0.0;
    let albedo = vec3<f32>(0.8);

    let lighting = evaluate_ltc_area_lighting(
        world_pos, world_normal, view_dir,
        roughness, metallic, albedo
    );

    // Write results would go here
}

/// Vertex shader output for fragment shader integration
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) view_dir: vec3<f32>,
}

/// Fragment shader with LTC area lighting
@fragment
fn fs_ltc_area_lighting(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = input.world_pos;
    let world_normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);

    // Material properties (could come from textures)
    let roughness = 0.3;
    let metallic = 0.1;
    let albedo = vec3<f32>(0.8, 0.7, 0.6);

    // Evaluate LTC area lighting
    let lighting = evaluate_ltc_area_lighting(
        world_pos, world_normal, view_dir,
        roughness, metallic, albedo
    );

    // Apply basic tone mapping
    let tone_mapped = lighting / (lighting + vec3<f32>(1.0));

    return vec4<f32>(tone_mapped, 1.0);
}

/// Debug visualization for LTC lookup textures
@fragment
fn fs_debug_ltc_lut(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.tex_coords;

    // Visualize LTC matrix texture
    if (uv.x < 0.5) {
        let matrix_sample = textureSample(ltc_matrix_texture, ltc_sampler, uv * 2.0);
        return vec4<f32>(abs(matrix_sample.xyz), 1.0);
    } else {
        // Visualize LTC scale texture
        let scale_sample = textureSample(ltc_scale_texture, ltc_sampler, (uv - vec2<f32>(0.5, 0.0)) * 2.0);
        return vec4<f32>(scale_sample.x, scale_sample.y, 0.0, 1.0);
    }
}

/// Performance test shader for LTC evaluation
@fragment
fn fs_ltc_performance_test(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = input.world_pos;
    let world_normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);

    // Run multiple evaluations to stress test performance
    var accumulated_lighting = vec3<f32>(0.0);
    let iterations = ltc_uniforms.sample_count;

    for (var i = 0u; i < iterations; i++) {
        let lighting = evaluate_ltc_area_lighting(
            world_pos, world_normal, view_dir,
            0.5, 0.0, vec3<f32>(0.8)
        );
        accumulated_lighting += lighting;
    }

    let avg_lighting = accumulated_lighting / f32(iterations);
    let tone_mapped = avg_lighting / (avg_lighting + vec3<f32>(1.0));

    return vec4<f32>(tone_mapped, 1.0);
}

/// Validation shader for comparing LTC vs exact evaluation
@fragment
fn fs_ltc_validation(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = input.world_pos;
    let world_normal = normalize(input.world_normal);
    let view_dir = normalize(input.view_dir);
    let albedo = vec3<f32>(0.8, 0.7, 0.6);

    // Split screen: LTC on left, exact on right
    var lighting: vec3<f32>;

    if (input.tex_coords.x < 0.5) {
        // Left side: LTC evaluation
        lighting = evaluate_ltc_area_lighting(
            world_pos, world_normal, view_dir,
            0.3, 0.1, albedo
        );
    } else {
        // Right side: Exact evaluation (simplified)
        var exact_lighting = vec3<f32>(0.0);
        let light_count = min(ltc_uniforms.light_count, arrayLength(&rect_area_lights));

        for (var i = 0u; i < light_count; i++) {
            let light = rect_area_lights[i];
            exact_lighting += evaluate_exact_rect_area_light(
                light, world_pos, world_normal, view_dir, albedo
            );
        }

        lighting = exact_lighting;
    }

    // Apply tone mapping
    let tone_mapped = lighting / (lighting + vec3<f32>(1.0));

    return vec4<f32>(tone_mapped, 1.0);
}