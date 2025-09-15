// Soft area lights with penumbra control shader
//
// Bind Groups:
// Group 0: Area lights data and parameters
// Group 1: Scene geometry and materials
//
// Formats:
// - Light data: Custom struct (see AreaLight in Rust)
// - Shadow maps: R32_FLOAT depth textures
//
// Workgroup size: 8x8 for compute shaders, N/A for fragment

// Area light types
const LIGHT_RECTANGLE: u32 = 0u;
const LIGHT_DISC: u32 = 1u;
const LIGHT_SPHERE: u32 = 2u;
const LIGHT_CYLINDER: u32 = 3u;

// Area light structure (must match Rust struct)
struct AreaLight {
    position: vec3<f32>,
    light_type: u32,
    direction: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
    size: vec4<f32>,        // width, height, depth, unused
    softness: f32,
    energy_factor: f32,
    shadow_bias: f32,
    _padding: f32,
}

// Lighting parameters
struct LightingParams {
    light_count: u32,
    max_samples: u32,
    shadow_samples: u32,
    global_intensity: f32,
}

// Bind group 0: Lighting data
@group(0) @binding(0) var<storage, read> area_lights: array<AreaLight>;
@group(0) @binding(1) var<uniform> lighting_params: LightingParams;

// Bind group 1: Shadow sampling
@group(1) @binding(0) var shadow_sampler: sampler_comparison;
@group(1) @binding(1) var shadow_maps: texture_depth_2d_array;

// Random number generation for sampling
var<private> rng_state: u32;

fn init_rng(seed: u32) {
    rng_state = seed;
}

fn rng_next() -> u32 {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

fn rng_float() -> f32 {
    return f32(rng_next()) / 4294967296.0;
}

// Sample point on area light surface
fn sample_area_light(light: AreaLight, random1: f32, random2: f32) -> vec3<f32> {
    let light_type = light.light_type;

    if (light_type == LIGHT_RECTANGLE) {
        // Sample rectangle
        let u = random1 - 0.5;
        let v = random2 - 0.5;
        let local_pos = vec3<f32>(u * light.size.x, 0.0, v * light.size.y);

        // Transform to world space (simplified - assumes light faces down)
        return light.position + local_pos;

    } else if (light_type == LIGHT_DISC) {
        // Sample disc using concentric mapping
        let r = sqrt(random1) * light.size.x;
        let theta = random2 * 2.0 * 3.14159265;
        let local_pos = vec3<f32>(r * cos(theta), 0.0, r * sin(theta));

        return light.position + local_pos;

    } else if (light_type == LIGHT_SPHERE) {
        // Sample sphere surface
        let z = 1.0 - 2.0 * random1;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.0 * 3.14159265 * random2;
        let local_pos = vec3<f32>(r * cos(phi), r * sin(phi), z) * light.size.x;

        return light.position + local_pos;

    } else {
        // Default to point light behavior
        return light.position;
    }
}

// Calculate soft shadow with multiple samples
fn calculate_soft_shadow(world_pos: vec3<f32>, light: AreaLight, light_index: u32) -> f32 {
    let shadow_samples = max(1u, lighting_params.shadow_samples);
    var shadow_sum = 0.0;

    // Use light radius to control penumbra size
    let penumbra_scale = light.radius * light.softness;

    for (var i = 0u; i < shadow_samples; i++) {
        let random_seed = u32(world_pos.x * 1000.0) + u32(world_pos.z * 1000.0) + i * 73u;
        init_rng(random_seed);

        let random1 = rng_float();
        let random2 = rng_float();

        // Sample point on light surface with penumbra offset
        var light_sample_pos = sample_area_light(light, random1, random2);

        // Add penumbra variation
        let penumbra_offset = vec3<f32>(
            (rng_float() - 0.5) * penumbra_scale,
            (rng_float() - 0.5) * penumbra_scale,
            (rng_float() - 0.5) * penumbra_scale
        );
        light_sample_pos += penumbra_offset;

        // Calculate shadow ray
        let light_dir = normalize(light_sample_pos - world_pos);
        let light_distance = length(light_sample_pos - world_pos);

        // Simple shadow test (in real implementation, would trace ray)
        // For now, assume no occlusion for testing
        let shadow_factor = 1.0;

        shadow_sum += shadow_factor;
    }

    return shadow_sum / f32(shadow_samples);
}

// Calculate area light contribution
fn calculate_area_light_contribution(world_pos: vec3<f32>, world_normal: vec3<f32>,
                                   view_dir: vec3<f32>, light: AreaLight,
                                   light_index: u32) -> vec3<f32> {
    let light_dir = normalize(light.position - world_pos);
    let light_distance = length(light.position - world_pos);

    // Basic Lambert shading
    let ndotl = max(0.0, dot(world_normal, light_dir));

    if (ndotl <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Attenuation with distance
    let attenuation = 1.0 / (1.0 + 0.01 * light_distance + 0.001 * light_distance * light_distance);

    // Shadow factor with soft penumbra
    let shadow_factor = calculate_soft_shadow(world_pos, light, light_index);

    // Energy normalization
    let energy = light.intensity * light.energy_factor * lighting_params.global_intensity;

    // Final contribution
    return light.color * energy * ndotl * attenuation * shadow_factor;
}

// Main area lighting function
fn evaluate_area_lighting(world_pos: vec3<f32>, world_normal: vec3<f32>,
                         view_dir: vec3<f32>, material_albedo: vec3<f32>) -> vec3<f32> {
    var total_lighting = vec3<f32>(0.0);

    let light_count = min(lighting_params.light_count, arrayLength(&area_lights));

    for (var i = 0u; i < light_count; i++) {
        let light = area_lights[i];
        let light_contribution = calculate_area_light_contribution(
            world_pos, world_normal, view_dir, light, i
        );

        total_lighting += light_contribution;
    }

    return total_lighting * material_albedo;
}

// Compute shader for area light preprocessing
@compute @workgroup_size(8, 8, 1)
fn cs_precompute_area_lights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let light_index = global_id.x;

    if (light_index >= arrayLength(&area_lights)) {
        return;
    }

    let light = area_lights[light_index];

    // Validate light parameters
    if (light.radius <= 0.0 || light.intensity <= 0.0) {
        return;
    }

    // Could compute light space matrices, shadow map parameters, etc.
    // For now, this is a placeholder for more complex preprocessing
}

// Fragment shader integration (example usage)
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

@fragment
fn fs_area_lighting(input: VertexOutput) -> @location(0) vec4<f32> {
    let world_pos = input.world_pos;
    let world_normal = normalize(input.world_normal);
    let view_dir = normalize(-world_pos); // Assumes camera at origin

    // Material properties (could come from textures)
    let material_albedo = vec3<f32>(0.8, 0.8, 0.8);

    // Evaluate area lighting
    let lighting = evaluate_area_lighting(world_pos, world_normal, view_dir, material_albedo);

    // Tone mapping (simple)
    let toneMapped = lighting / (lighting + vec3<f32>(1.0));

    return vec4<f32>(toneMapped, 1.0);
}

// Energy validation compute shader
@compute @workgroup_size(64, 1, 1)
fn cs_validate_energy_conservation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    if (thread_id == 0u) {
        var total_energy = 0.0;
        let light_count = min(lighting_params.light_count, arrayLength(&area_lights));

        for (var i = 0u; i < light_count; i++) {
            let light = area_lights[i];
            total_energy += light.intensity * light.energy_factor;
        }

        // Store result somewhere for validation
        // In a real implementation, would write to a buffer
    }
}