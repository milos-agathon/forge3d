// src/shaders/restir_spatial.wgsl
// ReSTIR spatial reuse implementation

#include "restir_reservoir.wgsl"

// Spatial reservoir buffers
@group(2) @binding(0) var<storage, read> input_reservoirs: array<Reservoir>;
@group(2) @binding(1) var<storage, read_write> output_reservoirs: array<Reservoir>;
@group(2) @binding(2) var<storage, read> depth_buffer: array<f32>;
@group(2) @binding(3) var<storage, read> normal_buffer: array<vec3f>;
@group(2) @binding(4) var<storage, read> world_pos_buffer: array<vec3f>;

struct SpatialParams {
    screen_width: u32,
    screen_height: u32,
    spatial_radius: f32,
    num_spatial_samples: u32,
    depth_threshold: f32,
    normal_threshold: f32,
    spatial_bias_correction: u32,
    pass_index: u32,
}

@group(2) @binding(5) var<uniform> spatial_params: SpatialParams;

// Convert screen coordinates to buffer index
fn coord_to_index_spatial(coord: vec2u) -> u32 {
    return coord.y * spatial_params.screen_width + coord.x;
}

// Check if spatial neighbor is valid for reuse
fn is_spatial_neighbor_valid(
    center_coord: vec2u,
    neighbor_coord: vec2u,
    center_depth: f32,
    center_normal: vec3f,
    center_world_pos: vec3f
) -> bool {
    // Check bounds
    if (neighbor_coord.x >= spatial_params.screen_width || neighbor_coord.y >= spatial_params.screen_height) {
        return false;
    }

    let neighbor_index = coord_to_index_spatial(neighbor_coord);

    // Check depth similarity
    let neighbor_depth = depth_buffer[neighbor_index];
    if (neighbor_depth <= 0.0) {
        return false; // Background pixel
    }

    let depth_diff = abs(center_depth - neighbor_depth) / max(center_depth, neighbor_depth);
    if (depth_diff > spatial_params.depth_threshold) {
        return false;
    }

    // Check normal similarity
    let neighbor_normal = normal_buffer[neighbor_index];
    let normal_similarity = dot(center_normal, neighbor_normal);
    if (normal_similarity < spatial_params.normal_threshold) {
        return false;
    }

    // Check world space distance (optional additional constraint)
    let neighbor_world_pos = world_pos_buffer[neighbor_index];
    let world_distance = length(center_world_pos - neighbor_world_pos);
    if (world_distance > spatial_params.spatial_radius * 0.1) { // Scale factor
        return false;
    }

    return true;
}

// Calculate Jacobian for spatial reuse
fn calculate_spatial_jacobian(
    center_shading_point: vec3f,
    center_normal: vec3f,
    neighbor_shading_point: vec3f,
    neighbor_normal: vec3f,
    light_sample: LightSample
) -> f32 {
    // Calculate target PDFs for both positions
    let center_pdf = calculate_target_pdf(light_sample, center_shading_point, center_normal);
    let neighbor_pdf = calculate_target_pdf(light_sample, neighbor_shading_point, neighbor_normal);

    if (neighbor_pdf <= 0.0) {
        return 0.0;
    }

    return center_pdf / neighbor_pdf;
}

// Generate spatial neighbor offset using blue noise or structured pattern
fn get_spatial_offset(sample_index: u32, rand_state: ptr<function, u32>) -> vec2f {
    // Simple spiral pattern with randomization
    let angle = f32(sample_index) * 2.399963 + rand_xorshift(rand_state) * 6.283185; // Golden angle + random
    let radius = sqrt(f32(sample_index) + rand_xorshift(rand_state)) * spatial_params.spatial_radius;

    return vec2f(cos(angle) * radius, sin(angle) * radius);
}

// Spatial reuse compute kernel
@compute @workgroup_size(8, 8)
fn spatial_reuse(@builtin(global_invocation_id) global_id: vec3u) {
    let coord = global_id.xy;
    if (coord.x >= spatial_params.screen_width || coord.y >= spatial_params.screen_height) {
        return;
    }

    let center_index = coord_to_index_spatial(coord);

    // Get center pixel data
    let center_depth = depth_buffer[center_index];
    let center_normal = normal_buffer[center_index];
    let center_world_pos = world_pos_buffer[center_index];

    // Skip background pixels
    if (center_depth <= 0.0) {
        output_reservoirs[center_index] = init_reservoir();
        return;
    }

    // Initialize random state (different seed for each pass to avoid correlation)
    var rand_state = center_index * 2654435761u + spatial_params.pass_index * 1973u + params.frame_index * 1327u;

    // Start with input reservoir from temporal reuse
    var reservoir = input_reservoirs[center_index];

    // Perform spatial reuse
    for (var i = 0u; i < spatial_params.num_spatial_samples; i++) {
        // Generate neighbor offset
        let offset = get_spatial_offset(i, &rand_state);
        let neighbor_coord_f = vec2f(coord) + offset;
        let neighbor_coord = vec2u(neighbor_coord_f + 0.5); // Round to nearest pixel

        // Check if neighbor is valid for reuse
        if (is_spatial_neighbor_valid(coord, neighbor_coord, center_depth, center_normal, center_world_pos)) {
            let neighbor_index = coord_to_index_spatial(neighbor_coord);
            let neighbor_reservoir = input_reservoirs[neighbor_index];

            if (is_valid_reservoir(neighbor_reservoir)) {
                let neighbor_world_pos = world_pos_buffer[neighbor_index];
                let neighbor_normal = normal_buffer[neighbor_index];

                // Calculate Jacobian for the neighbor's sample
                let jacobian = calculate_spatial_jacobian(
                    center_world_pos,
                    center_normal,
                    neighbor_world_pos,
                    neighbor_normal,
                    neighbor_reservoir.sample
                );

                if (jacobian > 0.0) {
                    let random = rand_xorshift(&rand_state);
                    combine_reservoirs(&reservoir, neighbor_reservoir, jacobian, random);
                }
            }
        }
    }

    // Finalize and store the result
    finalize_reservoir(&reservoir);
    output_reservoirs[center_index] = reservoir;
}

// Multi-pass spatial reuse for better quality
@compute @workgroup_size(8, 8)
fn spatial_reuse_multipass(@builtin(global_invocation_id) global_id: vec3u) {
    let coord = global_id.xy;
    if (coord.x >= spatial_params.screen_width || coord.y >= spatial_params.screen_height) {
        return;
    }

    // This kernel can be called multiple times with different pass indices
    // Each pass uses a different sampling pattern to reduce correlation
    spatial_reuse(global_id);
}

// Bias correction pass (optional, for unbiased ReSTIR)
@compute @workgroup_size(8, 8)
fn bias_correction(@builtin(global_invocation_id) global_id: vec3u) {
    let coord = global_id.xy;
    if (coord.x >= spatial_params.screen_width || coord.y >= spatial_params.screen_height) {
        return;
    }

    if (spatial_params.spatial_bias_correction == 0u) {
        return; // Bias correction disabled
    }

    let center_index = coord_to_index_spatial(coord);
    var reservoir = output_reservoirs[center_index];

    if (!is_valid_reservoir(reservoir)) {
        return;
    }

    let center_world_pos = world_pos_buffer[center_index];
    let center_normal = normal_buffer[center_index];

    // Recalculate weight with bias correction
    // This involves recomputing the normalization factor by considering
    // how many neighbors could have contributed this sample
    let current_target_pdf = calculate_target_pdf(reservoir.sample, center_world_pos, center_normal);

    if (current_target_pdf > 0.0) {
        // Simplified bias correction - in practice this would be more complex
        let correction_factor = current_target_pdf / max(reservoir.target_pdf, 1e-8);
        reservoir.weight *= correction_factor;
        output_reservoirs[center_index] = reservoir;
    }
}