// src/shaders/restir_temporal.wgsl
// ReSTIR temporal reuse implementation

#include "restir_reservoir.wgsl"

// Temporal reservoir buffers
@group(1) @binding(0) var<storage, read> prev_reservoirs: array<Reservoir>;
@group(1) @binding(1) var<storage, read_write> curr_reservoirs: array<Reservoir>;
@group(1) @binding(2) var<storage, read> motion_vectors: array<vec2f>;
@group(1) @binding(3) var<storage, read> depth_buffer: array<f32>;
@group(1) @binding(4) var<storage, read> normal_buffer: array<vec3f>;

struct TemporalParams {
    screen_width: u32,
    screen_height: u32,
    max_temporal_age: u32,
    temporal_bias_correction: u32,
    depth_threshold: f32,
    normal_threshold: f32,
}

@group(1) @binding(5) var<uniform> temporal_params: TemporalParams;

// Convert screen coordinates to buffer index
fn coord_to_index(coord: vec2u) -> u32 {
    return coord.y * temporal_params.screen_width + coord.x;
}

// Convert buffer index to screen coordinates
fn index_to_coord(index: u32) -> vec2u {
    let y = index / temporal_params.screen_width;
    let x = index - y * temporal_params.screen_width;
    return vec2u(x, y);
}

// Check if temporal reuse is valid based on geometry
fn is_temporal_valid(
    curr_coord: vec2u,
    prev_coord: vec2u,
    curr_depth: f32,
    curr_normal: vec3f
) -> bool {
    // Check bounds
    if (prev_coord.x >= temporal_params.screen_width || prev_coord.y >= temporal_params.screen_height) {
        return false;
    }

    let prev_index = coord_to_index(prev_coord);

    // Check depth similarity
    let prev_depth = depth_buffer[prev_index];
    let depth_diff = abs(curr_depth - prev_depth) / max(curr_depth, prev_depth);
    if (depth_diff > temporal_params.depth_threshold) {
        return false;
    }

    // Check normal similarity
    let prev_normal = normal_buffer[prev_index];
    let normal_similarity = dot(curr_normal, prev_normal);
    if (normal_similarity < temporal_params.normal_threshold) {
        return false;
    }

    return true;
}

// Calculate Jacobian for temporal reuse (accounts for change in visibility/geometry)
fn calculate_temporal_jacobian(
    curr_shading_point: vec3f,
    curr_normal: vec3f,
    prev_shading_point: vec3f,
    prev_normal: vec3f,
    light_sample: LightSample
) -> f32 {
    // Calculate target PDFs for both positions
    let curr_pdf = calculate_target_pdf(light_sample, curr_shading_point, curr_normal);
    let prev_pdf = calculate_target_pdf(light_sample, prev_shading_point, prev_normal);

    if (prev_pdf <= 0.0) {
        return 0.0;
    }

    return curr_pdf / prev_pdf;
}

// Temporal reuse compute kernel
@compute @workgroup_size(8, 8)
fn temporal_reuse(@builtin(global_invocation_id) global_id: vec3u) {
    let coord = global_id.xy;
    if (coord.x >= temporal_params.screen_width || coord.y >= temporal_params.screen_height) {
        return;
    }

    let curr_index = coord_to_index(coord);

    // Get current pixel data
    let curr_depth = depth_buffer[curr_index];
    let curr_normal = normal_buffer[curr_index];

    // Skip background pixels
    if (curr_depth <= 0.0) {
        curr_reservoirs[curr_index] = init_reservoir();
        return;
    }

    // Get motion vector and calculate previous position
    let motion = motion_vectors[curr_index];
    let prev_coord_f = vec2f(coord) - motion;
    let prev_coord = vec2u(prev_coord_f + 0.5); // Round to nearest pixel

    // Initialize random state
    var rand_state = curr_index * 1973u + temporal_params.screen_width * temporal_params.screen_height * params.frame_index;

    // Start with initial sampling for current pixel
    let curr_shading_point = vec3f(0.0); // This should come from G-buffer
    var reservoir = initial_sampling(curr_shading_point, curr_normal, &rand_state);

    // Attempt temporal reuse
    if (is_temporal_valid(coord, prev_coord, curr_depth, curr_normal)) {
        let prev_index = coord_to_index(prev_coord);
        let prev_reservoir = prev_reservoirs[prev_index];

        if (is_valid_reservoir(prev_reservoir)) {
            // Calculate previous shading point (simplified)
            let prev_shading_point = vec3f(0.0); // This should come from previous G-buffer
            let prev_normal = normal_buffer[prev_index];

            // Calculate Jacobian for the reused sample
            let jacobian = calculate_temporal_jacobian(
                curr_shading_point,
                curr_normal,
                prev_shading_point,
                prev_normal,
                prev_reservoir.sample
            );

            if (jacobian > 0.0) {
                let random = rand_xorshift(&rand_state);
                combine_reservoirs(&reservoir, prev_reservoir, jacobian, random);
            }
        }
    }

    // Finalize and store the result
    finalize_reservoir(&reservoir);
    curr_reservoirs[curr_index] = reservoir;
}

// Temporal history management
@compute @workgroup_size(64)
fn update_temporal_history(@builtin(global_invocation_id) global_id: vec3u) {
    let index = global_id.x;
    let total_pixels = temporal_params.screen_width * temporal_params.screen_height;

    if (index >= total_pixels) {
        return;
    }

    // Copy current reservoirs to previous for next frame
    // This would typically be done with a buffer swap, but shown here for clarity
    // prev_reservoirs[index] = curr_reservoirs[index];
}