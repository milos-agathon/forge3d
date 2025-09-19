// src/shaders/restir_reservoir.wgsl
// ReSTIR reservoir operations and light sampling

struct LightSample {
    position: vec3f,
    light_index: u32,
    direction: vec3f,
    intensity: f32,
    light_type: u32,
    params: vec3f,
}

struct Reservoir {
    sample: LightSample,
    w_sum: f32,
    m: u32,
    weight: f32,
    target_pdf: f32,
}

struct AliasEntry {
    prob: f32,
    alias_idx: u32,
}

// Bind groups
@group(0) @binding(0) var<uniform> params: RestirParams;
@group(0) @binding(1) var<storage, read> lights: array<LightSample>;
@group(0) @binding(2) var<storage, read> alias_table: array<AliasEntry>;

struct RestirParams {
    num_lights: u32,
    initial_candidates: u32,
    spatial_neighbors: u32,
    temporal_neighbors: u32,
    spatial_radius: f32,
    max_temporal_age: u32,
    frame_index: u32,
    bias_correction: u32,
}

// Random number generation (XorShift)
fn rand_xorshift(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return f32(x) * (1.0 / 4294967296.0);
}

// Sample from alias table
fn sample_alias_table(u1: f32, u2: f32) -> u32 {
    if (params.num_lights == 0u) {
        return 0u;
    }

    let n = params.num_lights;
    let scaled_u1 = u1 * f32(n);
    let bin = min(u32(scaled_u1), n - 1u);
    let frac = scaled_u1 - f32(bin);

    let entry = alias_table[bin];

    if (frac < entry.prob) {
        return bin;
    } else {
        return entry.alias_idx;
    }
}

// Calculate target PDF for a light sample
fn calculate_target_pdf(sample: LightSample, shading_point: vec3f, normal: vec3f) -> f32 {
    let light_dir = sample.position - shading_point;
    let dist_sq = dot(light_dir, light_dir);

    if (dist_sq <= 0.0) {
        return 0.0;
    }

    let dist = sqrt(dist_sq);
    let light_dir_norm = light_dir / dist;

    // Cosine term (N · L)
    let cos_theta = max(0.0, dot(normal, light_dir_norm));

    if (cos_theta <= 0.0) {
        return 0.0;
    }

    // Simplified BRDF * G * Le / distance^2
    let geometric_term = cos_theta / dist_sq;
    return sample.intensity * geometric_term;
}

// Update reservoir with a new sample
fn update_reservoir(reservoir: ptr<function, Reservoir>, sample: LightSample, weight: f32, random: f32) -> bool {
    (*reservoir).w_sum += weight;
    (*reservoir).m += 1u;

    // Reservoir sampling: accept with probability weight / w_sum
    if (random * (*reservoir).w_sum <= weight) {
        (*reservoir).sample = sample;
        (*reservoir).target_pdf = weight;
        return true;
    }

    return false;
}

// Finalize reservoir by computing final weight
fn finalize_reservoir(reservoir: ptr<function, Reservoir>) {
    if ((*reservoir).w_sum > 0.0 && (*reservoir).target_pdf > 0.0) {
        (*reservoir).weight = (*reservoir).w_sum / (f32((*reservoir).m) * (*reservoir).target_pdf);
    } else {
        (*reservoir).weight = 0.0;
    }
}

// Combine two reservoirs (for spatial/temporal reuse)
fn combine_reservoirs(
    reservoir: ptr<function, Reservoir>,
    other: Reservoir,
    other_jacobian: f32,
    random: f32
) {
    if (other.m == 0u || other.weight == 0.0) {
        return;
    }

    // Calculate the weight for the other reservoir's sample in our context
    let other_contribution = other.target_pdf * other_jacobian * f32(other.m);

    (*reservoir).w_sum += other_contribution;
    (*reservoir).m += other.m;

    // Reservoir sampling: accept other's sample with probability other_contribution / w_sum
    if (random * (*reservoir).w_sum <= other_contribution) {
        (*reservoir).sample = other.sample;
        (*reservoir).target_pdf = other.target_pdf * other_jacobian;
    }
}

// Initialize empty reservoir
fn init_reservoir() -> Reservoir {
    var reservoir: Reservoir;
    reservoir.sample.position = vec3f(0.0);
    reservoir.sample.light_index = 0u;
    reservoir.sample.direction = vec3f(0.0);
    reservoir.sample.intensity = 0.0;
    reservoir.sample.light_type = 0u;
    reservoir.sample.params = vec3f(0.0);
    reservoir.w_sum = 0.0;
    reservoir.m = 0u;
    reservoir.weight = 0.0;
    reservoir.target_pdf = 0.0;
    return reservoir;
}

// Perform initial sampling to fill a reservoir
fn initial_sampling(
    shading_point: vec3f,
    normal: vec3f,
    rand_state: ptr<function, u32>
) -> Reservoir {
    var reservoir = init_reservoir();

    if (params.num_lights == 0u) {
        return reservoir;
    }

    for (var i = 0u; i < params.initial_candidates; i++) {
        let u1 = rand_xorshift(rand_state);
        let u2 = rand_xorshift(rand_state);
        let u3 = rand_xorshift(rand_state);

        // Sample a light using alias table
        let light_idx = sample_alias_table(u1, u2);

        if (light_idx < params.num_lights) {
            let light_sample = lights[light_idx];
            let target_pdf = calculate_target_pdf(light_sample, shading_point, normal);

            if (target_pdf > 0.0) {
                update_reservoir(&reservoir, light_sample, target_pdf, u3);
            }
        }
    }

    finalize_reservoir(&reservoir);
    return reservoir;
}

// Validate reservoir sample
fn is_valid_reservoir(reservoir: Reservoir) -> bool {
    return reservoir.m > 0u && reservoir.weight > 0.0 && reservoir.target_pdf > 0.0;
}