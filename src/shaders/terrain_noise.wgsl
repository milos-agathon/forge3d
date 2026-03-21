// src/shaders/terrain_noise.wgsl
// Shared procedural-noise helpers for terrain micro-detail and material variation.
// Keeps terrain-specific noise in one unit so multiple terrain callsites can reuse
// the same value-noise, FBM, ridged, and cellular-distance implementations.

const TERRAIN_NOISE_MAX_OCTAVES: i32 = 8;

fn terrain_hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 = p3 + dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn terrain_hash33(p: vec3<f32>) -> vec3<f32> {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q) * 43758.5453);
}

fn terrain_value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    let n000 = terrain_hash31(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = terrain_hash31(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = terrain_hash31(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = terrain_hash31(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = terrain_hash31(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = terrain_hash31(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = terrain_hash31(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = terrain_hash31(i + vec3<f32>(1.0, 1.0, 1.0));

    let x0 = mix(n000, n100, u.x);
    let x1 = mix(n010, n110, u.x);
    let x2 = mix(n001, n101, u.x);
    let x3 = mix(n011, n111, u.x);
    let y0 = mix(x0, x1, u.y);
    let y1 = mix(x2, x3, u.y);
    return mix(y0, y1, u.z);
}

fn terrain_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    let clamped_octaves = clamp(octaves, 1, TERRAIN_NOISE_MAX_OCTAVES);
    var amplitude = 0.5;
    var frequency = 1.0;
    var sum = 0.0;
    var amplitude_sum = 0.0;

    for (var i = 0; i < TERRAIN_NOISE_MAX_OCTAVES; i = i + 1) {
        if (i >= clamped_octaves) {
            break;
        }
        sum = sum + terrain_value_noise(p * frequency) * amplitude;
        amplitude_sum = amplitude_sum + amplitude;
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
    }

    return select(0.0, sum / amplitude_sum, amplitude_sum > 0.0);
}

fn terrain_ridged_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    let clamped_octaves = clamp(octaves, 1, TERRAIN_NOISE_MAX_OCTAVES);
    var amplitude = 0.5;
    var frequency = 1.0;
    var sum = 0.0;
    var amplitude_sum = 0.0;
    var ridge_weight = 1.0;

    for (var i = 0; i < TERRAIN_NOISE_MAX_OCTAVES; i = i + 1) {
        if (i >= clamped_octaves) {
            break;
        }
        let n = terrain_value_noise(p * frequency);
        let ridge = 1.0 - abs(n * 2.0 - 1.0);
        let weighted = ridge * ridge * ridge_weight;
        sum = sum + weighted * amplitude;
        amplitude_sum = amplitude_sum + amplitude;
        ridge_weight = clamp(weighted * 1.8, 0.0, 1.0);
        amplitude = amplitude * 0.5;
        frequency = frequency * 2.0;
    }

    return select(0.0, sum / amplitude_sum, amplitude_sum > 0.0);
}

fn terrain_cellular_distance(p: vec3<f32>) -> f32 {
    let base_cell = floor(p);
    let local = fract(p);
    var min_distance = 10.0;

    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let offset = vec3<f32>(f32(x), f32(y), f32(z));
                let jitter = terrain_hash33(base_cell + offset);
                let feature = offset + jitter - local;
                min_distance = min(min_distance, length(feature));
            }
        }
    }

    return clamp(min_distance / 1.7320508, 0.0, 1.0);
}
