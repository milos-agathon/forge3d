// A21: Ambient Occlusion compute shader
// Half-precision G-buffer; cosine AO

@group(0) @binding(0) var depth_texture: texture_2d<f32>;
@group(0) @binding(1) var normal_texture: texture_2d<f32>;
@group(0) @binding(2) var ao_output: texture_storage_2d<r16float, write>;

struct AOParams {
    radius: f32,
    intensity: f32,
    samples: u32,
    bias: f32,
}

@group(1) @binding(0) var<uniform> params: AOParams;

@compute @workgroup_size(8, 8, 1)
fn cs_ambient_occlusion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let coord = vec2<i32>(global_id.xy);
    let size = textureDimensions(depth_texture);

    if (coord.x >= i32(size.x) || coord.y >= i32(size.y)) {
        return;
    }

    let depth = textureLoad(depth_texture, coord, 0).r;
    let normal = textureLoad(normal_texture, coord, 0).xyz;

    // Simple cosine AO implementation
    var ao = 0.0;
    let sample_count = f32(params.samples);

    for (var i = 0u; i < params.samples; i++) {
        // Sample hemisphere around normal
        let sample_dir = generate_sample(coord, i);
        let occlusion = sample_occlusion(coord, normal, sample_dir, depth);
        ao += occlusion;
    }

    ao = ao / sample_count;
    ao = max(0.0, 1.0 - ao * params.intensity);

    textureStore(ao_output, coord, vec4<f32>(ao, 0.0, 0.0, 1.0));
}

fn generate_sample(coord: vec2<i32>, index: u32) -> vec3<f32> {
    // Simple hemisphere sampling
    let angle = f32(index) * 2.399963; // Golden angle
    let z = sqrt(f32(index) / f32(16)); // Assuming 16 samples
    let r = sqrt(1.0 - z * z);
    return vec3<f32>(r * cos(angle), r * sin(angle), z);
}

fn sample_occlusion(coord: vec2<i32>, normal: vec3<f32>, sample_dir: vec3<f32>, depth: f32) -> f32 {
    // Simplified occlusion test
    return max(0.0, dot(normal, sample_dir));
}