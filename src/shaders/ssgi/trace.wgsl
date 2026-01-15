// src/shaders/ssgi/trace.wgsl
// P5.2: Half-resolution SSGI view-space tracing using HZB sphere tracing.
// Writes outHit as Rgba16Float where:
//   xy = hit UV in [0,1],
//   z  = travelled distance along the ray in view units,
//   w  = hit mask in {0,1}.

struct SsgiSettings {
    radius: f32,
    intensity: f32,
    num_steps: u32,
    step_size: f32,
    inv_resolution: vec2<f32>,
    temporal_alpha: f32,
    temporal_enabled: u32,
    use_half_res: u32,
    upsample_depth_sigma: f32,
    upsample_normal_sigma: f32,
    use_edge_aware: u32,
    _pad1: u32,
    frame_index: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: vec4<u32>,
    _pad7: vec3<u32>,
    _pad8: vec4<u32>,
};

struct CameraParams {
    view_matrix: mat4x4<f32>,
    inv_view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    inv_proj_matrix: mat4x4<f32>,
    // P1.1: Previous frame view-projection for motion vectors
    prev_view_proj_matrix: mat4x4<f32>,
    camera_pos: vec3<f32>,
    frame_index: u32,
    // P1.2: Sub-pixel jitter offset for TAA (pixel units, [-0.5, 0.5])
    jitter_offset: vec2<f32>,
    _pad_jitter: vec2<f32>,
};

@group(0) @binding(0) var tDepth: texture_2d<f32>;
@group(0) @binding(1) var tNormal: texture_2d<f32>;
@group(0) @binding(2) var tHzb: texture_2d<f32>;
@group(0) @binding(3) var outHit: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> uSsgi: SsgiSettings;
@group(0) @binding(5) var<uniform> uCam: CameraParams;

const PI: f32 = 3.14159265;

fn decode_normal(encoded: vec4<f32>) -> vec3<f32> {
    return normalize(encoded.xyz * 2.0 - 1.0);
}

fn reconstruct_view_position(uv: vec2<f32>, linear_depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, linear_depth, 1.0);
    let view = uCam.inv_proj_matrix * ndc;
    let v = view.xyz / view.w;
    return vec3<f32>(v.xy, -linear_depth);
}

fn project_to_screen(view_pos: vec3<f32>) -> vec3<f32> {
    let clip = uCam.proj_matrix * vec4<f32>(view_pos, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return vec3<f32>(uv, ndc.z);
}

fn select_hzb_mip(step_dist: f32, resolution: vec2<f32>) -> u32 {
    let footprint = step_dist * max(resolution.x, resolution.y);
    let mip = clamp(i32(floor(log2(max(footprint, 1.0)))), 0, i32(textureNumLevels(tHzb)) - 1);
    return u32(mip);
}

fn hzb_hit(uv: vec2<f32>, view_depth: f32, mip: u32) -> bool {
    let dims = vec2<f32>(textureDimensions(tHzb, i32(mip)));
    let texel = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (dims - vec2<f32>(1.0));
    let coords = vec2<i32>(texel);
    let depth_hzb = textureLoad(tHzb, coords, i32(mip)).r;
    let thickness = 0.02;
    return view_depth >= depth_hzb - thickness;
}

fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn random_float(pixel: vec2<u32>, frame: u32, salt: u32) -> f32 {
    let n = pixel.x * 1973u ^ pixel.y * 9277u ^ (frame + 1u) * 26699u ^ salt * 374761393u;
    let hashed = hash_u32(n);
    return f32(hashed & 0x00FFFFFFu) / f32(0x01000000u);
}

fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let theta = 2.0 * PI * u2;
    let x = r * cos(theta);
    let y = r * sin(theta);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

fn hemisphere_direction(normal_vs: vec3<f32>, view_dir: vec3<f32>, pixel: vec2<u32>) -> vec3<f32> {
    var hemi_normal = normalize(normal_vs);
    if (dot(hemi_normal, view_dir) > 0.0) {
        hemi_normal = -hemi_normal;
    }

    let u1 = random_float(pixel, uSsgi.frame_index, 0u);
    let u2 = random_float(pixel, uSsgi.frame_index, 1u);
    let hemi = sample_cosine_hemisphere(u1, u2);

    var tangent = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), hemi_normal));
    if (all(abs(tangent) < vec3<f32>(1e-3))) {
        tangent = normalize(cross(vec3<f32>(1.0, 0.0, 0.0), hemi_normal));
    }
    let bitangent = cross(hemi_normal, tangent);
    let dir = tangent * hemi.x + bitangent * hemi.y + hemi_normal * hemi.z;
    return normalize(dir);
}

@compute @workgroup_size(8, 8, 1)
fn cs_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    var pixel = gid.xy;
    if (uSsgi.use_half_res == 1u) {
        pixel *= 2u;
    }

    let full_dims = textureDimensions(tDepth);
    if (pixel.x >= full_dims.x || pixel.y >= full_dims.y) {
        return;
    }

    let depth = textureLoad(tDepth, pixel, 0).r;
    if (depth <= 0.0) {
        textureStore(outHit, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let uv = (vec2<f32>(vec2<u32>(pixel)) + vec2<f32>(0.5)) * uSsgi.inv_resolution;
    let normal_vs = decode_normal(textureLoad(tNormal, pixel, 0));
    let origin_vs = reconstruct_view_position(uv, depth);
    let view_dir = normalize(-origin_vs);
    let dir_vs = hemisphere_direction(normal_vs, view_dir, pixel);

    // Task 2: If steps==0, skip ray marching and mark as guaranteed miss
    if (uSsgi.num_steps == 0u || uSsgi.radius <= 0.0) {
        // Set hit_mask to 0.0 to indicate miss, hit_uv doesn't matter
        textureStore(outHit, gid.xy, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let steps = max(uSsgi.num_steps, 1u);
    let step_len = max(uSsgi.step_size, uSsgi.radius / f32(steps));

    var traveled = 0.0;
    var hit_mask = 0.0;
    var hit_uv = uv;

    for (var i: u32 = 0u; i < steps; i = i + 1u) {
        traveled = f32(i + 1u) * step_len;
        if (traveled > uSsgi.radius) {
            break;
        }
        let sample_vs = origin_vs + dir_vs * traveled;
        let proj = project_to_screen(sample_vs);
        if (proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
            break;
        }
        let mip = select_hzb_mip(traveled / uSsgi.radius, vec2<f32>(full_dims));
        if (hzb_hit(proj.xy, -sample_vs.z, mip)) {
            hit_uv = proj.xy;
            hit_mask = 1.0;
            break;
        }
    }

    textureStore(outHit, gid.xy, vec4<f32>(hit_uv, traveled, hit_mask));
}
