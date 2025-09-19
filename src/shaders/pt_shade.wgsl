// src/shaders/pt_shade.wgsl
// Placeholder WGSL for path tracing shading stage.
// Exists to reserve file location and naming for future compute pipeline wiring.
// RELEVANT FILES:src/shaders/pbr_textured.wgsl,src/shaders/pt_kernel.wgsl

// Wavefront Path Tracer: Shading Stage (Lambertian MVP)
// Pops hit records, evaluates simple diffuse BRDF, samples next-bounce direction,
// and pushes ScatterRay into the scatter queue. Accumulation of background is
// handled in pt_scatter.wgsl's miss processing.

// -----------------------------------------------------------------------------
// Bindings and shared structures (keep in sync with other PT shaders)
// -----------------------------------------------------------------------------
// Bind Group 0: Uniforms (width, height, frame_index, camera params, exposure, seed_hi/lo)
struct Uniforms {
    width: u32,
    height: u32,
    frame_index: u32,
    spp: u32,
    cam_origin: vec3<f32>,
    cam_fov_y: f32,
    cam_right: vec3<f32>,
    cam_aspect: f32,
    cam_up: vec3<f32>,
    cam_exposure: f32,
    cam_forward: vec3<f32>,
    seed_hi: u32,
    seed_lo: u32,
    _pad: u32,
}

// Participating media (single scatter HG) minimal parameters
// Bound at @group(1) @binding(19)
struct MediumParams {
    g: f32,         // Henyey–Greenstein anisotropy (unused in MVP)
    sigma_t: f32,   // extinction coefficient
    density: f32,   // medium density scale
    enable: f32,    // >0.5 to enable
}

// BRDF evaluation + PDF used for MIS against light sampling
struct BrdfEval { f: vec3<f32>, pdf: f32 };

fn bsdf_eval_pdf(
    wo: vec3<f32>,
    wi: vec3<f32>,
    n: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ax: f32,
    ay: f32,
) -> BrdfEval {
    let n_dot_l = max(dot(n, wi), 0.0);
    let n_dot_v = max(dot(n, wo), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return BrdfEval(vec3<f32>(0.0), 0.0);
    }

    // Lambert
    let kd = saturate(1.0 - metallic);
    let fd = (albedo / PI) * kd;
    let pdf_d = n_dot_l / PI;

    // Microfacet GGX (isotropic unless anisotropy provided)
    let m = max(0.02, roughness * roughness);
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    var D: f32;
    var G: f32;
    // Build tangent basis for anisotropic terms
    let basis = make_tangent_basis(n);
    let t = vec3<f32>(basis[0][0], basis[1][0], basis[2][0]);
    let bb = vec3<f32>(basis[0][1], basis[1][1], basis[2][1]);
    let nn = vec3<f32>(basis[0][2], basis[1][2], basis[2][2]);
    if (abs(ax - ay) < 1e-4) {
        D = ggx_D(n_dot_h, m);
        G = smith_G(n_dot_l, n_dot_v, m);
    } else {
        D = ggx_D_aniso(h, t, bb, nn, ax, ay);
        G = smith_G_aniso(wi, wo, t, bb, nn, ax, ay);
    }
    let F0 = mix(vec3<f32>(0.04), albedo, saturate(metallic));
    let F = fresnel_schlick(v_dot_h, F0);
    let spec = (D * G) / max(4.0 * n_dot_l * n_dot_v, 1e-6);
    let fs = spec * F;
    // Specular PDF (half-vector sampling model approx)
    let pdf_s = (D * n_dot_h) / max(4.0 * v_dot_h, 1e-6);

    // Mixture model PDF with simple weights kd/ks
    let ks = 1.0 - kd;
    let pdf_mix = kd * pdf_d + ks * pdf_s;
    let f = fd + fs;
    return BrdfEval(f, max(pdf_mix, 1e-8));
}

// -----------------------------------------------------------------------------
// Lighting helpers: environment and area discs (A4/A20)
// -----------------------------------------------------------------------------
fn env_color(wi: vec3<f32>) -> vec3<f32> {
    // Simple gradient environment as placeholder importance target
    let t = 0.5 * (wi.y + 1.0);
    let sky = vec3<f32>(0.6, 0.75, 1.0);
    let ground = vec3<f32>(0.2, 0.22, 0.25);
    return mix(ground, sky, t);
}

struct AreaSample {
    wi: vec3<f32>,
    pdf: f32,
    dist: f32,
    cos_on_light: f32,
    Li: vec3<f32>,
}

fn sample_area_light_disc(P: vec3<f32>, N: vec3<f32>, L: AreaLight, u1: f32, u2: f32) -> AreaSample {
    // Build tangent frame for disc in light space
    let nL = normalize(L.normal);
    let basisL = make_tangent_basis(nL);
    let tL = vec3<f32>(basisL[0][0], basisL[1][0], basisL[2][0]);
    let bL = vec3<f32>(basisL[0][1], basisL[1][1], basisL[2][1]);
    // Uniform sample on disk
    let r = sqrt(u1) * max(L.radius, 1e-6);
    let phi = 2.0 * PI * u2;
    let dx = r * cos(phi);
    let dy = r * sin(phi);
    let X = L.position + tL * dx + bL * dy;
    let dir = X - P;
    let d = length(dir);
    if (d <= 1e-6) {
        return AreaSample(vec3<f32>(0.0), 0.0, 0.0, 0.0, vec3<f32>(0.0));
    }
    let wi = dir / d;
    let cos_surf = max(dot(N, wi), 0.0);
    let cos_on_light = max(dot(nL, -wi), 0.0);
    if (cos_surf <= 0.0 || cos_on_light <= 0.0) {
        return AreaSample(vec3<f32>(0.0), 0.0, d, 0.0, vec3<f32>(0.0));
    }
    let area = PI * max(L.radius, 1e-6) * max(L.radius, 1e-6);
    let p_area = 1.0 / area;
    let pdf = p_area * (d * d) / max(cos_on_light, 1e-6);
    let Li = L.color * L.intensity;
    return AreaSample(wi, pdf, d, cos_on_light, Li);
}

// Power-cosine lobe about world up (0,1,0)
fn sample_power_cosine_about_up(u1: f32, u2: f32, m: f32) -> vec3<f32> {
    let phi = 2.0 * PI * u2;
    let cosTheta = pow(1.0 - u1, 1.0 / (m + 1.0));
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    let x = sinTheta * cos(phi);
    let y = cosTheta;
    let z = sinTheta * sin(phi);
    return vec3<f32>(x, y, z);
}

fn power_cosine_pdf_about_up(w: vec3<f32>, m: f32) -> f32 {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let c = max(dot(up, normalize(w)), 0.0);
    return (m + 1.0) * pow(c, m) / (2.0 * PI);
}

// Mixture sampler for environment: p * power-cosine(up) + (1-p) * cosine-hemisphere(n)
struct EnvSample { wi: vec3<f32>, pdf: f32 };

fn sample_env_mixture(n: vec3<f32>, basis: mat3x3<f32>, u1: f32, u2: f32, u3: f32) -> EnvSample {
    let p = 0.5;
    if (u1 < p) {
        let wi = sample_power_cosine_about_up(u2, u3, 16.0);
        let pdf_up = power_cosine_pdf_about_up(wi, 16.0);
        let cos_surf = max(dot(n, wi), 0.0);
        let pdf_cos = cos_surf / PI;
        return EnvSample(wi, p * pdf_up + (1.0 - p) * pdf_cos);
    } else {
        let wi_local = sample_cosine_hemisphere(u2, u3);
        let wi = to_world(basis, wi_local);
        let cos_surf = max(dot(n, wi), 0.0);
        let pdf_cos = cos_surf / PI;
        let pdf_up = power_cosine_pdf_about_up(wi, 16.0);
        return EnvSample(wi, p * pdf_up + (1.0 - p) * pdf_cos);
    }
}

// Shadow ray for NEE visibility test
struct ShadowRay {
    o: vec3<f32>,           // origin
    tmin: f32,
    d: vec3<f32>,           // direction
    tmax: f32,
    contrib: vec3<f32>,     // RGB contribution if visible
    _pad0: f32,
    pixel: u32,             // destination pixel
    _pad1: vec3<u32>,       // alignment
}

// Simple area light for NEE (disc-oriented)
struct AreaLight {
    position: vec3<f32>,
    radius: f32,            // disc radius
    normal: vec3<f32>,
    intensity: f32,         // scalar intensity multiplier
    color: vec3<f32>,       // RGB color
    importance: f32,        // sampling weight
}

// Directional light (delta light)
struct DirectionalLight {
    direction: vec3<f32>,   // direction light travels (from light toward scene)
    intensity: f32,
    color: vec3<f32>,
    importance: f32,
}

// Bind Group 1: Scene (readonly storage: materials, textures/handles, accel/BVH)
struct Sphere {
    center: vec3<f32>,
    radius: f32,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    ior: f32,
    emissive: vec3<f32>,
    ax: f32, // anisotropic alpha_x
    ay: f32, // anisotropic alpha_y
}

// Bind Group 2: Queues (read/write storage buffers with atomic counters)
struct Hit {
    p: vec3<f32>,           // hit position
    t: f32,                 // ray parameter
    n: vec3<f32>,           // surface normal
    wo: vec3<f32>,          // outgoing (to camera) direction
    _pad_wo: f32,           // alignment
    mat: u32,               // material index
    throughput: vec3<f32>,  // inherited throughput
    pdf: f32,               // inherited pdf
    pixel: u32,             // pixel index
    depth: u32,             // bounce depth
    rng_hi: u32,            // RNG state high
    rng_lo: u32,            // RNG state low
    tangent: vec3<f32>,     // strand or surface tangent
    flags: u32,             // bit0 = is_hair
}

struct ScatterRay {
    o: vec3<f32>,           // origin
    tmin: f32,              // minimum ray parameter
    d: vec3<f32>,           // direction
    tmax: f32,              // maximum ray parameter
    throughput: vec3<f32>,  // updated throughput
    pdf: f32,               // updated pdf
    pixel: u32,             // pixel index
    depth: u32,             // bounce depth + 1
    rng_hi: u32,            // updated RNG state high
    rng_lo: u32,            // updated RNG state low
}

struct QueueHeader {
    in_count: atomic<u32>,  // number of items pushed
    out_count: atomic<u32>, // number of items popped
    capacity: u32,          // maximum capacity
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> scene_spheres: array<Sphere>;
@group(1) @binding(4) var<storage, read> area_lights: array<AreaLight>;
@group(1) @binding(5) var<storage, read> directional_lights: array<DirectionalLight>;
// ReSTIR reservoirs (temporal result) for guiding direct lighting
struct LightSample {
    position: vec3<f32>,
    light_index: u32,
    direction: vec3<f32>,
    intensity: f32,
    light_type: u32,
    params: vec3<f32>,
}
struct Reservoir {
    sample: LightSample,
    w_sum: f32,
    m: u32,
    weight: f32,
    target_pdf: f32,
}
@group(1) @binding(7) var<storage, read> restir_reservoirs: array<Reservoir>;
@group(1) @binding(8) var<storage, read> restir_diag_flags: array<u32>;
@group(1) @binding(9) var<storage, read> restir_debug_aov: array<vec4<f32>>;
@group(1) @binding(10) var<storage, read_write> restir_gbuffer: array<vec4<f32>>;      // normal.xyz, roughness
@group(1) @binding(11) var<storage, read_write> restir_gbuffer_pos: array<vec4<f32>>;  // world pos.xyz, 1
struct RestirSettings {
    debug_aov_mode: u32,
    qmc_mode: u32,
    adaptive_threshold_u32: u32,
    _pad: u32,
};
@group(1) @binding(12) var<uniform> restir_settings: RestirSettings;
@group(1) @binding(13) var<storage, read_write> restir_gbuffer_mat: array<u32>;        // material id per pixel
@group(1) @binding(6) var<storage, read> object_importance: array<f32>;
// SVGF guidance AOVs (threaded through wavefront)
@group(1) @binding(16) var<storage, read_write> aov_albedo_buf: array<vec4<f32>>; // RGB in .xyz
@group(1) @binding(17) var<storage, read_write> aov_depth_buf: array<vec4<f32>>;  // linear depth in .x
@group(1) @binding(18) var<storage, read_write> aov_normal_buf: array<vec4<f32>>; // normal in .xyz
@group(1) @binding(19) var<uniform> medium_params: MediumParams;
@group(2) @binding(0) var<storage, read_write> hit_queue_header: QueueHeader;
@group(2) @binding(1) var<storage, read_write> hit_queue: array<Hit>;
@group(2) @binding(2) var<storage, read_write> scatter_queue_header: QueueHeader;
@group(2) @binding(3) var<storage, read_write> scatter_queue: array<ScatterRay>;
@group(2) @binding(4) var<storage, read_write> shadow_queue_header: QueueHeader;
@group(2) @binding(5) var<storage, read_write> shadow_queue: array<ShadowRay>;
@group(3) @binding(0) var<storage, read_write> accum_hdr: array<vec4<f32>>;

// -----------------------------------------------------------------------------
// Utilities: RNG and sampling
// -----------------------------------------------------------------------------
const PI: f32 = 3.14159265358979323846;

// XorShift32 RNG for consistency with other stages
fn xorshift32(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x ^= (x << 13u);
    x ^= (x >> 17u);
    x ^= (x << 5u);
    *state = x;
    return f32(x) / 4294967296.0;
}

// Orthonormal basis from normal; returns matrix whose columns are (t, b, n)
fn make_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bvec = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    // Columns are tangent, bitangent, normal
    return mat3x3<f32>(t, bvec, n);
}

// Cosine-weighted hemisphere sample in local space (z up)
fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn to_world(basis: mat3x3<f32>, v: vec3<f32>) -> vec3<f32> {
    return basis * v;
}

// Fresnel-Schlick (vector F0)
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - saturate(cos_theta), 5.0);
}

// GGX/Trowbridge-Reitz normal distribution (isotropic)
fn ggx_D(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ndh2 = n_dot_h * n_dot_h;
    let denom = PI * pow(ndh2 * (a2 - 1.0) + 1.0, 2.0);
    return a2 / max(denom, 1e-6);
}

// Smith's masking-shadowing (height-correlated approx)
fn smith_G1(n_dot_v: f32, alpha: f32) -> f32 {
    // k from UE4: (a+1)^2 / 8
    let k = pow(alpha + 1.0, 2.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn smith_G(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    return smith_G1(n_dot_l, alpha) * smith_G1(n_dot_v, alpha);
}

// Isotropic GGX half-vector sampling around +Z
fn sample_ggx_isotropic(u1: f32, u2: f32, alpha: f32) -> vec3<f32> {
    let a2 = alpha * alpha;
    let cos_theta_h = sqrt((1.0 - u1) / (1.0 + (a2 - 1.0) * u1));
    let sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));
    let phi = 2.0 * PI * u2;
    let x = sin_theta_h * cos(phi);
    let y = sin_theta_h * sin(phi);
    let z = cos_theta_h;
    return vec3<f32>(x, y, z);
}

// Anisotropic GGX helpers
fn ggx_D_aniso(h: vec3<f32>, t: vec3<f32>, b: vec3<f32>, n: vec3<f32>, ax: f32, ay: f32) -> f32 {
    let hx = dot(h, t);
    let hy = dot(h, b);
    let hz = max(dot(h, n), 0.0);
    let x2 = (hx * hx) / (ax * ax + 1e-8);
    let y2 = (hy * hy) / (ay * ay + 1e-8);
    let denom = (x2 + y2 + hz * hz);
    return 1.0 / max(PI * ax * ay * denom * denom, 1e-6);
}

fn smith_G1_aniso(v: vec3<f32>, t: vec3<f32>, b: vec3<f32>, n: vec3<f32>, ax: f32, ay: f32) -> f32 {
    // Approximate anisotropic G1 for GGX using direction-dependent alpha
    let vx = dot(v, t);
    let vy = dot(v, b);
    let vz = max(dot(v, n), 0.0);
    let alpha_v = sqrt((vx * vx) * (ax * ax) + (vy * vy) * (ay * ay)) / max(vz, 1e-6);
    // GGX masking approximation
    return 2.0 / (1.0 + sqrt(1.0 + alpha_v * alpha_v));
}

fn smith_G_aniso(l: vec3<f32>, v: vec3<f32>, t: vec3<f32>, b: vec3<f32>, n: vec3<f32>, ax: f32, ay: f32) -> f32 {
    return smith_G1_aniso(l, t, b, n, ax, ay) * smith_G1_aniso(v, t, b, n, ax, ay);
}

fn sample_ggx_anisotropic(u1: f32, u2: f32, ax: f32, ay: f32) -> vec3<f32> {
    // Heitz-style anisotropic GGX sampling
    let two_pi = 6.283185307179586;
    var phi = atan((ay / max(ax, 1e-6)) * tan(two_pi * u2));
    // Wrap phi to correct quadrant
    if (u2 > 0.5) { phi = phi + PI; }
    let cosPhi = cos(phi);
    let sinPhi = sin(phi);
    let denom = (cosPhi * cosPhi) / max(ax * ax, 1e-8) + (sinPhi * sinPhi) / max(ay * ay, 1e-8);
    let cosTheta = 1.0 / sqrt(1.0 + (u1 / max(1.0 - u1, 1e-6)) * denom);
    let sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    return vec3<f32>(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}

// -----------------------------------------------------------------------------
// Main shading kernel
// -----------------------------------------------------------------------------
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Persistent threads: pop hits until queue is empty
    loop {
        let hit_idx = atomicAdd(&hit_queue_header.out_count, 1u);
        if hit_idx >= hit_queue_header.in_count { break; }
        if hit_idx >= hit_queue_header.capacity { break; }

        let h = hit_queue[hit_idx];

        // Fetch material parameters from scene
        let sphere_count = arrayLength(&scene_spheres);
        let mat_idx = select(0u, h.mat, h.mat < sphere_count);
        let albedo = scene_spheres[mat_idx].albedo;
        let metallic = scene_spheres[mat_idx].metallic;
        let roughness = scene_spheres[mat_idx].roughness;
        let ior = scene_spheres[mat_idx].ior;
        let emissive = scene_spheres[mat_idx].emissive;

        // Direct emissive accumulation (energy conserving via throughput)
        if (emissive.x > 0.0 || emissive.y > 0.0 || emissive.z > 0.0) {
            let add = h.throughput * emissive;
            let pix = h.pixel;
            accum_hdr[pix] = accum_hdr[pix] + vec4<f32>(add, 0.0);
        }

        // Prepare RNG from hit state
        var rng_state = h.rng_hi ^ (h.pixel * 9781u) ^ (uniforms.frame_index * 6271u);

        let n = normalize(h.n);
        let wo = normalize(h.wo); // to camera
        let n_dot_v = max(dot(n, wo), 0.0);

        // Media transmittance for primary segment (MVP): use hit distance h.t
        let medium_on = (medium_params.enable > 0.5);
        let mu = medium_params.sigma_t * medium_params.density;
        let mtrans = select(1.0, media_transmittance(h.t, mu), medium_on);

        // Write ReSTIR G-buffer and AOVs at primary hit (depth==0)
        if (h.depth == 0u) {
            let pix = h.pixel;
            restir_gbuffer[pix] = vec4<f32>(n, scene_spheres[mat_idx].roughness);
            restir_gbuffer_pos[pix] = vec4<f32>(h.p, 1.0);
            restir_gbuffer_mat[pix] = mat_idx;
            // SVGF guidance AOVs
            aov_albedo_buf[pix] = vec4<f32>(albedo, 1.0);
            aov_normal_buf[pix] = vec4<f32>(normalize(n), 1.0);
            aov_depth_buf[pix] = vec4<f32>(h.t, 0.0, 0.0, 0.0);
            // Add simple in-scattering fog toward environment for primary only
            if (medium_on) {
                let fog = media_fog_factor(h.t, mu);
                let fog_col = env_color(-wo) * fog;
                accum_hdr[pix] = accum_hdr[pix] + vec4<f32>(fog_col, 0.0);
            }
        }

        // Dispatch to BSDF (hair vs surface)
        var wi: vec3<f32>;
        var pdf: f32;
        var new_throughput: vec3<f32>;
        let basis = make_tangent_basis(n);

        // Parameters
        let a = max(0.02, roughness * roughness);
        let ax = max(0.002, scene_spheres[mat_idx].ax);
        let ay = max(0.002, scene_spheres[mat_idx].ay);
        let F0 = mix(vec3<f32>(0.04), albedo, saturate(metallic));

        // --- ReSTIR-driven direct lighting (temporal reservoir) ---
        {
            let r = restir_reservoirs[h.pixel];
            if (r.m > 0u && r.weight > 0.0 && r.target_pdf > 0.0) {
                var wi_r: vec3<f32> = vec3<f32>(0.0);
                var dist_r: f32 = 1e30;
                var Li_r: vec3<f32> = vec3<f32>(0.0);
                if (r.sample.light_type == 1u) {
                    // Directional: direction stored as incoming wi
                    wi_r = normalize(r.sample.direction);
                    Li_r = vec3<f32>(r.sample.intensity);
                } else if (r.sample.light_type == 2u) {
                    // Area disc: params.x = radius, position is sample pos
                    let dir = r.sample.position - h.p;
                    let d = length(dir);
                    if (d > 1e-6) {
                        wi_r = dir / d;
                        dist_r = d;
                        Li_r = vec3<f32>(r.sample.intensity);
                    }
                }
                let cos_surf_r = max(dot(n, wi_r), 0.0);
                if (cos_surf_r > 0.0) {
                    let br_r = bsdf_eval_pdf(wo, wi_r, n, albedo, metallic, roughness, ax, ay);
                    let imp = select(1.0, object_importance[mat_idx], mat_idx < arrayLength(&object_importance));
                    let contrib_r = h.throughput * br_r.f * Li_r * (cos_surf_r) * r.weight * imp * mtrans;

                    // Push shadow ray for visibility
                    var sr: ShadowRay;
                    sr.o = h.p + n * 1e-3;
                    sr.tmin = 1e-3;
                    sr.d = wi_r;
                    sr.tmax = select(1e30, dist_r - 1e-3, r.sample.light_type == 2u);
                    sr.contrib = contrib_r;
                    sr._pad0 = 0.0;
                    sr.pixel = h.pixel;
                    sr._pad1 = vec3<u32>(0u,0u,0u);
                    let q = atomicAdd(&shadow_queue_header.in_count, 1u);
                    if (q < shadow_queue_header.capacity) { shadow_queue[q] = sr; }

                    // Debug AOV: visualize reused vs. original via alpha channel
                    // alpha += weight if spatially reused; else alpha += 0
                    let pix = h.pixel;
                    let diag_ok = (arrayLength(&restir_diag_flags) > pix);
                    var reused: bool = false;
                    if (diag_ok) {
                        reused = (restir_diag_flags[pix] & 1u) == 1u;
                    }
                    let add_a = select(0.0, r.weight, reused);
                    accum_hdr[pix] = accum_hdr[pix] + vec4<f32>(0.0, 0.0, 0.0, add_a);
                }
            }
        }

        // --- Start NEE: Environment light sample (mixture sampler, visibility via shadow pass) ---
        {
            let u1_l = xorshift32(&rng_state);
            let u2_l = xorshift32(&rng_state);
            let u3_l = xorshift32(&rng_state);
            let s_env = sample_env_mixture(n, basis, u1_l, u2_l, u3_l);
            let wi_l = s_env.wi;
            let cos_surf = max(dot(n, wi_l), 0.0);
            if (cos_surf > 0.0) {
                let L_env = env_color(wi_l);
                let pdf_light = s_env.pdf;
                let br = bsdf_eval_pdf(wo, wi_l, n, albedo, metallic, roughness, ax, ay);
                // Balance heuristic
                let w_mis = pdf_light / max(pdf_light + br.pdf, 1e-8);
                let imp = select(1.0, object_importance[mat_idx], mat_idx < arrayLength(&object_importance));
                let contrib = h.throughput * br.f * L_env * (cos_surf / max(pdf_light, 1e-8)) * w_mis * imp * mtrans;
                // Push shadow ray (to env): use large tmax
                var sr: ShadowRay;
                sr.o = h.p + n * 1e-3;
                sr.tmin = 1e-3;
                sr.d = wi_l;
                sr.tmax = 1e30;
                sr.contrib = contrib;
                sr._pad0 = 0.0;
                sr.pixel = h.pixel;
                sr._pad1 = vec3<u32>(0u,0u,0u);
                let q = atomicAdd(&shadow_queue_header.in_count, 1u);
                if (q < shadow_queue_header.capacity) { shadow_queue[q] = sr; }
            }
        }

        // --- Start NEE: Directional lights (delta), importance-weighted selection, visibility via shadow pass ---
        {
            let count = arrayLength(&directional_lights);
            if (count > 0u) {
                // Compute total importance
                var sum_imp = 0.0;
                for (var i: u32 = 0u; i < count; i = i + 1u) { sum_imp = sum_imp + max(directional_lights[i].importance, 0.0); }
                var idx: u32 = 0u;
                if (sum_imp > 0.0) {
                    let rsel = xorshift32(&rng_state) * sum_imp;
                    var acc = 0.0;
                    for (var i: u32 = 0u; i < count; i = i + 1u) { acc = acc + max(directional_lights[i].importance, 0.0); if (rsel <= acc) { idx = i; break; } }
                } else {
                    idx = u32(floor(xorshift32(&rng_state) * f32(count)));
                }
                let L = directional_lights[min(idx, count - 1u)];
                let wi = normalize(-L.direction);
                let cos_surf = max(dot(n, wi), 0.0);
                if (cos_surf > 0.0) {
                    let br = bsdf_eval_pdf(wo, wi, n, albedo, metallic, roughness, ax, ay);
                    let Li = L.color * L.intensity;
                    // Selection probability only (delta light)
                    let p_sel = select(1.0 / f32(count), max(L.importance, 0.0) / max(sum_imp, 1e-8), sum_imp > 0.0);
                    // For delta lights, use weight 1 (no MIS with BSDF)
                    let imp = select(1.0, object_importance[mat_idx], mat_idx < arrayLength(&object_importance));
                    let contrib = h.throughput * br.f * Li * (cos_surf / max(p_sel, 1e-8)) * imp * mtrans;
                    // Push infinite shadow ray
                    var sr: ShadowRay;
                    sr.o = h.p + n * 1e-3;
                    sr.tmin = 1e-3;
                    sr.d = wi;
                    sr.tmax = 1e30;
                    sr.contrib = contrib;
                    sr._pad0 = 0.0;
                    sr.pixel = h.pixel;
                    sr._pad1 = vec3<u32>(0u,0u,0u);
                    let q = atomicAdd(&shadow_queue_header.in_count, 1u);
                    if (q < shadow_queue_header.capacity) { shadow_queue[q] = sr; }
                }
            }
        }

        // --- Start NEE: Area lights (disc), importance-weighted selection, visibility via shadow pass ---
        {
            let count = arrayLength(&area_lights);
            if (count > 0u) {
                // Compute total importance
                var sum_imp = 0.0;
                for (var i: u32 = 0u; i < count; i = i + 1u) { sum_imp = sum_imp + max(area_lights[i].importance, 0.0); }
                var idx: u32 = 0u;
                if (sum_imp > 0.0) {
                    let rsel = xorshift32(&rng_state) * sum_imp;
                    var acc = 0.0;
                    for (var i: u32 = 0u; i < count; i = i + 1u) { acc = acc + max(area_lights[i].importance, 0.0); if (rsel <= acc) { idx = i; break; } }
                } else {
                    idx = u32(floor(xorshift32(&rng_state) * f32(count)));
                }
                let light = area_lights[min(idx, count - 1u)];
                let u1_l = xorshift32(&rng_state);
                let u2_l = xorshift32(&rng_state);
                let s = sample_area_light_disc(h.p, n, light, u1_l, u2_l);
                if (s.pdf > 0.0 && s.cos_on_light > 0.0) {
                    let br = bsdf_eval_pdf(wo, s.wi, n, albedo, metallic, roughness, ax, ay);
                    let cos_surf = max(dot(n, s.wi), 0.0);
                    if (cos_surf > 0.0) {
                        // Light selection probability
                        let p_sel = select(1.0 / f32(count), max(light.importance, 0.0) / max(sum_imp, 1e-8), sum_imp > 0.0);
                        let pdf_light = p_sel * s.pdf;
                        // MIS weight (balance heuristic)
                        let w_mis = pdf_light / max(pdf_light + br.pdf, 1e-8);
                        let imp = select(1.0, object_importance[mat_idx], mat_idx < arrayLength(&object_importance));
                        let contrib = h.throughput * br.f * s.Li * (cos_surf / max(pdf_light, 1e-8)) * w_mis * imp * mtrans;
                        // Push shadow ray with finite tmax to light
                        var sr: ShadowRay;
                        sr.o = h.p + n * 1e-3;
                        sr.tmin = 1e-3;
                        sr.d = s.wi;
                        sr.tmax = s.dist - 1e-3;
                        sr.contrib = contrib;
                        sr._pad0 = 0.0;
                        sr.pixel = h.pixel;
                        sr._pad1 = vec3<u32>(0u,0u,0u);
                        let q = atomicAdd(&shadow_queue_header.in_count, 1u);
                        if (q < shadow_queue_header.capacity) { shadow_queue[q] = sr; }
                    }
                }
            }
        }

        let is_hair = (h.flags & 1u) == 1u;
        if (is_hair) {
            // Simple Kajiya–Kay: two-lobe specular aligned with strand tangent
            let T = normalize(h.tangent);
            let B = normalize(cross(n, T));
            // Build an oriented frame (T, B, N)
            let Vt = dot(wo, T);
            // Sample direction using cosine hemisphere around N for continuation
            let u1 = xorshift32(&rng_state);
            let u2 = xorshift32(&rng_state);
            let local_dir = sample_cosine_hemisphere(u1, u2);
            wi = normalize(basis * local_dir);
            let Lt = dot(wi, T);
            // Two specular lobes around tangent direction
            let m1 = 20.0; // roughness lobe 1
            let m2 = 80.0; // tighter highlight
            let f1 = pow(max(0.0, dot(normalize(reflect(-wo, T)), wi)), m1);
            let f2 = pow(max(0.0, dot(normalize(reflect(-wo, T)), wi)), m2);
            let kd = 0.2; // small diffuse along normal
            let ks = 1.0 - kd;
            let spec_col = mix(vec3<f32>(0.04), albedo, saturate(metallic));
            let f = kd * (albedo / PI) + ks * spec_col * (0.6 * f1 + 0.4 * f2);
            let cos_theta = max(0.0, dot(n, wi));
            pdf = cos_theta / PI + 1e-8;
            new_throughput = h.throughput * f * (cos_theta / pdf);
        } else if (metallic > 0.5) {
            // GGX metal reflection
            let u1 = xorshift32(&rng_state);
            let u2 = xorshift32(&rng_state);
            var h_world: vec3<f32>;
            // Build tangent, bitangent from basis columns
            let t = vec3<f32>(basis[0][0], basis[1][0], basis[2][0]);
            let bb = vec3<f32>(basis[0][1], basis[1][1], basis[2][1]);
            let nn = vec3<f32>(basis[0][2], basis[1][2], basis[2][2]);
            if (abs(ax - ay) < 1e-4) {
                // Nearly isotropic
                let h_local = sample_ggx_isotropic(u1, u2, a);
                h_world = normalize(basis * h_local);
            } else {
                let h_local = sample_ggx_anisotropic(u1, u2, ax, ay);
                h_world = normalize(t * h_local.x + bb * h_local.y + nn * h_local.z);
            }
            wi = normalize(reflect(-wo, h_world));
            let n_dot_l = max(dot(n, wi), 0.0);
            let n_dot_h = max(dot(n, h_world), 0.0);
            let v_dot_h = max(dot(wo, h_world), 0.0);
            if (n_dot_l > 0.0 && n_dot_v > 0.0) {
                let D = select(
                    ggx_D(n_dot_h, a),
                    ggx_D_aniso(h_world, t, bb, nn, ax, ay),
                    abs(ax - ay) >= 1e-4
                );
                let G = select(
                    smith_G(n_dot_l, n_dot_v, a),
                    smith_G_aniso(wi, wo, t, bb, nn, ax, ay),
                    abs(ax - ay) >= 1e-4
                );
                let F = fresnel_schlick(v_dot_h, F0);
                let spec = (D * G) / max(4.0 * n_dot_l * n_dot_v, 1e-6) * F;
                pdf = (D * n_dot_h) / max(4.0 * v_dot_h, 1e-6);
                new_throughput = h.throughput * spec * (n_dot_l / max(pdf, 1e-6));
            } else {
                // Invalid sample
                continue;
            }
        } else if (ior > 1.01) {
            // Dielectric (perfect specular): reflect or refract using Schlick
            let cosi = saturate(dot(n, wo));
            let n1 = 1.0;
            let n2 = ior;
            let F0s = pow((n2 - n1) / (n2 + n1), 2.0);
            let F = F0s + (1.0 - F0s) * pow(1.0 - cosi, 5.0);
            let u = xorshift32(&rng_state);
            if (u < F) {
                // Reflect
                wi = normalize(reflect(-wo, n));
            } else {
                // Refract (assume exiting/entering based on sign)
                let entering = dot(n, wo) > 0.0;
                let eta = select(n2 / n1, n1 / n2, entering);
                let N = select(-n, n, entering);
                // WGSL refract(I, N, eta)
                wi = normalize(refract(-wo, N, eta));
                // If TIR produced zero-length/invalid, fallback to reflection
                let len2 = dot(wi, wi);
                if (len2 < 1e-12) {
                    wi = normalize(reflect(-wo, n));
                }
            }
            // Delta lobe: pdf=1, color via albedo as tint
            pdf = 1.0;
            new_throughput = h.throughput * max(albedo, vec3<f32>(0.0));
        } else {
            // Lambertian
            let u1 = xorshift32(&rng_state);
            let u2 = xorshift32(&rng_state);
            let local_dir = sample_cosine_hemisphere(u1, u2);
            wi = normalize(basis * local_dir);
            let cos_theta = max(0.0, dot(n, wi));
            pdf = cos_theta / PI + 1e-8;
            let brdf = albedo / PI;
            new_throughput = h.throughput * brdf * (cos_theta / pdf);
        }

        // Russian roulette with optional adaptive threshold (A16)
        var continue_path = true;
        var rr_scale = 1.0;
        if (h.depth >= 4u) {
            let max_c = max(new_throughput.x, max(new_throughput.y, new_throughput.z));
            var q: f32 = clamp(1.0 - max_c, 0.0, 0.95);
            let thr_bits = restir_settings.adaptive_threshold_u32;
            let thr = bitcast<f32>(thr_bits);
            if (thr > 0.0) {
                let q_extra = clamp(1.0 - (max_c / thr), 0.0, 0.90);
                q = clamp(q + q_extra, 0.0, 0.95);
            }
            let u = xorshift32(&rng_state);
            if (u < q) {
                continue_path = false;
            } else {
                // Balance throughput for RR
                rr_scale = 1.0 / (1.0 - q);
            }
        }

        if continue_path && (h.depth + 1u) < 16u {
            // Create scatter ray
            var s: ScatterRay;
            s.o = h.p + normalize(h.n) * 1e-3;
            s.tmin = 1e-3;
            s.d = wi;
            s.tmax = 1e30;
            s.throughput = new_throughput * rr_scale;
            s.pdf = pdf;
            s.pixel = h.pixel;
            s.depth = h.depth + 1u;
            s.rng_hi = rng_state;
            s.rng_lo = h.rng_lo ^ uniforms.seed_lo;

            let qidx = atomicAdd(&scatter_queue_header.in_count, 1u);
            if qidx < scatter_queue_header.capacity {
                scatter_queue[qidx] = s;
            }
        } else {
            // Path terminated at surface: no direct accumulation here for MVP.
            // Background accumulation is handled by miss processing in pt_scatter.
        }

        // Debug AOV preview: overwrite RGB with debug AOV if enabled
        if (restir_settings.debug_aov_mode != 0u) {
            let pix = h.pixel;
            let c = restir_debug_aov[pix];
            accum_hdr[pix] = vec4<f32>(c.xyz, 1.0);
        }
    }
}

