// src/shaders/adjudication_raster.wgsl
// AEQUITAS raster twin of the wavefront PT reference.
// Rasterizes the ReferenceSceneDesc geometry and evaluates, per fragment, an
// analytic approximation of the estimate the multi-bounce path-traced
// reference converges to:
//  - exact terms: sun NEE and MIS-weighted constant-ambient NEE with analytic
//    shadow visibility, plus the cosine-sampled continuation's constant-sky
//    escape (pt_scatter's unweighted miss accumulation) — all integrated by
//    deterministic stratified quadrature against the same spheres and
//    ground-plane quad the PT kernels trace;
//  - closed-form closures: continuation directions that hit geometry receive
//    a secondary-vertex estimate (occlusion-tested sun NEE, analytic-openness
//    ambient/sky closures, and tertiary plane/sphere-side exit-radiance
//    closures over the blocked hemisphere fractions); third-and-deeper
//    indirect scattering is truncated at the quaternary sky escape.
// The perceptual adjudication gate (dE2000 + shadow-band SSIM in
// tests/test_adjudication_gate.py) bounds the residual difference against the
// PT ground truth. All lighting/BRDF functions below are verbatim ports from
// src/shaders/pt_shade.wgsl; the background sky is the verbatim primary-miss
// background from src/shaders/pt_scatter.wgsl. Do not "fix" constants here —
// parity with the PT kernels is the contract.
// RELEVANT FILES: src/offscreen/adjudication_raster.rs, src/shaders/pt_shade.wgsl

// Uniform layout (304 bytes) — must match AdjUniforms in adjudication_raster.rs:
//   [0]  cam_origin.xyz, fov_y(rad)
//   [1]  cam_right.xyz, aspect
//   [2]  cam_up.xyz, render_width (SSAA target)
//   [3]  cam_forward.xyz, render_height (SSAA target)
//   [4]  sun dir-of-travel.xyz (normalized), sun intensity
//   [5]  sun color.rgb, unused
//   [6..9]  environment: env_ground, env_sky, miss_ground, miss_sky
//   [10..12]  spheres: center.xyz, radius
//   [13..16] materials: albedo.rgb, roughness
//   [17] draw: center.xyz, radius (sphere draws; ignored for the plane)
//   [18] misc: plane_half_extent, draw_material, draw_kind (0 plane, 1 sphere), unused
struct AdjUniforms {
    cam_origin_fovy: vec4<f32>,
    cam_right_aspect: vec4<f32>,
    cam_up_w: vec4<f32>,
    cam_forward_h: vec4<f32>,
    sun_dir_intensity: vec4<f32>,
    sun_color_pad: vec4<f32>,
    environment: array<vec4<f32>, 4>,
    sph: array<vec4<f32>, 3>,
    mats: array<vec4<f32>, 4>,
    draw_center_radius: vec4<f32>,
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> U: AdjUniforms;
@group(0) @binding(1) var<uniform> VP: mat4x4<f32>;

const PI: f32 = 3.14159265358979323846;
// Fixed stratified quadrature resolution for the environment-NEE integral.
// 24x48 = 1152 directions: enough that visibility quantization (contour
// banding where a stratum crosses a sphere silhouette) stays below the
// shadow-band SSIM tolerance.
const ENV_QUAD_U: u32 = 24u;
const ENV_QUAD_V: u32 = 48u;

// ---------------------------------------------------------------------------
// Verbatim ports from pt_shade.wgsl
// ---------------------------------------------------------------------------
fn saturate1(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn env_color(wi: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (wi.y + 1.0);
    return mix(U.environment[0].rgb, U.environment[1].rgb, t);
}

// Verbatim port of the miss background in pt_scatter.wgsl (NOT env_color).
fn miss_sky(d: vec3<f32>) -> vec3<f32> {
    let sky_t = 0.5 * (d.y + 1.0);
    return mix(U.environment[2].rgb, U.environment[3].rgb, sky_t);
}

fn make_tangent_basis(n: vec3<f32>) -> mat3x3<f32> {
    let sign = select(1.0, -1.0, n.z < 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let t = vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let bvec = vec3<f32>(b, sign + n.y * n.y * a, -n.y);
    return mat3x3<f32>(t, bvec, n);
}

fn sample_cosine_hemisphere(u1: f32, u2: f32) -> vec3<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let x = r * cos(phi);
    let y = r * sin(phi);
    let z = sqrt(max(0.0, 1.0 - u1));
    return vec3<f32>(x, y, z);
}

fn power_cosine_pdf_about_up(w: vec3<f32>, m: f32) -> f32 {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let c = max(dot(up, normalize(w)), 0.0);
    return (m + 1.0) * pow(c, m) / (2.0 * PI);
}

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1.0 - saturate1(cos_theta), 5.0);
}

fn ggx_D(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ndh2 = n_dot_h * n_dot_h;
    let denom = PI * pow(ndh2 * (a2 - 1.0) + 1.0, 2.0);
    return a2 / max(denom, 1e-6);
}

fn smith_G1(n_dot_v: f32, alpha: f32) -> f32 {
    let k = pow(alpha + 1.0, 2.0) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

fn smith_G(n_dot_l: f32, n_dot_v: f32, alpha: f32) -> f32 {
    return smith_G1(n_dot_l, alpha) * smith_G1(n_dot_v, alpha);
}

struct BrdfEval { f: vec3<f32>, pdf: f32 };

// Isotropic-only port of pt_shade.wgsl::bsdf_eval_pdf: the adjudication scene
// sets ax=ay (=0.002 after the shade kernel's clamp), so |ax-ay| < 1e-4 and
// only the isotropic branch is ever executed by the PT reference.
fn bsdf_eval_pdf(
    wo: vec3<f32>,
    wi: vec3<f32>,
    n: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> BrdfEval {
    let n_dot_l = max(dot(n, wi), 0.0);
    let n_dot_v = max(dot(n, wo), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return BrdfEval(vec3<f32>(0.0), 0.0);
    }
    let kd = saturate1(1.0 - metallic);
    let fd = (albedo / PI) * kd;
    let pdf_d = n_dot_l / PI;

    let m = max(0.02, roughness * roughness);
    let h = normalize(wi + wo);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(wo, h), 0.0);
    let D = ggx_D(n_dot_h, m);
    let G = smith_G(n_dot_l, n_dot_v, m);
    let F0 = mix(vec3<f32>(0.04), albedo, saturate1(metallic));
    let F = fresnel_schlick(v_dot_h, F0);
    let spec = (D * G) / max(4.0 * n_dot_l * n_dot_v, 1e-6);
    let fs = spec * F;
    let pdf_s = (D * n_dot_h) / max(4.0 * v_dot_h, 1e-6);

    let ks = 1.0 - kd;
    let pdf_mix = kd * pdf_d + ks * pdf_s;
    let f = fd + fs;
    return BrdfEval(f, max(pdf_mix, 1e-8));
}

// Mixture pdf of pt_shade.wgsl::sample_env_mixture (p = 0.5, exponent 16).
fn env_mixture_pdf(n: vec3<f32>, wi: vec3<f32>) -> f32 {
    let p = 0.5;
    let pdf_up = power_cosine_pdf_about_up(wi, 16.0);
    let cos_surf = max(dot(n, wi), 0.0);
    let pdf_cos = cos_surf / PI;
    return p * pdf_up + (1.0 - p) * pdf_cos;
}

// ---------------------------------------------------------------------------
// Analytic shadow visibility — mirrors pt_shadow.wgsl (spheres + plane mesh)
// ---------------------------------------------------------------------------
fn ray_sphere_occluded(
    ro: vec3<f32>, rd: vec3<f32>, c: vec3<f32>, r: f32, tmin: f32, tmax: f32,
) -> bool {
    let oc = ro - c;
    let b = dot(oc, rd);
    let cterm = dot(oc, oc) - r * r;
    let disc = b * b - cterm;
    if (disc <= 0.0) { return false; }
    let s = sqrt(disc);
    let t0 = -b - s;
    let t1 = -b + s;
    let hit0 = t0 > tmin && t0 < tmax;
    let hit1 = t1 > tmin && t1 < tmax;
    return hit0 || hit1;
}

// Finite ground quad at y=0, |x|,|z| <= half_extent (identical coverage to the
// two BVH triangles the PT shadow kernel traverses).
fn ray_plane_occluded(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32, half_extent: f32) -> bool {
    if (abs(rd.y) < 1e-7) { return false; }
    let t = -ro.y / rd.y;
    if (t <= tmin || t >= tmax) { return false; }
    let p = ro + rd * t;
    return abs(p.x) <= half_extent && abs(p.z) <= half_extent;
}

fn occluded(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> bool {
    for (var i = 0u; i < 3u; i = i + 1u) {
        let s = U.sph[i];
        if (ray_sphere_occluded(ro, rd, s.xyz, s.w, tmin, tmax)) { return true; }
    }
    return ray_plane_occluded(ro, rd, tmin, tmax, U.misc.x);
}

// Nearest hit against the analytic scene (3 spheres + finite ground quad),
// mirroring pt_intersect. idx: 0..2 = sphere, 3 = plane, -1 = miss.
struct SceneHit { t: f32, idx: i32 }

fn nearest_hit(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> SceneHit {
    var best = SceneHit(tmax, -1);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let s = U.sph[i];
        let oc = ro - s.xyz;
        let b = dot(oc, rd);
        let cterm = dot(oc, oc) - s.w * s.w;
        let disc = b * b - cterm;
        if (disc <= 0.0) { continue; }
        let sq = sqrt(disc);
        let t0 = -b - sq;
        let t1 = -b + sq;
        if (t0 > tmin && t0 < best.t) {
            best = SceneHit(t0, i32(i));
        } else if (t1 > tmin && t1 < best.t) {
            best = SceneHit(t1, i32(i));
        }
    }
    if (abs(rd.y) > 1e-7) {
        let t = -ro.y / rd.y;
        if (t > tmin && t < best.t) {
            let p = ro + rd * t;
            if (abs(p.x) <= U.misc.x && abs(p.z) <= U.misc.x) {
                best = SceneHit(t, 3);
            }
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Secondary-vertex radiance closure for the indirect estimate.
//
// The PT reference's cosine-sampled continuation ray (Lambert branch of
// pt_shade: throughput *= albedo exactly) lands on a secondary vertex y and
// there receives: sun NEE (delta light, weight 1, occlusion-tested), constant
// -ambient env NEE with the mixture-MIS weight, and its own continuation.
// This closure evaluates that vertex analytically:
//  - sun NEE: exact (full isotropic BRDF + analytic occlusion);
//  - ambient NEE: Lambert-only closed form
//      (albedo2/pi) * C_amb * INT(cos * w_mis dwi)
//    where w_mis depends only on (wi.up, wi.n2) for a Lambert pdf, so the
//    integral is a smooth function of c = n2.y approximated by the cubic
//    fit below (least squares over c in [-1,1], max abs error 0.004),
//    attenuated by the cosine-weighted openness;
//  - deeper continuation: the cosine hemisphere is partitioned into an open
//    fraction (escapes to the constant sky, pt_scatter's unweighted miss), a
//    plane-blocked fraction (receives the plane's exit radiance below the
//    vertex — the second-bounce path plane -> sphere bottom -> plane that
//    dominates the shadow pools), and per-sphere blocked fractions (receive
//    that sphere's plane-facing side, approximated as its albedo times the
//    plane exit radiance below it). Third-and-deeper indirect inside those
//    closures is truncated at the quaternary sky escape.
// ---------------------------------------------------------------------------

// Radiance leaving the ground plane at q = (qx, 0, qz), evaluated with the
// same estimator terms the PT collects at a tertiary vertex: occlusion-tested
// Lambert sun NEE, constant-ambient NEE with the Lambert-pdf MIS weight
// (the tmis cubic fit at n.y = 1 evaluates to 0.43752), and quaternary escape
// to the constant sky through the plane point's openness. Sphere occluders
// use the classical (r^2/d^2) * cos solid-angle approximation; the flat plane
// never self-occludes.
fn plane_exit_radiance(qx: f32, qz: f32) -> vec3<f32> {
    let q = vec3<f32>(qx, 0.0, qz);
    let alb_p = U.mats[3].rgb;
    var L = vec3<f32>(0.0);
    let wi_s = normalize(-U.sun_dir_intensity.xyz);
    if (wi_s.y > 0.0 && !occluded(q + vec3<f32>(0.0, 1e-3, 0.0), wi_s, 1e-3, 1e30)) {
        L += (alb_p / PI) * U.sun_color_pad.rgb * U.sun_dir_intensity.w * wi_s.y;
    }
    var ao = 1.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let s = U.sph[i];
        let dvec = s.xyz - q;
        let d2 = dot(dvec, dvec);
        if (d2 <= s.w * s.w) { continue; }
        let cosf = clamp(normalize(dvec).y, 0.0, 1.0);
        ao = ao - (s.w * s.w / d2) * cosf;
    }
    ao = clamp(ao, 0.0, 1.0);
    L += alb_p * U.environment[0].rgb * 0.43752 * ao;
    L += alb_p * U.environment[2].rgb * ao;
    return L;
}

fn secondary_radiance(P2: vec3<f32>, n2: vec3<f32>, mat_idx2: u32, wo2: vec3<f32>) -> vec3<f32> {
    let mat = U.mats[mat_idx2];
    let alb2 = mat.rgb;
    let r2 = mat.a;
    let self_idx = i32(mat_idx2);
    var L = vec3<f32>(0.0);

    // Sun NEE (delta light; single light so p_sel = 1).
    let wi_s = normalize(-U.sun_dir_intensity.xyz);
    let c2 = max(dot(n2, wi_s), 0.0);
    if (c2 > 0.0 && !occluded(P2 + n2 * 1e-3, wi_s, 1e-3, 1e30)) {
        let br2 = bsdf_eval_pdf(wo2, wi_s, n2, alb2, 0.0, r2);
        L += br2.f * U.sun_color_pad.rgb * U.sun_dir_intensity.w * c2;
    }

    // Cosine-weighted hemisphere partition. For points on a sphere the
    // effectively-infinite ground plane blocks the half-space below the
    // horizontal: its cosine-weighted fraction is exactly (1 - n.y) / 2
    // (numerically verified closed form). Sphere occluders use the classical
    // solid-angle approximation (r^2/d^2) * clamp(cos, 0, 1). The own convex
    // surface never self-occludes its outward hemisphere.
    var fp = 0.0;
    if (self_idx != 3) {
        fp = 0.5 * (1.0 - n2.y);
    }
    var fs = array<f32, 3>(0.0, 0.0, 0.0);
    var ao = 1.0 - fp;
    for (var i = 0; i < 3; i = i + 1) {
        if (i == self_idx) { continue; }
        let s = U.sph[i];
        let dvec = s.xyz - P2;
        let d2 = dot(dvec, dvec);
        if (d2 <= s.w * s.w) { continue; }
        let cosf = clamp(dot(n2, normalize(dvec)), 0.0, 1.0);
        let f = (s.w * s.w / d2) * cosf;
        fs[i] = f;
        ao = ao - f;
    }
    ao = clamp(ao, 0.0, 1.0);

    // Ambient NEE closure: (alb2/pi) * C_amb * pi * Tfit(n2.y), attenuated by
    // the openness (the PT env-NEE shadow ray sees real occluders).
    let c = clamp(n2.y, -1.0, 1.0);
    let tmis = 0.35583 + c * (0.06546 + c * (0.03152 - c * 0.01529));
    L += alb2 * U.environment[0].rgb * tmis * ao;

    // Tertiary continuation over the hemisphere partition:
    //  - open fraction escapes to the constant sky (pt_scatter's unweighted
    //    miss accumulation);
    L += alb2 * U.environment[2].rgb * ao;
    //  - the plane-blocked fraction receives the plane's exit radiance at the
    //    point below the vertex;
    if (fp > 0.0) {
        L += alb2 * plane_exit_radiance(P2.x, P2.z) * fp;
    }
    //  - each sphere-blocked fraction sees that sphere's plane-facing side:
    //    Lambert exit radiance ~ albedo_j * (plane exit radiance below j).
    for (var i = 0; i < 3; i = i + 1) {
        if (fs[i] > 0.0) {
            let s = U.sph[i];
            L += alb2 * U.mats[i].rgb * plane_exit_radiance(s.x, s.z) * fs[i];
        }
    }
    return L;
}

// ---------------------------------------------------------------------------
// Radiance: converged multi-bounce estimate at a primary surface hit.
// Sun NEE + ambient NEE quadrature (exact estimator terms) plus the
// cosine-sampled continuation: exact constant-sky escape and the
// secondary_radiance closure above for directions that hit geometry.
// ---------------------------------------------------------------------------
fn surface_radiance(P: vec3<f32>, n: vec3<f32>, mat_idx: u32) -> vec3<f32> {
    let cam_origin = U.cam_origin_fovy.xyz;
    let wo = normalize(cam_origin - P);
    let mat = U.mats[mat_idx];
    let albedo = mat.rgb;
    let roughness = mat.a;
    let metallic = 0.0;
    let shadow_o = P + n * 1e-3; // same origin offset as pt_shade's shadow rays

    var L = vec3<f32>(0.0);

    // --- Sun NEE (delta light; weight 1, single light so p_sel = 1) ---
    {
        let wi = normalize(-U.sun_dir_intensity.xyz);
        let cos_surf = max(dot(n, wi), 0.0);
        if (cos_surf > 0.0) {
            let br = bsdf_eval_pdf(wo, wi, n, albedo, metallic, roughness);
            let Li = U.sun_color_pad.rgb * U.sun_dir_intensity.w;
            if (!occluded(shadow_o, wi, 1e-3, 1e30)) {
                L += br.f * Li * cos_surf;
            }
        }
    }

    // --- Environment NEE + continuation: deterministic stratified quadrature
    // over wi ~ cosine hemisphere (pdf cos/pi), (pi/N) * sum(term/pi-scaled).
    // Per direction this accumulates BOTH terms the PT reference collects:
    //  - env NEE:  f * C_amb * w_mis      when the direction escapes
    //    (PT estimator E[f * env * cos * w_mis * V / pdf_light]);
    //  - continuation with throughput exactly `albedo` (pt_shade's Lambert
    //    branch): (albedo/pi) * C_sky on escape (pt_scatter's unweighted miss
    //    accumulation) or (albedo/pi) * secondary_radiance(y) on a hit. ---
    {
        let basis = make_tangent_basis(n);
        var accum = vec3<f32>(0.0);
        for (var i = 0u; i < ENV_QUAD_U; i = i + 1u) {
            for (var j = 0u; j < ENV_QUAD_V; j = j + 1u) {
                let u1 = (f32(i) + 0.5) / f32(ENV_QUAD_U);
                let u2 = (f32(j) + 0.5) / f32(ENV_QUAD_V);
                let wi = normalize(basis * sample_cosine_hemisphere(u1, u2));
                let cos_surf = max(dot(n, wi), 0.0);
                if (cos_surf <= 0.0) { continue; }
                let hit = nearest_hit(shadow_o, wi, 1e-3, 1e30);
                if (hit.idx < 0) {
                    // Escaped: env NEE with MIS weight + unweighted sky
                    // continuation (matching pt_scatter's miss processing).
                    let br = bsdf_eval_pdf(wo, wi, n, albedo, metallic, roughness);
                    let pdf_light = env_mixture_pdf(n, wi);
                    let w_mis = pdf_light / max(pdf_light + br.pdf, 1e-8);
                    accum += br.f * env_color(wi) * w_mis;
                    accum += (albedo / PI) * miss_sky(wi);
                } else {
                    // Hit geometry: indirect via the secondary
                    // vertex closure. Sphere normals match pt_intersect's
                    // normalize(hp - center); the plane normal is +Y.
                    let P2 = shadow_o + wi * hit.t;
                    var n2 = vec3<f32>(0.0, 1.0, 0.0);
                    if (hit.idx < 3) {
                        n2 = normalize(P2 - U.sph[hit.idx].xyz);
                    }
                    accum += (albedo / PI) * secondary_radiance(P2, n2, u32(hit.idx), -wi);
                }
            }
        }
        L += accum * (PI / f32(ENV_QUAD_U * ENV_QUAD_V));
    }

    return L;
}

// ---------------------------------------------------------------------------
// Sky pass (fullscreen triangle): verbatim primary-miss background
// ---------------------------------------------------------------------------
struct SkyOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> SkyOut {
    // Fullscreen triangle covering NDC.
    var out: SkyOut;
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32(vid >> 1u) * 4 - 1);
    out.pos = vec4<f32>(x, y, 1.0, 1.0);
    return out;
}

@fragment
fn fs_sky(in: SkyOut) -> @location(0) vec4<f32> {
    // Reconstruct the pt_raygen camera ray at this pixel center.
    let w = U.cam_up_w.w;
    let h = U.cam_forward_h.w;
    let ndc_x = (in.pos.x / w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (in.pos.y / h) * 2.0;
    let half_h = tan(0.5 * U.cam_origin_fovy.w);
    let half_w = U.cam_right_aspect.w * half_h;
    let d = normalize(
        ndc_x * half_w * U.cam_right_aspect.xyz
        + ndc_y * half_h * U.cam_up_w.xyz
        + U.cam_forward_h.xyz
    );
    return vec4<f32>(miss_sky(d), 1.0);
}

// ---------------------------------------------------------------------------
// Forward mesh pass
// ---------------------------------------------------------------------------
struct MeshOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
}

@vertex
fn vs_mesh(@location(0) position: vec3<f32>) -> MeshOut {
    var out: MeshOut;
    var world = position;
    if (U.misc.z > 0.5) {
        // Sphere draw: unit-sphere vertex -> center + radius * p.
        world = U.draw_center_radius.xyz + U.draw_center_radius.w * position;
    }
    out.world_pos = world;
    out.pos = VP * vec4<f32>(world, 1.0);
    return out;
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    var n = vec3<f32>(0.0, 1.0, 0.0);
    if (U.misc.z > 0.5) {
        // Analytic sphere normal, matching pt_intersect's normalize(hp - center).
        n = normalize(in.world_pos - U.draw_center_radius.xyz);
    }
    let mat_idx = u32(U.misc.y + 0.5);
    let L = surface_radiance(in.world_pos, n, mat_idx);
    return vec4<f32>(L, 1.0);
}
