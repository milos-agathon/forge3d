// src/shaders/splat/gaussian_intersect.wgsl
// SPLAT-FUSED: analytic ray / anisotropic-Gaussian intersection.
//
// For a ray o + t d and a Gaussian with centre mu and inverse covariance
// S = Sigma^-1 (packed upper triangle), the squared Mahalanobis distance
//
//     g(t) = (o + t d - mu)^T S (o + t d - mu)
//
// is a parabola in t. g'(t) = 0 gives the parameter of closest approach
//
//     t* = -(Delta^T S d) / (d^T S d),   Delta = o - mu,
//
// and g(t) = g* + a (t - t*)^2 with a = d^T S d. The density along the ray is
// therefore a 1-D Gaussian in t, so the line integral over a ray segment is
// closed form: rho = alpha * exp(-g*/2) * (Phi(sqrt(a)(tmax - t*)) -
// Phi(sqrt(a)(tmin - t*))). For a segment that contains the whole splat the
// bracket is 1 and rho is the peak response alpha * exp(-g*/2).
//
// A splat is a SOFT occluder: its transmittance is T = exp(-kappa * rho),
// never a hard surface. Nothing here is rasterized or projected — this is the
// traversal primitive. The module is binding-free: callers pass the record.
// CPU mirror: src/splat/kernel.rs (same arithmetic, pinned by unit tests).
// RELEVANT FILES: src/shaders/fusion/unified_occlusion.wgsl, src/splat/kernel.rs

// Truncation: (3 sigma)^2. The per-splat bounding proxy is the AABB of this
// ellipsoid, so the kernel only sees candidates the traversal returns.
const GAUSSIAN_MAHALANOBIS_CUTOFF: f32 = 9.0;
// Mahalanobis level of the shell a surface hit is reported on (1 sigma).
const GAUSSIAN_SHELL_LEVEL: f32 = 1.0;
const GAUSSIAN_SH_C0: f32 = 0.2820948;
const GAUSSIAN_SH_C1: f32 = 0.4886025;

// S v for the packed symmetric matrix m0 = (xx, xy, xz, yy), m1 = (yz, zz, -, -).
fn gaussian_sym_mul(m0: vec4<f32>, m1: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        m0.x * v.x + m0.y * v.y + m0.z * v.z,
        m0.y * v.x + m0.w * v.y + m1.x * v.z,
        m0.z * v.x + m1.x * v.y + m1.y * v.z,
    );
}

// Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
fn gaussian_erf(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.2548296 + t * (-0.28449674 + t * (1.4214137
        + t * (-1.453152 + t * 1.0614054))));
    let value = 1.0 - poly * exp(-ax * ax);
    return select(value, -value, x < 0.0);
}

fn gaussian_normal_cdf(x: f32) -> f32 {
    return 0.5 * (1.0 + gaussian_erf(x * 0.70710678));
}

// Fraction of the Gaussian's line integral inside the segment [tmin, tmax].
fn gaussian_segment_fraction(a: f32, t_star: f32, tmin: f32, tmax: f32) -> f32 {
    let root = sqrt(max(a, 0.0));
    let hi = clamp((tmax - t_star) * root, -8.0, 8.0);
    let lo = clamp((tmin - t_star) * root, -8.0, 8.0);
    return clamp(gaussian_normal_cdf(hi) - gaussian_normal_cdf(lo), 0.0, 1.0);
}

struct GaussianResponse {
    t_star: f32,    // ray parameter of closest Mahalanobis approach
    g_star: f32,    // squared Mahalanobis distance at t_star
    a: f32,         // d^T S d
    response: f32,  // rho = alpha * exp(-g*/2) * segment fraction (0 beyond 3 sigma)
}

// The analytic kernel. The minimum g* is evaluated at the closest point
// itself rather than as c - b^2/a, which cancels catastrophically in f32
// for distant ray origins.
fn gaussian_response(
    origin: vec3<f32>,
    dir: vec3<f32>,
    tmin: f32,
    tmax: f32,
    center: vec3<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    opacity: f32,
) -> GaussianResponse {
    var out: GaussianResponse;
    let delta = origin - center;
    let sd = gaussian_sym_mul(m0, m1, dir);
    out.a = max(dot(dir, sd), 1e-30);
    out.t_star = -dot(delta, sd) / out.a;
    let closest = delta + out.t_star * dir;
    out.g_star = max(dot(closest, gaussian_sym_mul(m0, m1, closest)), 0.0);
    out.response = 0.0;
    if (out.g_star <= GAUSSIAN_MAHALANOBIS_CUTOFF) {
        out.response = opacity * exp(-0.5 * out.g_star)
            * gaussian_segment_fraction(out.a, out.t_star, tmin, tmax);
    }
    return out;
}

// Soft-occluder transmittance of one splat. kappa is the optical depth of a
// fully opaque splat through its centre, so kappa * rho adds to any other
// Beer-Lambert optical depth along the ray (fog) without double counting.
fn gaussian_transmittance(response: f32, kappa: f32) -> f32 {
    return exp(-kappa * response);
}

// 3DGS spherical-harmonic colour (bands 0 and 1), view direction `dir` from
// the viewer toward the splat.
fn gaussian_sh_color(
    sh0: vec3<f32>,
    degree: f32,
    sh1_0: vec3<f32>,
    sh1_1: vec3<f32>,
    sh1_2: vec3<f32>,
    dir: vec3<f32>,
) -> vec3<f32> {
    var color = GAUSSIAN_SH_C0 * sh0;
    if (degree >= 1.0) {
        color = color - GAUSSIAN_SH_C1 * dir.y * sh1_0
            + GAUSSIAN_SH_C1 * dir.z * sh1_1
            - GAUSSIAN_SH_C1 * dir.x * sh1_2;
    }
    return max(color + vec3<f32>(0.5), vec3<f32>(0.0));
}

struct GaussianSurfaceHit {
    t_star: f32,            // closest-approach parameter
    t_hit: f32,             // entry into the 1-sigma shell (t_star if outside it)
    response: f32,          // rho
    // Geometric normal: grad g at t_star = 2 S (o + t* d - mu). By
    // construction g'(t*) = grad g . d = 0, so it is orthogonal to the ray.
    normal: vec3<f32>,
    // grad g at t_hit — faces the ray origin; the normal BRDF evaluation uses.
    shading_normal: vec3<f32>,
}

// Closest-hit form: t*, response and the gradient normals. The caller decides
// acceptance (stochastically against 1 - T, or the 50 % threshold) and
// shades with gaussian_sh_color.
fn gaussian_closest_hit(
    origin: vec3<f32>,
    dir: vec3<f32>,
    tmin: f32,
    tmax: f32,
    center: vec3<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    opacity: f32,
) -> GaussianSurfaceHit {
    var hit: GaussianSurfaceHit;
    let r = gaussian_response(origin, dir, tmin, tmax, center, m0, m1, opacity);
    hit.t_star = r.t_star;
    hit.response = r.response;
    hit.t_hit = r.t_star - sqrt(max(GAUSSIAN_SHELL_LEVEL - r.g_star, 0.0) / r.a);
    let at_star = origin + r.t_star * dir - center;
    hit.normal = 2.0 * gaussian_sym_mul(m0, m1, at_star);
    let at_hit = origin + hit.t_hit * dir - center;
    var n = gaussian_sym_mul(m0, m1, at_hit);
    let len = length(n);
    if (len > 1e-20) {
        n = n / len;
    } else {
        n = -dir;
    }
    // A grazing pass (outside the shell) leaves n orthogonal to the ray;
    // never let it face away from the viewer.
    if (dot(n, dir) > 0.0) { n = -n; }
    hit.shading_normal = n;
    return hit;
}

// Any-hit / shadow form: multiply the running transmittance by this splat's
// T = exp(-kappa * rho). The caller early-outs once the product is below its
// epsilon.
fn gaussian_any_hit(
    transmittance: f32,
    origin: vec3<f32>,
    dir: vec3<f32>,
    tmin: f32,
    tmax: f32,
    center: vec3<f32>,
    m0: vec4<f32>,
    m1: vec4<f32>,
    opacity: f32,
    kappa: f32,
) -> f32 {
    let r = gaussian_response(origin, dir, tmin, tmax, center, m0, m1, opacity);
    return transmittance * gaussian_transmittance(r.response, kappa);
}
