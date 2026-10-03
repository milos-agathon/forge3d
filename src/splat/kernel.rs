// src/splat/kernel.rs
// CPU mirror of the analytic fused-occlusion kernels in
// src/shaders/splat/gaussian_intersect.wgsl and
// src/shaders/fusion/unified_occlusion.wgsl. The arithmetic follows the WGSL
// statement for statement (f32, same evaluation order) so unit tests can pin
// the closed forms without a GPU and the reference/proxy builders derive the
// same iso-levels the shaders use.
// RELEVANT FILES: src/shaders/splat/gaussian_intersect.wgsl,
//                 src/shaders/fusion/unified_occlusion.wgsl, src/splat/mod.rs

use super::symmetric_mul;

/// Splats are truncated at this many standard deviations: the per-splat
/// bounding proxy is the AABB of the 3-sigma ellipsoid.
pub const SIGMA_CUTOFF: f32 = 3.0;
/// Squared Mahalanobis radius of the truncation ellipsoid.
pub const MAHALANOBIS_CUTOFF: f32 = SIGMA_CUTOFF * SIGMA_CUTOFF;
/// Mahalanobis level of the shell a surface hit is reported on (1 sigma).
pub const SHELL_LEVEL: f32 = 1.0;
/// Coverage/opacity at which the deterministic (G-buffer/AOV) traversal
/// accepts a soft primitive as the visible surface.
pub const ACCEPT_THRESHOLD: f32 = 0.5;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Abramowitz & Stegun 7.1.26 rational approximation (|error| < 1.5e-7).
pub fn erf(x: f32) -> f32 {
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * ax);
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let value = 1.0 - poly * (-ax * ax).exp();
    if x < 0.0 {
        -value
    } else {
        value
    }
}

/// Standard normal CDF.
pub fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Fraction of a Gaussian's line integral that falls inside the ray segment
/// `[tmin, tmax]`. Along the ray `g(t) = g* + a (t - t*)^2`, so the density is
/// a 1-D Gaussian in `t` with standard deviation `1 / sqrt(a)`.
pub fn segment_fraction(a: f32, t_star: f32, tmin: f32, tmax: f32) -> f32 {
    let root = a.max(0.0).sqrt();
    let hi = ((tmax - t_star) * root).clamp(-8.0, 8.0);
    let lo = ((tmin - t_star) * root).clamp(-8.0, 8.0);
    (normal_cdf(hi) - normal_cdf(lo)).clamp(0.0, 1.0)
}

/// Analytic ray / anisotropic-Gaussian intersection record.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianRayResponse {
    /// Ray parameter of closest Mahalanobis approach.
    pub t_star: f32,
    /// Squared Mahalanobis distance at `t_star`.
    pub g_star: f32,
    /// Quadratic coefficient `d^T Sigma^-1 d`.
    pub a: f32,
    /// `opacity * exp(-g*/2) * segment_fraction`; zero outside 3 sigma.
    pub response: f32,
    /// Entry of the ray into the 1-sigma shell (equals `t_star` when the
    /// ray passes outside it).
    pub t_hit: f32,
}

/// Solve `g'(t) = 0` for `g(t) = (o + t d - mu)^T Sigma^-1 (o + t d - mu)`.
///
/// `t* = -(Delta^T Sigma^-1 d) / (d^T Sigma^-1 d)` with `Delta = o - mu`. The
/// minimum `g*` is evaluated at the closest point itself rather than as
/// `c - b^2 / a`, which cancels catastrophically for distant ray origins.
pub fn ray_gaussian(
    origin: [f32; 3],
    dir: [f32; 3],
    tmin: f32,
    tmax: f32,
    center: [f32; 3],
    inv_cov: [f32; 6],
    opacity: f32,
) -> GaussianRayResponse {
    let delta = [
        origin[0] - center[0],
        origin[1] - center[1],
        origin[2] - center[2],
    ];
    let sd = symmetric_mul(inv_cov, dir);
    let a = dot(dir, sd).max(1e-30);
    let t_star = -dot(delta, sd) / a;
    let closest = [
        delta[0] + t_star * dir[0],
        delta[1] + t_star * dir[1],
        delta[2] + t_star * dir[2],
    ];
    let g_star = dot(closest, symmetric_mul(inv_cov, closest)).max(0.0);
    let response = if g_star > MAHALANOBIS_CUTOFF {
        0.0
    } else {
        opacity * (-0.5 * g_star).exp() * segment_fraction(a, t_star, tmin, tmax)
    };
    let t_hit = t_star - ((SHELL_LEVEL - g_star).max(0.0) / a).sqrt();
    GaussianRayResponse {
        t_star,
        g_star,
        a,
        response,
        t_hit,
    }
}

/// Gradient of the quadratic form at ray parameter `t`:
/// `grad g = 2 Sigma^-1 (o + t d - mu)`.
pub fn gaussian_gradient(
    origin: [f32; 3],
    dir: [f32; 3],
    t: f32,
    center: [f32; 3],
    inv_cov: [f32; 6],
) -> [f32; 3] {
    let p = [
        origin[0] + t * dir[0] - center[0],
        origin[1] + t * dir[1] - center[1],
        origin[2] + t * dir[2] - center[2],
    ];
    symmetric_mul(inv_cov, p).map(|v| 2.0 * v)
}

/// Soft-occluder transmittance of one splat: `T = exp(-kappa * rho)`.
/// `kappa` is the optical depth of a fully opaque splat through its centre,
/// so `kappa * rho` composes additively with any other Beer-Lambert optical
/// depth (fog) along the same ray.
pub fn splat_transmittance(response: f32, kappa: f32) -> f32 {
    (-kappa * response).exp()
}

/// Squared Mahalanobis level at which a splat's transmittance crosses the
/// accept threshold: rays whose closest approach lies inside this iso-ellipsoid
/// see `T < 0.5`. `None` when the splat is too faint to ever cross it.
pub fn hard_proxy_level(opacity: f32, kappa: f32) -> Option<f32> {
    let ratio = opacity * kappa / -(1.0 - ACCEPT_THRESHOLD).ln();
    (ratio > 1.0).then(|| (2.0 * ratio.ln()).min(MAHALANOBIS_CUTOFF))
}

/// Coverage record of one LiDAR sphelet along a ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpheletCoverage {
    /// `opacity * max(1 - b^2 / r^2, 0)` with `b` the perpendicular miss
    /// distance; zero when the closest approach is outside the segment.
    pub coverage: f32,
    /// Ray parameter of closest approach to the centre.
    pub t_closest: f32,
    /// Entry of the ray into the sphere (equals `t_closest` on a miss).
    pub t_hit: f32,
}

/// Fixed-radius disc/sphelet coverage of a LiDAR return.
pub fn sphelet_coverage(
    origin: [f32; 3],
    dir: [f32; 3],
    tmin: f32,
    tmax: f32,
    center: [f32; 3],
    radius: f32,
    opacity: f32,
) -> SpheletCoverage {
    let oc = [
        center[0] - origin[0],
        center[1] - origin[1],
        center[2] - origin[2],
    ];
    let t_closest = dot(oc, dir);
    let perp = [
        oc[0] - t_closest * dir[0],
        oc[1] - t_closest * dir[1],
        oc[2] - t_closest * dir[2],
    ];
    let b2 = dot(perp, perp);
    let r2 = radius * radius;
    let inside = t_closest > tmin && t_closest < tmax && b2 < r2;
    let coverage = if inside {
        opacity * (1.0 - b2 / r2)
    } else {
        0.0
    };
    SpheletCoverage {
        coverage,
        t_closest,
        t_hit: t_closest - (r2 - b2).max(0.0).sqrt(),
    }
}

/// Radius of the hard sphere whose silhouette is the sphelet's 50 % coverage
/// contour; `None` when the point never reaches the accept threshold.
pub fn sphelet_proxy_radius(radius: f32, opacity: f32) -> Option<f32> {
    (opacity > ACCEPT_THRESHOLD).then(|| radius * (1.0 - ACCEPT_THRESHOLD / opacity).sqrt())
}

/// Density-adaptive LiDAR transmittance: coverage accumulates as
/// `1 - prod(1 - c_i)`, so the transmittance is `prod(1 - c_i)`.
pub fn coverage_transmittance(coverages: impl IntoIterator<Item = f32>) -> f32 {
    coverages
        .into_iter()
        .fold(1.0, |t, c| t * (1.0 - c.clamp(0.0, 1.0)))
}

/// Analytic optical depth of exponential height fog along a ray segment,
/// following the `VolumetricParams` convention (`density` at height 0,
/// `exp(-height_falloff * y)` vertical profile).
pub fn media_optical_depth(
    origin_y: f32,
    dir_y: f32,
    distance: f32,
    density: f32,
    height_falloff: f32,
) -> f32 {
    if density <= 0.0 || distance <= 0.0 {
        return 0.0;
    }
    let base = density * (-height_falloff * origin_y).exp();
    let k = height_falloff * dir_y;
    if k.abs() < 1e-6 {
        base * distance
    } else {
        base * (1.0 - (-k * distance).exp()) / k
    }
}

/// The unified occlusion model: every sub-occluder is a transmittance in
/// `[0, 1]` and they compose multiplicatively.
pub fn compose_transmittance(t_splat: f32, t_lidar: f32, t_terrain: f32) -> f32 {
    t_splat.clamp(0.0, 1.0) * t_lidar.clamp(0.0, 1.0) * t_terrain.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::super::{inverse_covariance, num::f32_from_usize};
    use super::*;

    const FAR: f32 = 1e30;

    #[test]
    fn erf_matches_reference_values() {
        for (x, expected) in [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (-1.0, -0.842_700_8),
        ] {
            assert!((erf(x) - expected).abs() < 2e-6, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn t_star_and_response_match_closed_form_isotropic_gaussian() {
        // Isotropic sigma, ray along +x offset by `miss` in y: the closed
        // form is t* = mu_x - o_x and g* = miss^2 / sigma^2.
        let sigma = 0.75f32;
        let icov = inverse_covariance([sigma; 3], [1.0, 0.0, 0.0, 0.0]);
        let center = [4.0, 1.0, -2.0];
        for miss in [0.0f32, 0.3, 0.9, 1.8] {
            let origin = [-3.0, 1.0 + miss, -2.0];
            let hit = ray_gaussian(origin, [1.0, 0.0, 0.0], 0.0, FAR, center, icov, 0.9);
            assert!((hit.t_star - 7.0).abs() < 1e-5, "t* {}", hit.t_star);
            let g = (miss / sigma) * (miss / sigma);
            assert!((hit.g_star - g).abs() < 1e-4, "g* {} vs {g}", hit.g_star);
            let expected = 0.9 * (-0.5 * f64::from(g)).exp();
            assert!(
                (f64::from(hit.response) - expected).abs() < 1e-5,
                "response {} vs {expected}",
                hit.response
            );
        }
    }

    #[test]
    fn anisotropic_response_matches_one_dimensional_gaussian_along_axis() {
        // Axis-aligned anisotropic splat probed along its own y axis: along
        // the ray the density is exp(-(t - t*)^2 / (2 sigma_y^2)) scaled by
        // the transverse factor exp(-(dx^2/sx^2 + dz^2/sz^2)/2).
        let scales = [0.5f32, 2.0, 0.25];
        let icov = inverse_covariance(scales, [1.0, 0.0, 0.0, 0.0]);
        let (dx, dz) = (0.4f32, -0.1f32);
        let origin = [dx, -10.0, dz];
        let hit = ray_gaussian(origin, [0.0, 1.0, 0.0], 0.0, FAR, [0.0; 3], icov, 1.0);
        assert!((hit.t_star - 10.0).abs() < 1e-5);
        assert!((hit.a - 1.0 / (scales[1] * scales[1])).abs() < 1e-6);
        let transverse =
            f64::from(dx / scales[0]).powi(2) + f64::from(dz / scales[2]).powi(2);
        assert!((f64::from(hit.g_star) - transverse).abs() < 1e-4);
        assert!((f64::from(hit.response) - (-0.5 * transverse).exp()).abs() < 1e-5);
        // Numerically integrate the 1-D profile over a half segment and
        // compare with the closed-form segment fraction.
        let (tmin, tmax) = (0.0f32, 10.0f32); // exactly half the Gaussian
        let half = ray_gaussian(origin, [0.0, 1.0, 0.0], tmin, tmax, [0.0; 3], icov, 1.0);
        assert!((half.response / hit.response - 0.5).abs() < 1e-5);
        let steps = 400_000;
        let (lo, hi) = (9.0f32, 12.5f32);
        let mut inside = 0.0f64;
        let mut total = 0.0f64;
        for i in 0..steps {
            let t = -20.0 + 60.0 * (f64::from(i as u32) + 0.5) / f64::from(steps);
            let w = (-0.5 * (t - 10.0).powi(2) / f64::from(scales[1] * scales[1])).exp();
            total += w;
            if t > f64::from(lo) && t < f64::from(hi) {
                inside += w;
            }
        }
        let clipped = ray_gaussian(origin, [0.0, 1.0, 0.0], lo, hi, [0.0; 3], icov, 1.0);
        assert!(
            (f64::from(clipped.response / hit.response) - inside / total).abs() < 2e-4,
            "{} vs {}",
            clipped.response / hit.response,
            inside / total
        );
    }

    #[test]
    fn rotated_gaussian_matches_rotated_frame_solution() {
        // Rotating the splat and the ray together must not change t*/g*.
        let scales = [0.3f32, 1.2, 0.6];
        let q = {
            let raw = [0.8f32, 0.3, -0.4, 0.2];
            let n = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            raw.map(|v| v / n)
        };
        let r = super::super::quat_to_mat3(q);
        let rot = |v: [f32; 3]| {
            [
                r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
                r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
                r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
            ]
        };
        let local_o = [-5.0f32, 0.4, 0.2];
        let local_d = {
            let raw = [1.0f32, 0.05, -0.02];
            let n = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            raw.map(|v| v / n)
        };
        let reference = ray_gaussian(
            local_o,
            local_d,
            0.0,
            FAR,
            [0.0; 3],
            inverse_covariance(scales, [1.0, 0.0, 0.0, 0.0]),
            0.7,
        );
        let rotated = ray_gaussian(
            rot(local_o),
            rot(local_d),
            0.0,
            FAR,
            [0.0; 3],
            inverse_covariance(scales, q),
            0.7,
        );
        assert!((reference.t_star - rotated.t_star).abs() < 1e-4);
        assert!((reference.g_star - rotated.g_star).abs() < 1e-3);
        assert!((reference.response - rotated.response).abs() < 1e-4);
    }

    #[test]
    fn closest_approach_is_stationary_and_gradient_is_orthogonal_to_the_ray() {
        let icov = inverse_covariance([0.4, 0.9, 0.2], [0.7, 0.1, 0.5, -0.5]);
        let (origin, center) = ([-3.0, 0.7, 1.1], [0.2, 0.1, -0.3]);
        let dir = {
            let raw = [0.9f32, -0.1, -0.4];
            let n = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
            raw.map(|v| v / n)
        };
        let hit = ray_gaussian(origin, dir, 0.0, FAR, center, icov, 1.0);
        let gradient = gaussian_gradient(origin, dir, hit.t_star, center, icov);
        // g'(t*) = grad g . d = 0.
        assert!(dot(gradient, dir).abs() < 1e-3 * hit.a.max(1.0));
        // g is larger on either side of t*.
        for dt in [-0.05f32, 0.05] {
            let p = [
                origin[0] + (hit.t_star + dt) * dir[0] - center[0],
                origin[1] + (hit.t_star + dt) * dir[1] - center[1],
                origin[2] + (hit.t_star + dt) * dir[2] - center[2],
            ];
            assert!(dot(p, symmetric_mul(icov, p)) > hit.g_star);
        }
        // The reported surface hit sits on the 1-sigma shell.
        if hit.g_star < SHELL_LEVEL {
            let p = [
                origin[0] + hit.t_hit * dir[0] - center[0],
                origin[1] + hit.t_hit * dir[1] - center[1],
                origin[2] + hit.t_hit * dir[2] - center[2],
            ];
            assert!((dot(p, symmetric_mul(icov, p)) - SHELL_LEVEL).abs() < 1e-3);
        }
    }

    #[test]
    fn distant_origins_keep_the_mahalanobis_minimum_stable() {
        let sigma = 0.05f32;
        let icov = inverse_covariance([sigma; 3], [1.0, 0.0, 0.0, 0.0]);
        let origin = [-900.0, 0.03, 0.0];
        let hit = ray_gaussian(origin, [1.0, 0.0, 0.0], 0.0, FAR, [100.0, 0.0, 0.0], icov, 1.0);
        let expected = (0.03f32 / sigma).powi(2);
        assert!((hit.g_star - expected).abs() < 5e-3, "{}", hit.g_star);
    }

    #[test]
    fn transmittance_is_monotone_in_opacity_and_lidar_density() {
        // Larger opacity -> lower transmittance.
        let icov = inverse_covariance([0.5; 3], [1.0, 0.0, 0.0, 0.0]);
        let mut previous = 1.0f32;
        for step in 1..=10 {
            let opacity = f32_from_usize(step) / 10.0;
            let hit = ray_gaussian(
                [-4.0, 0.1, 0.0],
                [1.0, 0.0, 0.0],
                0.0,
                FAR,
                [0.0; 3],
                icov,
                opacity,
            );
            let t = splat_transmittance(hit.response, 4.0);
            assert!(t < previous, "opacity {opacity}: {t} !< {previous}");
            previous = t;
        }
        // Denser LiDAR swath -> lower transmittance, saturating to opaque.
        let mut previous = 1.0f32;
        for points in [1usize, 2, 4, 8, 16, 64] {
            let coverages = (0..points).map(|i| {
                // Returns scattered across the beam at a fixed miss distance.
                let z = 0.02 * f32_from_usize(i % 5);
                sphelet_coverage(
                    [-4.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    0.0,
                    FAR,
                    [f32_from_usize(i), 0.06, z],
                    0.1,
                    0.6,
                )
                .coverage
            });
            let t = coverage_transmittance(coverages);
            assert!(t < previous, "{points} points: {t} !< {previous}");
            previous = t;
        }
        assert!(previous < 1e-3, "a dense swath must saturate, got {previous}");
        // A single thin return stays translucent.
        let single = sphelet_coverage(
            [-4.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            0.0,
            FAR,
            [0.0, 0.08, 0.0],
            0.1,
            0.6,
        );
        let t = coverage_transmittance([single.coverage]);
        assert!(t > 0.5 && t < 1.0, "{t}");
    }

    #[test]
    fn hard_proxy_iso_level_is_where_transmittance_crosses_one_half() {
        let (opacity, kappa) = (0.9f32, 4.0f32);
        let level = hard_proxy_level(opacity, kappa).unwrap();
        let at = |g: f32| splat_transmittance(opacity * (-0.5 * g).exp(), kappa);
        assert!((at(level) - 0.5).abs() < 1e-5);
        assert!(at(level - 0.05) < 0.5 && at(level + 0.05) > 0.5);
        assert!(hard_proxy_level(0.1, 4.0).is_none());
        let radius = sphelet_proxy_radius(0.2, 1.0).unwrap();
        let grazing = sphelet_coverage(
            [-1.0, radius, 0.0],
            [1.0, 0.0, 0.0],
            0.0,
            FAR,
            [0.0; 3],
            0.2,
            1.0,
        );
        assert!((grazing.coverage - 0.5).abs() < 1e-5);
        assert!(sphelet_proxy_radius(0.2, 0.4).is_none());
    }

    #[test]
    fn fog_and_splat_optical_depths_add_exactly_once() {
        // Homogeneous fog: tau = density * distance; composing with a splat
        // multiplies transmittances, i.e. adds optical depths.
        let tau_fog = media_optical_depth(0.0, 0.0, 12.0, 0.05, 0.0);
        assert!((tau_fog - 0.6).abs() < 1e-6);
        let rho = 0.35f32;
        let combined = splat_transmittance(rho, 4.0) * (-tau_fog).exp();
        assert!((combined - (-(4.0 * rho + tau_fog)).exp()).abs() < 1e-6);
        // Height falloff: closed form of the exponential profile.
        let tau = media_optical_depth(2.0, 0.5, 8.0, 0.1, 0.25);
        let expected = 0.1 * (-0.5f64).exp() * (1.0 - (-0.125f64 * 8.0).exp()) / 0.125;
        assert!((f64::from(tau) - expected).abs() < 1e-6);
        assert_eq!(media_optical_depth(0.0, 1.0, 5.0, 0.0, 0.3), 0.0);
        assert_eq!(compose_transmittance(0.5, 0.5, 0.0), 0.0);
        assert!((compose_transmittance(0.5, 0.4, 1.0) - 0.2).abs() < 1e-7);
    }
}
