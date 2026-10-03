// src/splat/mod.rs
// SPLAT-FUSED: anisotropic 3D Gaussian splat representation for the fused
// path-traced integrator. `GaussianSplatCloud` is a structure-of-arrays store
// (positions, per-axis sigmas, unit quaternions, opacities, SH colour) with
// the inverse covariance of every splat precomputed and packed as the upper
// triangle the analytic ray/Gaussian kernel consumes. The cloud registers its
// host bytes with the global memory tracker; clouds too large to be resident
// are paged through `stream`.
// RELEVANT FILES: src/splat/load.rs, src/splat/kernel.rs,
//                 src/shaders/splat/gaussian_intersect.wgsl

pub mod bvh;
pub mod fixture;
pub mod fusion;
pub mod kernel;
pub mod load;
pub mod stream;

use crate::core::error::RenderError;
use crate::core::resource_tracker::{tracked_host_allocation, ResourceHandle};

pub use load::{load_gaussian_splats, save_gaussian_splats};

/// Zeroth-order real spherical-harmonic basis constant (3DGS convention).
pub const SH_C0: f32 = 0.282_094_8;
/// First-order real spherical-harmonic basis constant.
pub const SH_C1: f32 = 0.488_602_5;
const SH_C2: [f32; 5] = [
    1.092_548_4,
    -1.092_548_4,
    0.315_391_57,
    -1.092_548_4,
    0.546_274_2,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_34,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

/// Integer-to-f32 conversions used by the fused path: counts, indices and
/// grid coordinates only. World positions never pass through these.
pub(crate) mod num {
    #[cfg(test)]
    #[inline]
    pub fn f32_from_usize(value: usize) -> f32 {
        value as f32
    }

    #[inline]
    pub fn f32_from_u32(value: u32) -> f32 {
        value as f32
    }
}

/// Higher spherical-harmonic bands (everything above the DC term), stored as
/// `[splat][coefficient][rgb]` with `(degree + 1)^2 - 1` coefficients per splat.
#[derive(Clone, Debug, PartialEq)]
pub struct ShRest {
    pub degree: u32,
    pub coeffs: Vec<[f32; 3]>,
}

impl ShRest {
    /// Coefficients per splat for this degree (DC excluded).
    pub fn per_splat(&self) -> usize {
        sh_rest_count(self.degree)
    }
}

/// Number of non-DC SH coefficients for `degree`.
pub fn sh_rest_count(degree: u32) -> usize {
    ((degree as usize + 1) * (degree as usize + 1)).saturating_sub(1)
}

/// Structure-of-arrays anisotropic Gaussian splat cloud.
///
/// `scales` are per-axis standard deviations in world units (already
/// exponentiated), `rotations` are unit quaternions `(w, x, y, z)`,
/// `opacities` are in `[0, 1]` (already passed through the logistic), and
/// `sh0` holds the raw DC spherical-harmonic coefficients.
#[derive(Debug)]
pub struct GaussianSplatCloud {
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    pub sh0: Vec<[f32; 3]>,
    pub sh_rest: Option<ShRest>,
    /// Packed upper-triangular inverse covariance (xx, xy, xz, yy, yz, zz).
    inv_cov: Vec<[f32; 6]>,
    /// Host-memory registration; released when the cloud is dropped.
    _tracker: ResourceHandle,
}

impl GaussianSplatCloud {
    /// Assemble a cloud from its attribute arrays. Validates lengths and
    /// values, normalizes the quaternions, precomputes the packed inverse
    /// covariance and registers the host bytes with the memory tracker.
    pub fn from_parts(
        positions: Vec<[f32; 3]>,
        scales: Vec<[f32; 3]>,
        mut rotations: Vec<[f32; 4]>,
        opacities: Vec<f32>,
        sh0: Vec<[f32; 3]>,
        sh_rest: Option<ShRest>,
    ) -> Result<Self, RenderError> {
        let n = positions.len();
        if scales.len() != n || rotations.len() != n || opacities.len() != n || sh0.len() != n {
            return Err(RenderError::Upload(format!(
                "splat attribute lengths differ: positions {n}, scales {}, rotations {}, \
                 opacities {}, sh0 {}",
                scales.len(),
                rotations.len(),
                opacities.len(),
                sh0.len()
            )));
        }
        if let Some(rest) = &sh_rest {
            if rest.degree == 0 || rest.degree > 3 {
                return Err(RenderError::Upload(format!(
                    "splat SH degree must be 1..=3 when higher bands are present, got {}",
                    rest.degree
                )));
            }
            if rest.coeffs.len() != n * rest.per_splat() {
                return Err(RenderError::Upload(format!(
                    "splat SH band length {} does not match {n} splats x {} coefficients",
                    rest.coeffs.len(),
                    rest.per_splat()
                )));
            }
            if rest.coeffs.iter().flatten().any(|v| !v.is_finite()) {
                return Err(RenderError::Upload(
                    "splat SH bands contain non-finite coefficients".into(),
                ));
            }
        }
        for i in 0..n {
            if positions[i].iter().any(|v| !v.is_finite()) {
                return Err(RenderError::Upload(format!(
                    "splat {i} has a non-finite position"
                )));
            }
            if scales[i].iter().any(|v| !(v.is_finite() && *v > 0.0)) {
                return Err(RenderError::Upload(format!(
                    "splat {i} has a non-positive or non-finite scale {:?}",
                    scales[i]
                )));
            }
            let q = rotations[i];
            let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            if !(norm.is_finite() && norm > 1e-12) {
                return Err(RenderError::Upload(format!(
                    "splat {i} has a degenerate rotation quaternion {q:?}"
                )));
            }
            rotations[i] = [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm];
            if !(opacities[i].is_finite() && (0.0..=1.0).contains(&opacities[i])) {
                return Err(RenderError::Upload(format!(
                    "splat {i} opacity {} is outside [0, 1]",
                    opacities[i]
                )));
            }
            if sh0[i].iter().any(|v| !v.is_finite()) {
                return Err(RenderError::Upload(format!(
                    "splat {i} has a non-finite DC colour"
                )));
            }
        }
        let inv_cov = (0..n)
            .map(|i| inverse_covariance(scales[i], rotations[i]))
            .collect::<Vec<_>>();
        let rest_bytes = sh_rest.as_ref().map_or(0, |rest| rest.coeffs.len() * 12);
        let bytes = n * Self::BYTES_PER_SPLAT + rest_bytes;
        let tracker = tracked_host_allocation(bytes as u64, "splat-cloud-soa")?;
        Ok(Self {
            positions,
            scales,
            rotations,
            opacities,
            sh0,
            sh_rest,
            inv_cov,
            _tracker: tracker,
        })
    }

    /// Host bytes per splat without higher SH bands: position, scale,
    /// rotation, opacity, DC colour and the packed inverse covariance.
    pub const BYTES_PER_SPLAT: usize = 12 + 12 + 16 + 4 + 12 + 24;

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Stored SH degree (0 when only the DC term is present).
    pub fn sh_degree(&self) -> u32 {
        self.sh_rest.as_ref().map_or(0, |rest| rest.degree)
    }

    /// Packed upper-triangular inverse covariance per splat.
    pub fn inv_cov(&self) -> &[[f32; 6]] {
        &self.inv_cov
    }

    /// Recompute the packed inverse covariance after editing `scales` or
    /// `rotations` in place.
    pub fn rebuild_inverse_covariance(&mut self) {
        self.inv_cov = (0..self.len())
            .map(|i| inverse_covariance(self.scales[i], self.rotations[i]))
            .collect();
    }

    /// Host bytes this cloud keeps resident.
    pub fn byte_size(&self) -> usize {
        self.len() * Self::BYTES_PER_SPLAT
            + self
                .sh_rest
                .as_ref()
                .map_or(0, |rest| rest.coeffs.len() * 12)
    }

    /// World-space bounds of the 3-sigma ellipsoids (the per-splat proxies).
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        if self.is_empty() {
            return None;
        }
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for i in 0..self.len() {
            let (a, b) = self.splat_aabb(i);
            for axis in 0..3 {
                lo[axis] = lo[axis].min(a[axis]);
                hi[axis] = hi[axis].max(b[axis]);
            }
        }
        Some((lo, hi))
    }

    /// Bounding proxy of splat `i`: the AABB of its 3-sigma ellipsoid.
    pub fn splat_aabb(&self, i: usize) -> ([f32; 3], [f32; 3]) {
        let half = sigma_extent(self.scales[i], self.rotations[i], kernel::SIGMA_CUTOFF);
        let p = self.positions[i];
        (
            [p[0] - half[0], p[1] - half[1], p[2] - half[2]],
            [p[0] + half[0], p[1] + half[1], p[2] + half[2]],
        )
    }

    /// View-dependent colour of splat `i` seen along unit direction `dir`
    /// (from the viewer toward the splat), evaluating every stored SH band.
    pub fn color(&self, i: usize, dir: [f32; 3]) -> [f32; 3] {
        let rest = self.sh_rest.as_ref().map(|rest| {
            let k = rest.per_splat();
            (rest.degree, &rest.coeffs[i * k..(i + 1) * k])
        });
        eval_sh(self.sh0[i], rest, dir)
    }

    /// Apply a similarity transform (uniform scale, rotation quaternion
    /// `(w, x, y, z)`, translation) in place: positions, sigmas and
    /// orientations move together and the inverse covariance is rebuilt.
    /// SH bands are left in the source frame (the DC term is rotation
    /// invariant; higher bands are view-dependent detail).
    pub fn apply_similarity(
        &mut self,
        scale: f32,
        rotation: [f32; 4],
        translation: [f32; 3],
    ) -> Result<(), RenderError> {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(RenderError::Upload(format!(
                "splat similarity scale must be finite and > 0, got {scale}"
            )));
        }
        let norm = rotation.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !(norm.is_finite() && norm > 1e-12) || translation.iter().any(|v| !v.is_finite()) {
            return Err(RenderError::Upload(
                "splat similarity rotation/translation must be finite and non-degenerate".into(),
            ));
        }
        let q = rotation.map(|v| v / norm);
        let m = quat_to_mat3(q);
        for i in 0..self.len() {
            let p = self.positions[i];
            for axis in 0..3 {
                self.positions[i][axis] =
                    scale * (m[axis][0] * p[0] + m[axis][1] * p[1] + m[axis][2] * p[2])
                        + translation[axis];
            }
            self.scales[i] = self.scales[i].map(|s| s * scale);
            self.rotations[i] = quat_mul(q, self.rotations[i]);
        }
        self.rebuild_inverse_covariance();
        Ok(())
    }
}

/// Hamilton product `a * b` of two `(w, x, y, z)` quaternions.
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Rotation matrix (row-major) of a unit quaternion `(w, x, y, z)`.
pub fn quat_to_mat3(q: [f32; 4]) -> [[f32; 3]; 3] {
    let [w, x, y, z] = q;
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Packed upper triangle (xx, xy, xz, yy, yz, zz) of `R diag(d) R^T`.
fn rotated_diagonal(d: [f32; 3], rotation: [f32; 4]) -> [f32; 6] {
    let r = quat_to_mat3(rotation);
    let entry =
        |i: usize, j: usize| r[i][0] * d[0] * r[j][0] + r[i][1] * d[1] * r[j][1] + r[i][2] * d[2] * r[j][2];
    [
        entry(0, 0),
        entry(0, 1),
        entry(0, 2),
        entry(1, 1),
        entry(1, 2),
        entry(2, 2),
    ]
}

/// Packed covariance `R diag(sigma^2) R^T`.
pub fn covariance(scale: [f32; 3], rotation: [f32; 4]) -> [f32; 6] {
    rotated_diagonal(scale.map(|s| s * s), rotation)
}

/// Packed inverse covariance `R diag(1 / sigma^2) R^T` — the quadratic form
/// the analytic ray/Gaussian kernel evaluates.
pub fn inverse_covariance(scale: [f32; 3], rotation: [f32; 4]) -> [f32; 6] {
    rotated_diagonal(scale.map(|s| 1.0 / (s * s)), rotation)
}

/// Expand a packed symmetric matrix to its dense row-major form.
pub fn unpack_symmetric(m: [f32; 6]) -> [[f32; 3]; 3] {
    [[m[0], m[1], m[2]], [m[1], m[3], m[4]], [m[2], m[4], m[5]]]
}

/// `M v` for a packed symmetric matrix.
pub fn symmetric_mul(m: [f32; 6], v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[1] * v[0] + m[3] * v[1] + m[4] * v[2],
        m[2] * v[0] + m[4] * v[1] + m[5] * v[2],
    ]
}

/// Half extents of the `k`-sigma ellipsoid's axis-aligned bounding box.
pub fn sigma_extent(scale: [f32; 3], rotation: [f32; 4], k: f32) -> [f32; 3] {
    let cov = covariance(scale, rotation);
    [
        k * cov[0].max(0.0).sqrt(),
        k * cov[3].max(0.0).sqrt(),
        k * cov[5].max(0.0).sqrt(),
    ]
}

/// Evaluate the 3DGS spherical-harmonic colour: `0.5 + sum(basis * coeff)`,
/// clamped at zero. `rest` carries the degree and the non-DC coefficients.
pub fn eval_sh(sh0: [f32; 3], rest: Option<(u32, &[[f32; 3]])>, dir: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = dir;
    let mut out = sh0.map(|c| SH_C0 * c);
    if let Some((degree, c)) = rest {
        let mut add = |weight: f32, k: usize| {
            for (channel, value) in out.iter_mut().enumerate() {
                *value += weight * c[k][channel];
            }
        };
        if degree >= 1 {
            add(-SH_C1 * y, 0);
            add(SH_C1 * z, 1);
            add(-SH_C1 * x, 2);
        }
        if degree >= 2 {
            let (xx, yy, zz) = (x * x, y * y, z * z);
            add(SH_C2[0] * x * y, 3);
            add(SH_C2[1] * y * z, 4);
            add(SH_C2[2] * (2.0 * zz - xx - yy), 5);
            add(SH_C2[3] * x * z, 6);
            add(SH_C2[4] * (xx - yy), 7);
            if degree >= 3 {
                add(SH_C3[0] * y * (3.0 * xx - yy), 8);
                add(SH_C3[1] * x * y * z, 9);
                add(SH_C3[2] * y * (4.0 * zz - xx - yy), 10);
                add(SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), 11);
                add(SH_C3[4] * x * (4.0 * zz - xx - yy), 12);
                add(SH_C3[5] * z * (xx - yy), 13);
                add(SH_C3[6] * x * (xx - 3.0 * yy), 14);
            }
        }
    }
    out.map(|c| (c + 0.5).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud_of(scales: [f32; 3], rotation: [f32; 4]) -> GaussianSplatCloud {
        GaussianSplatCloud::from_parts(
            vec![[1.0, 2.0, 3.0]],
            vec![scales],
            vec![rotation],
            vec![0.8],
            vec![[0.3, -0.2, 0.1]],
            None,
        )
        .unwrap()
    }

    fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
            }
        }
        out
    }

    #[test]
    fn inverse_covariance_packing_round_trips_against_r_diag_rt() {
        // An arbitrary non-axis-aligned orientation and anisotropic sigmas.
        let scales = [0.5f32, 1.5, 0.125];
        let rotation = [0.6f32, 0.2, -0.5, 0.4];
        let cloud = cloud_of(scales, rotation);
        let q = cloud.rotations[0];
        let r = quat_to_mat3(q);
        let mut rt = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                rt[i][j] = r[j][i];
            }
        }
        let diag = |d: [f32; 3]| [[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]];
        let expected = mat_mul(mat_mul(r, diag(scales.map(|s| 1.0 / (s * s)))), rt);
        let unpacked = unpack_symmetric(cloud.inv_cov()[0]);
        for i in 0..3 {
            for j in 0..3 {
                let tolerance = 1e-5 * expected[i][j].abs().max(1.0);
                assert!(
                    (unpacked[i][j] - expected[i][j]).abs() <= tolerance,
                    "entry ({i},{j}): {} vs {}",
                    unpacked[i][j],
                    expected[i][j]
                );
            }
        }
        // Sigma * Sigma^-1 = I closes the loop on the packed pair.
        let product = mat_mul(
            unpack_symmetric(covariance(scales, q)),
            unpack_symmetric(inverse_covariance(scales, q)),
        );
        for i in 0..3 {
            for j in 0..3 {
                let identity = if i == j { 1.0 } else { 0.0 };
                assert!((product[i][j] - identity).abs() < 2e-4, "{product:?}");
            }
        }
    }

    #[test]
    fn rotation_is_orthonormal_and_quaternions_are_normalized() {
        let cloud = cloud_of([1.0, 2.0, 3.0], [2.0, 0.0, 0.0, 2.0]);
        let q = cloud.rotations[0];
        assert!((q.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
        let r = quat_to_mat3(q);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f32 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn three_sigma_proxy_bounds_the_ellipsoid() {
        let scales = [0.5f32, 1.5, 0.125];
        let rotation = [0.6f32, 0.2, -0.5, 0.4];
        let cloud = cloud_of(scales, rotation);
        let (lo, hi) = cloud.splat_aabb(0);
        let r = quat_to_mat3(cloud.rotations[0]);
        // Sample the 3-sigma ellipsoid surface along a lattice of directions.
        for a in 0..24 {
            for b in 0..12 {
                let theta = num::f32_from_usize(a) * std::f32::consts::TAU / 24.0;
                let phi = num::f32_from_usize(b) * std::f32::consts::PI / 11.0;
                let local = [
                    3.0 * scales[0] * phi.sin() * theta.cos(),
                    3.0 * scales[1] * phi.sin() * theta.sin(),
                    3.0 * scales[2] * phi.cos(),
                ];
                for axis in 0..3 {
                    let world = cloud.positions[0][axis]
                        + r[axis][0] * local[0]
                        + r[axis][1] * local[1]
                        + r[axis][2] * local[2];
                    assert!(world >= lo[axis] - 1e-4 && world <= hi[axis] + 1e-4);
                }
            }
        }
    }

    #[test]
    fn dc_only_colour_is_view_independent() {
        let cloud = cloud_of([1.0; 3], [1.0, 0.0, 0.0, 0.0]);
        let a = cloud.color(0, [0.0, 0.0, 1.0]);
        let b = cloud.color(0, [1.0, 0.0, 0.0]);
        assert_eq!(a, b);
        assert!((a[0] - (0.5 + SH_C0 * 0.3)).abs() < 1e-6);
    }

    #[test]
    fn invalid_attributes_are_rejected_with_diagnostics() {
        let bad_scale = GaussianSplatCloud::from_parts(
            vec![[0.0; 3]],
            vec![[1.0, 0.0, 1.0]],
            vec![[1.0, 0.0, 0.0, 0.0]],
            vec![0.5],
            vec![[0.0; 3]],
            None,
        );
        assert!(format!("{}", bad_scale.unwrap_err()).contains("scale"));
        let bad_len = GaussianSplatCloud::from_parts(
            vec![[0.0; 3]],
            vec![],
            vec![[1.0, 0.0, 0.0, 0.0]],
            vec![0.5],
            vec![[0.0; 3]],
            None,
        );
        assert!(format!("{}", bad_len.unwrap_err()).contains("lengths differ"));
    }

    #[test]
    fn similarity_keeps_inverse_covariance_consistent() {
        let mut cloud = cloud_of([0.5, 1.5, 0.25], [0.6, 0.2, -0.5, 0.4]);
        cloud
            .apply_similarity(2.0, [0.7, 0.1, 0.7, -0.1], [10.0, -4.0, 3.0])
            .unwrap();
        let expected = inverse_covariance(cloud.scales[0], cloud.rotations[0]);
        assert_eq!(cloud.inv_cov()[0], expected);
        assert_eq!(cloud.scales[0], [1.0, 3.0, 0.5]);
    }
}
