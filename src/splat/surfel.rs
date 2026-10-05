// src/splat/surfel.rs
// LiDAR surfels: per-point normal estimation that turns returns on locally
// planar surfaces (ground, roofs, walls) into oriented discs, and the octahedral
// normal packing the GPU record carries. Points without a planar
// neighbourhood keep the isotropic sphelet (`SURFEL_SPHERE`).
// RELEVANT FILES: src/splat/stream.rs, src/shaders/fusion/unified_occlusion.wgsl,
//                 src/splat/kernel.rs

use std::collections::HashMap;

/// `normal_oct` value of a point rendered as an isotropic sphelet. Never
/// produced by [`oct_encode`].
pub const SURFEL_SPHERE: u32 = 0xFFFF_FFFF;

/// Neighbours (within `NEIGHBOUR_RADII` sphelet radii) a point needs before
/// its neighbourhood is tested for planarity.
const MIN_NEIGHBOURS: usize = 6;
const NEIGHBOUR_RADII: f64 = 3.0;
/// Planarity: smallest eigenvalue at most this share of the middle one...
const PLANAR_FLATNESS: f64 = 0.08;
/// ...and the neighbourhood not a line (middle at least this share of the
/// largest).
const PLANAR_SPREAD: f64 = 0.2;

fn oct_pack(x: f64, y: f64) -> u32 {
    let q = |v: f64| ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 65535.0).round() as u32;
    let packed = q(x) | (q(y) << 16);
    // The (+1, +1) corner is the -z pole, which every octahedron corner
    // encodes; use the (-1, -1) corner so the sentinel stays unique.
    if packed == SURFEL_SPHERE {
        0
    } else {
        packed
    }
}

fn oct_encode_f64(n: [f64; 3]) -> u32 {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    oct_pack(x, y)
}

/// Octahedral packing of a unit normal into 2 x 16 bits. The WGSL
/// `fusion_oct_decode` (and [`oct_decode`]) is its exact inverse up to the
/// quantisation step.
pub fn oct_encode(n: [f32; 3]) -> u32 {
    oct_encode_f64(n.map(f64::from))
}

/// CPU mirror of the WGSL `fusion_oct_decode`.
pub fn oct_decode(packed: u32) -> [f32; 3] {
    let lo = u16::try_from(packed & 0xffff).unwrap_or(u16::MAX);
    let hi = u16::try_from(packed >> 16).unwrap_or(u16::MAX);
    let ex = f32::from(lo) * (2.0 / 65535.0) - 1.0;
    let ey = f32::from(hi) * (2.0 / 65535.0) - 1.0;
    let mut n = [ex, ey, 1.0 - ex.abs() - ey.abs()];
    let t = (-n[2]).max(0.0);
    n[0] += if n[0] >= 0.0 { -t } else { t };
    n[1] += if n[1] >= 0.0 { -t } else { t };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    [n[0] / l, n[1] / l, n[2] / l]
}

/// Eigen-decomposition of a symmetric 3x3 matrix by cyclic Jacobi rotations
/// (at most 32 sweeps; stops once the off-diagonal sum is below 1e-18).
/// Returns eigenvalues ascending and `vectors[i]`, the unit eigenvector of
/// `values[i]`.
pub fn sym_eigen3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..32 {
        if a[0][1].abs() + a[0][2].abs() + a[1][2].abs() < 1e-18 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in a.iter_mut() {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for row in v.iter_mut() {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let values = order.map(|i| a[i][i]);
    let vectors = order.map(|i| [v[0][i], v[1][i], v[2][i]]);
    (values, vectors)
}

/// Per-point surfel normals (`normal_oct`) of one page of LiDAR returns.
///
/// A point whose neighbours within `3 * radius` (grid hash of cell
/// `2 * radius`, self excluded) number at least 6 and span a plane
/// (eigenvalues of their covariance `l0 <= 0.08 * l1` and `l1 >= 0.2 * l2`)
/// becomes a disc with the smallest-eigenvalue normal, flipped so `n.y >= 0`
/// (for `|n.y| < 1e-6`: the first non-zero component positive). Every other
/// point stays a sphelet (`SURFEL_SPHERE`). Deterministic: the result depends
/// only on `positions` and `radius`.
pub fn estimate_point_surfels(positions: &[[f32; 3]], radius: f32) -> Vec<u32> {
    let r = f64::from(radius);
    if positions.is_empty() || !(r.is_finite() && r > 0.0) {
        return vec![SURFEL_SPHERE; positions.len()];
    }
    let cell = 2.0 * r;
    let reach = NEIGHBOUR_RADII * r;
    let reach2 = reach * reach;
    let p64: Vec<[f64; 3]> = positions.iter().map(|p| p.map(f64::from)).collect();
    let key = |p: [f64; 3]| p.map(|v| (v / cell).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, &p) in p64.iter().enumerate() {
        grid.entry(key(p))
            .or_default()
            .push(u32::try_from(i).unwrap_or(u32::MAX));
    }
    let span = (reach / cell).ceil() as i64;
    let mut out = Vec::with_capacity(positions.len());
    let mut neighbours: Vec<usize> = Vec::new();
    for (i, &p) in p64.iter().enumerate() {
        neighbours.clear();
        let k = key(p);
        for dx in -span..=span {
            for dy in -span..=span {
                for dz in -span..=span {
                    let Some(members) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) else {
                        continue;
                    };
                    for &j in members {
                        let j = j as usize;
                        if j == i {
                            continue;
                        }
                        let q = p64[j];
                        let d2 = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) + (q[2] - p[2]).powi(2);
                        if d2 <= reach2 {
                            neighbours.push(j);
                        }
                    }
                }
            }
        }
        if neighbours.len() < MIN_NEIGHBOURS {
            out.push(SURFEL_SPHERE);
            continue;
        }
        let count = f64::from(u32::try_from(neighbours.len() + 1).unwrap_or(u32::MAX));
        let mut mean = p;
        for &j in &neighbours {
            for a in 0..3 {
                mean[a] += p64[j][a];
            }
        }
        mean = mean.map(|v| v / count);
        let mut cov = [[0.0f64; 3]; 3];
        for q in std::iter::once(p).chain(neighbours.iter().map(|&j| p64[j])) {
            let d = [q[0] - mean[0], q[1] - mean[1], q[2] - mean[2]];
            for (r_, row) in cov.iter_mut().enumerate() {
                for (c, value) in row.iter_mut().enumerate() {
                    *value += d[r_] * d[c];
                }
            }
        }
        let (values, vectors) = sym_eigen3(cov);
        let planar = values[0] <= PLANAR_FLATNESS * values[1] && values[1] >= PLANAR_SPREAD * values[2];
        if !planar {
            out.push(SURFEL_SPHERE);
            continue;
        }
        let mut n = vectors[0];
        let flip = if n[1].abs() < 1e-6 {
            n.iter().find(|v| **v != 0.0).is_some_and(|v| *v < 0.0)
        } else {
            n[1] < 0.0
        };
        if flip {
            n = n.map(|v| -v);
        }
        out.push(oct_encode_f64(n));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::splat::stream::hash_unit;

    fn unit(k: u32) -> [f32; 3] {
        let z = hash_unit(3 * k + 1) * 2.0 - 1.0;
        let phi = hash_unit(3 * k + 2) * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        [r * phi.cos(), z, r * phi.sin()]
    }

    /// Angle between two directions as atan2(|a x b|, a . b): well
    /// conditioned near 0 and insensitive to the f32 normalisation error
    /// that makes acos(a . b) report ~0.03 deg for identical directions.
    fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f64 {
        let (a, b) = (a.map(f64::from), b.map(f64::from));
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let sin = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        let cos = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        sin.atan2(cos).to_degrees()
    }

    #[test]
    fn oct_round_trip_error_is_below_0_02_degrees() {
        let mut worst = 0.0f64;
        for k in 0..10_000 {
            let n = unit(k);
            let packed = oct_encode(n);
            assert_ne!(packed, SURFEL_SPHERE, "{n:?}");
            worst = worst.max(angle_deg(n, oct_decode(packed)));
        }
        for n in [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ] {
            let packed = oct_encode(n);
            assert_ne!(packed, SURFEL_SPHERE, "{n:?}");
            worst = worst.max(angle_deg(n, oct_decode(packed)));
        }
        println!("oct round trip: worst error {worst:.5} deg");
        assert!(worst <= 0.02, "{worst}");
    }

    #[test]
    fn sym_eigen3_decomposes_symmetric_matrices() {
        for k in 0..1000u32 {
            // Random SPD: B^T B + 0.01 I.
            let b: Vec<f64> = (0..9)
                .map(|j| f64::from(hash_unit(k * 9 + j + 77)) * 2.0 - 1.0)
                .collect();
            let mut a = [[0.0f64; 3]; 3];
            for (i, row) in a.iter_mut().enumerate() {
                for (j, value) in row.iter_mut().enumerate() {
                    *value = (0..3).map(|r| b[r * 3 + i] * b[r * 3 + j]).sum::<f64>();
                    if i == j {
                        *value += 0.01;
                    }
                }
            }
            let norm = a.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
            let (values, vectors) = sym_eigen3(a);
            assert!(values[0] <= values[1] && values[1] <= values[2]);
            for i in 0..3 {
                let v = vectors[i];
                for r in 0..3 {
                    let av = a[r][0] * v[0] + a[r][1] * v[1] + a[r][2] * v[2];
                    assert!((av - values[i] * v[r]).abs() <= 1e-9 * norm, "matrix {k}");
                }
                for j in 0..3 {
                    let d = (0..3).map(|r| vectors[i][r] * vectors[j][r]).sum::<f64>();
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((d - expect).abs() <= 1e-9, "matrix {k}: <v{i}, v{j}> = {d}");
                }
            }
        }
    }

    fn jittered_grid(axis_normal: usize) -> Vec<[f32; 3]> {
        let mut points = Vec::new();
        for a in 0..40u32 {
            for b in 0..40u32 {
                let k = a * 40 + b;
                let j = |s: u32, amp: f32| (hash_unit(k * 5 + s) * 2.0 - 1.0) * amp;
                let (u, v) = (
                    crate::splat::num::f32_from_u32(a) + j(1, 0.05),
                    crate::splat::num::f32_from_u32(b) + j(2, 0.05),
                );
                let off = j(3, 0.02);
                points.push(match axis_normal {
                    1 => [u, off, v],
                    _ => [off, u, v],
                });
            }
        }
        points
    }

    fn interior(i: usize) -> bool {
        let (a, b) = (i / 40, i % 40);
        (3..37).contains(&a) && (3..37).contains(&b)
    }

    #[test]
    fn flat_grid_becomes_horizontal_surfels() {
        let points = jittered_grid(1);
        let packed = estimate_point_surfels(&points, 0.8);
        let (mut total, mut good) = (0, 0);
        for (_, &p) in packed.iter().enumerate().filter(|(i, _)| interior(*i)) {
            total += 1;
            if p != SURFEL_SPHERE && oct_decode(p)[1].abs() >= 0.995 {
                good += 1;
            }
        }
        println!("flat grid: {good} of {total} interior points are horizontal surfels");
        assert!(good * 100 >= total * 95, "{good} of {total}");
    }

    #[test]
    fn vertical_wall_becomes_vertical_surfels() {
        let points = jittered_grid(0);
        let packed = estimate_point_surfels(&points, 0.8);
        let (mut total, mut good) = (0, 0);
        for (_, &p) in packed.iter().enumerate().filter(|(i, _)| interior(*i)) {
            total += 1;
            if p != SURFEL_SPHERE && oct_decode(p)[0].abs() >= 0.99 {
                good += 1;
            }
        }
        println!("vertical wall: {good} of {total} interior points are vertical surfels");
        assert!(good * 100 >= total * 95, "{good} of {total}");
    }

    #[test]
    fn isotropic_cloud_stays_spheres() {
        let mut points = Vec::new();
        let mut k = 0u32;
        while points.len() < 2000 {
            let p = [
                (hash_unit(k * 3 + 11) * 2.0 - 1.0) * 6.0,
                (hash_unit(k * 3 + 12) * 2.0 - 1.0) * 6.0,
                (hash_unit(k * 3 + 13) * 2.0 - 1.0) * 6.0,
            ];
            k += 1;
            if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 36.0 {
                points.push(p);
            }
        }
        let packed = estimate_point_surfels(&points, 0.8);
        let surfels = packed.iter().filter(|&&p| p != SURFEL_SPHERE).count();
        println!("isotropic ball: {surfels} of {} points are surfels", points.len());
        assert!(surfels * 100 <= points.len() * 5, "{surfels}");
    }

    #[test]
    fn estimator_is_deterministic() {
        let mut points = jittered_grid(1);
        points.extend(jittered_grid(0));
        assert_eq!(
            estimate_point_surfels(&points, 0.8),
            estimate_point_surfels(&points, 0.8)
        );
    }
}
