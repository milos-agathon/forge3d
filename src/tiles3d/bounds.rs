//! Bounding volume types for 3D Tiles
//!
//! World-space math here is f64 end to end (MENSURA): geodetic regions go
//! through the full-precision geocentric conversion and are never truncated
//! to f32 — ECEF magnitudes are ~6.38e6 m, where an f32 mantissa costs ≈0.5 m.

use glam::{DMat4, DVec3};
use serde::{Deserialize, Serialize};

use crate::geo::units::{Ellipsoidal, Height};

/// Bounding volume for a 3D Tile
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum BoundingVolume {
    /// Axis-aligned bounding box
    Box(BoundingBox),
    /// Bounding sphere
    Sphere(BoundingSphere),
    /// Geographic region (WGS84)
    Region(BoundingRegion),
}

/// Oriented bounding box defined by center and half-axes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    /// 12 world-space coordinates: [cx, cy, cz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
    /// center (3) + x-axis half-length (3) + y-axis (3) + z-axis (3)
    #[serde(rename = "box")]
    pub data: [f64; 12],
}

/// Bounding sphere defined by center and radius
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingSphere {
    /// 4 world-space coordinates: [cx, cy, cz, radius]
    pub sphere: [f64; 4],
}

/// Geographic bounding region in WGS84
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingRegion {
    /// 6 floats: [west, south, east, north, min_height, max_height]
    /// Longitude/latitude in radians, heights in meters
    pub region: [f64; 6],
}

impl BoundingVolume {
    /// Get the center point of the bounding volume, in f64 world space.
    pub fn center(&self) -> DVec3 {
        match self {
            Self::Box(b) => DVec3::new(b.data[0], b.data[1], b.data[2]),
            Self::Sphere(s) => DVec3::new(s.sphere[0], s.sphere[1], s.sphere[2]),
            Self::Region(r) => {
                let lon = (r.region[0] + r.region[2]) / 2.0;
                let lat = (r.region[1] + r.region[3]) / 2.0;
                // 3D Tiles region heights are ellipsoidal metres per spec §8.
                let height = Height::<Ellipsoidal>::new((r.region[4] + r.region[5]) / 2.0);
                wgs84_to_ecef(lon, lat, height)
            }
        }
    }

    /// Get approximate radius for SSE calculation, in metres (f64).
    pub fn radius(&self) -> f64 {
        match self {
            Self::Box(b) => {
                let x_len = DVec3::new(b.data[3], b.data[4], b.data[5]).length();
                let y_len = DVec3::new(b.data[6], b.data[7], b.data[8]).length();
                let z_len = DVec3::new(b.data[9], b.data[10], b.data[11]).length();
                (x_len * x_len + y_len * y_len + z_len * z_len).sqrt()
            }
            Self::Sphere(s) => s.sphere[3],
            Self::Region(r) => {
                // region[] is [west, south, east, north] in RADIANS. The
                // east-west arc shrinks with cos(latitude); use the region's
                // central latitude and the half-diagonal of the box.
                const R: f64 = 6_378_137.0;
                let d_lon = (r.region[2] - r.region[0]).abs();
                let d_lat = (r.region[3] - r.region[1]).abs();
                let d_h = (r.region[5] - r.region[4]).abs();
                let lat_c = (r.region[1] + r.region[3]) / 2.0;
                let half_ew = d_lon / 2.0 * R * lat_c.cos();
                let half_ns = d_lat / 2.0 * R;
                let half_h = d_h / 2.0;
                (half_ew * half_ew + half_ns * half_ns + half_h * half_h).sqrt()
            }
        }
    }

    /// Transform the bounding volume by a matrix
    pub fn transform(&self, matrix: &DMat4) -> Self {
        match self {
            Self::Box(b) => {
                let center = matrix.transform_point3(DVec3::new(b.data[0], b.data[1], b.data[2]));
                let x_axis = matrix.transform_vector3(DVec3::new(b.data[3], b.data[4], b.data[5]));
                let y_axis = matrix.transform_vector3(DVec3::new(b.data[6], b.data[7], b.data[8]));
                let z_axis =
                    matrix.transform_vector3(DVec3::new(b.data[9], b.data[10], b.data[11]));
                Self::Box(BoundingBox {
                    data: [
                        center.x, center.y, center.z, x_axis.x, x_axis.y, x_axis.z, y_axis.x,
                        y_axis.y, y_axis.z, z_axis.x, z_axis.y, z_axis.z,
                    ],
                })
            }
            Self::Sphere(s) => {
                let center =
                    matrix.transform_point3(DVec3::new(s.sphere[0], s.sphere[1], s.sphere[2]));
                let scale = matrix.to_scale_rotation_translation().0;
                let max_scale = scale.x.max(scale.y).max(scale.z);
                Self::Sphere(BoundingSphere {
                    sphere: [center.x, center.y, center.z, s.sphere[3] * max_scale],
                })
            }
            Self::Region(_) => self.clone(),
        }
    }

    /// Check if this volume intersects a frustum (simplified AABB check).
    /// Clip-space math runs in f64 so ECEF-magnitude centers keep precision.
    pub fn intersects_frustum(&self, view_proj: &DMat4) -> bool {
        let center = self.center();
        let radius = self.radius();
        let clip = *view_proj * center.extend(1.0);
        if clip.w <= 0.0 {
            return radius > clip.z.abs();
        }
        let ndc = clip.truncate() / clip.w;
        let margin = radius / clip.w;
        ndc.x.abs() <= 1.0 + margin
            && ndc.y.abs() <= 1.0 + margin
            && ndc.z >= -margin
            && ndc.z <= 1.0 + margin
    }
}

/// Convert WGS84 geodetic coordinates (radians) to ECEF, full f64.
///
/// The height parameter is TYPED: only an ellipsoidal height is accepted.
/// Orthometric DEM heights must first go through
/// `crate::geo::geoid::orthometric_to_ellipsoidal`.
pub fn wgs84_to_ecef(lon_rad: f64, lat_rad: f64, height: Height<Ellipsoidal>) -> DVec3 {
    crate::geo::projections::geocentric::wgs84_geodetic_to_ecef(
        lon_rad.to_degrees(),
        lat_rad.to_degrees(),
        height.metres(),
    )
    .expect("finite geodetic inputs")
}

impl Default for BoundingVolume {
    fn default() -> Self {
        Self::Sphere(BoundingSphere {
            sphere: [0.0, 0.0, 0.0, 1.0],
        })
    }
}
