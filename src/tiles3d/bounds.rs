//! Bounding volume types for 3D Tiles

use glam::{Vec3, Mat4};
use serde::{Deserialize, Serialize};

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
    /// 12 floats: [cx, cy, cz, xx, xy, xz, yx, yy, yz, zx, zy, zz]
    /// center (3) + x-axis half-length (3) + y-axis (3) + z-axis (3)
    #[serde(rename = "box")]
    pub data: [f32; 12],
}

/// Bounding sphere defined by center and radius
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingSphere {
    /// 4 floats: [cx, cy, cz, radius]
    pub sphere: [f32; 4],
}

/// Geographic bounding region in WGS84
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingRegion {
    /// 6 floats: [west, south, east, north, min_height, max_height]
    /// Longitude/latitude in radians, heights in meters
    pub region: [f64; 6],
}

impl BoundingVolume {
    /// Get the center point of the bounding volume
    pub fn center(&self) -> Vec3 {
        match self {
            Self::Box(b) => Vec3::new(b.data[0], b.data[1], b.data[2]),
            Self::Sphere(s) => Vec3::new(s.sphere[0], s.sphere[1], s.sphere[2]),
            Self::Region(r) => {
                let lon = (r.region[0] + r.region[2]) / 2.0;
                let lat = (r.region[1] + r.region[3]) / 2.0;
                let height = (r.region[4] + r.region[5]) / 2.0;
                wgs84_to_ecef(lon, lat, height)
            }
        }
    }

    /// Get approximate radius for SSE calculation
    pub fn radius(&self) -> f32 {
        match self {
            Self::Box(b) => {
                let x_len = Vec3::new(b.data[3], b.data[4], b.data[5]).length();
                let y_len = Vec3::new(b.data[6], b.data[7], b.data[8]).length();
                let z_len = Vec3::new(b.data[9], b.data[10], b.data[11]).length();
                (x_len * x_len + y_len * y_len + z_len * z_len).sqrt()
            }
            Self::Sphere(s) => s.sphere[3],
            Self::Region(r) => {
                let d_lon = (r.region[2] - r.region[0]).abs();
                let d_lat = (r.region[3] - r.region[1]).abs();
                let d_h = (r.region[5] - r.region[4]).abs();
                let approx_meters = (d_lon.max(d_lat) * 6378137.0) as f32;
                (approx_meters * approx_meters + (d_h as f32) * (d_h as f32)).sqrt() / 2.0
            }
        }
    }

    /// Transform the bounding volume by a matrix
    pub fn transform(&self, matrix: &Mat4) -> Self {
        match self {
            Self::Box(b) => {
                let center = matrix.transform_point3(Vec3::new(b.data[0], b.data[1], b.data[2]));
                let x_axis = matrix.transform_vector3(Vec3::new(b.data[3], b.data[4], b.data[5]));
                let y_axis = matrix.transform_vector3(Vec3::new(b.data[6], b.data[7], b.data[8]));
                let z_axis = matrix.transform_vector3(Vec3::new(b.data[9], b.data[10], b.data[11]));
                Self::Box(BoundingBox {
                    data: [
                        center.x, center.y, center.z,
                        x_axis.x, x_axis.y, x_axis.z,
                        y_axis.x, y_axis.y, y_axis.z,
                        z_axis.x, z_axis.y, z_axis.z,
                    ],
                })
            }
            Self::Sphere(s) => {
                let center = matrix.transform_point3(Vec3::new(s.sphere[0], s.sphere[1], s.sphere[2]));
                let scale = matrix.to_scale_rotation_translation().0;
                let max_scale = scale.x.max(scale.y).max(scale.z);
                Self::Sphere(BoundingSphere {
                    sphere: [center.x, center.y, center.z, s.sphere[3] * max_scale],
                })
            }
            Self::Region(_) => self.clone(),
        }
    }

    /// Check if this volume intersects a frustum (simplified AABB check)
    pub fn intersects_frustum(&self, view_proj: &Mat4) -> bool {
        let center = self.center();
        let radius = self.radius();
        let clip = *view_proj * center.extend(1.0);
        if clip.w <= 0.0 {
            return radius > clip.z.abs();
        }
        let ndc = clip.truncate() / clip.w;
        let margin = radius / clip.w;
        ndc.x.abs() <= 1.0 + margin && ndc.y.abs() <= 1.0 + margin && ndc.z >= -margin && ndc.z <= 1.0 + margin
    }
}

/// Convert WGS84 geodetic coordinates to ECEF
fn wgs84_to_ecef(lon_rad: f64, lat_rad: f64, height: f64) -> Vec3 {
    const A: f64 = 6378137.0;
    const E2: f64 = 0.00669437999014;
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let sin_lon = lon_rad.sin();
    let cos_lon = lon_rad.cos();
    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
    let x = (n + height) * cos_lat * cos_lon;
    let y = (n + height) * cos_lat * sin_lon;
    let z = (n * (1.0 - E2) + height) * sin_lat;
    Vec3::new(x as f32, y as f32, z as f32)
}

impl Default for BoundingVolume {
    fn default() -> Self {
        Self::Sphere(BoundingSphere {
            sphere: [0.0, 0.0, 0.0, 1.0],
        })
    }
}
