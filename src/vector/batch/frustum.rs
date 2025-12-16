//! View frustum for visibility culling.

use super::aabb::AABB;
use glam::{Mat4, Vec3, Vec4};

/// 2D view frustum defined by four planes (left, right, bottom, top).
#[derive(Debug, Clone)]
pub struct Frustum {
    /// Frustum planes: [left, right, bottom, top].
    pub planes: [Vec4; 4],
}

impl Frustum {
    /// Create frustum from view-projection matrix.
    ///
    /// Extracts the four side planes from the combined view-projection matrix.
    pub fn from_view_proj_matrix(vp_matrix: &Mat4) -> Self {
        let m = vp_matrix.transpose();

        let left = (m.w_axis + m.x_axis).normalize();
        let right = (m.w_axis - m.x_axis).normalize();
        let bottom = (m.w_axis + m.y_axis).normalize();
        let top = (m.w_axis - m.y_axis).normalize();

        Self {
            planes: [left, right, bottom, top],
        }
    }

    /// Alias for `from_view_proj_matrix`.
    pub fn from_view_proj(vp_matrix: &Mat4) -> Self {
        Self::from_view_proj_matrix(vp_matrix)
    }

    /// Test if an AABB is at least partially inside the frustum.
    ///
    /// Returns `true` if any corner of the AABB is inside all frustum planes.
    pub fn test_aabb(&self, aabb: &AABB) -> bool {
        let corners = [
            Vec3::new(aabb.min.x, aabb.min.y, 0.0),
            Vec3::new(aabb.max.x, aabb.min.y, 0.0),
            Vec3::new(aabb.min.x, aabb.max.y, 0.0),
            Vec3::new(aabb.max.x, aabb.max.y, 0.0),
        ];

        for plane in &self.planes {
            let mut inside = false;
            for corner in &corners {
                let distance =
                    plane.x * corner.x + plane.y * corner.y + plane.z * corner.z + plane.w;
                if distance >= 0.0 {
                    inside = true;
                    break;
                }
            }
            if !inside {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frustum_culling() {
        let identity = Mat4::IDENTITY;
        let frustum = Frustum::from_view_proj_matrix(&identity);

        let aabb = AABB::new(glam::Vec2::new(-0.5, -0.5), glam::Vec2::new(0.5, 0.5));
        assert!(frustum.test_aabb(&aabb));
    }
}
