//! 3D math for the canonical model.
//!
//! `Vec3` and `Transform` are the *wire* types — plain data with the camelCase
//! serde shape the schema and the TypeScript bridge depend on — and they stay
//! that way. The arithmetic behind them is `glam`, the same crate the engine
//! uses, so a rotation is one quaternion instead of three hand-rolled matrices.
//!
//! Rotation is Euler degrees in X, Y, Z order, applied as `Rz * Ry * Rx` to a
//! camera that looks down -Z. Both the TypeScript preview and the Three.js
//! viewport must use the same convention; `scene::tests` pins the camera
//! default against it.

use glam::{EulerRot, Quat};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    pub const ONE: Self = Self {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn is_finite(self) -> bool {
        self.geometry().is_finite()
    }

    /// The model vector as geometry, for the `glam` computations below.
    pub(crate) fn geometry(self) -> glam::Vec3 {
        glam::Vec3::new(self.x, self.y, self.z)
    }

    pub(crate) fn from_geometry(vector: glam::Vec3) -> Self {
        Self::new(vector.x, vector.y, vector.z)
    }
}

impl Default for Vec3 {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Local transform of a scene object (parent-relative).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Transform {
    pub translation: Vec3,
    pub rotation_deg: Vec3,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation_deg: Vec3::ZERO,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// The rotation as a quaternion, in the model's `Rz * Ry * Rx` order.
    pub fn rotation(&self) -> Quat {
        Quat::from_euler(
            EulerRot::ZYX,
            self.rotation_deg.z.to_radians(),
            self.rotation_deg.y.to_radians(),
            self.rotation_deg.x.to_radians(),
        )
    }

    /// Row-major 4x4 (T * R * S) for handing a placement to a renderer that
    /// consumes row-major instance matrices. Glam is column-major; transposing
    /// once here means every consumer sees one convention.
    pub fn row_major_matrix(&self) -> [f32; 16] {
        glam::Mat4::from_scale_rotation_translation(
            self.scale.geometry(),
            self.rotation(),
            self.translation.geometry(),
        )
        .transpose()
        .to_cols_array()
    }

    /// Local forward axis (`-Z`) in parent space.
    pub fn forward(&self) -> Vec3 {
        Vec3::from_geometry(self.rotation() * glam::Vec3::NEG_Z)
    }

    /// Local up axis (`+Y`) in parent space.
    pub fn up(&self) -> Vec3 {
        Vec3::from_geometry(self.rotation() * glam::Vec3::Y)
    }

    /// Euler rotation (degrees, the model's ZYX order) that points the local
    /// `-Z` axis along `forward`.
    ///
    /// Roll is zero: a light or camera only needs its direction, and a
    /// direction has two degrees of freedom. A vertical `forward` (straight up
    /// or down) is a degenerate yaw, so it gets pitch 0 or 180 with yaw 0.
    pub fn rotation_looking_along(forward: Vec3) -> Vec3 {
        let length = (forward.x * forward.x + forward.y * forward.y + forward.z * forward.z).sqrt();
        if !length.is_finite() || length <= 1e-9 {
            return Vec3::ZERO;
        }
        let (x, y, z) = (forward.x / length, forward.y / length, forward.z / length);
        let pitch = (-z).clamp(-1.0, 1.0).acos().to_degrees();
        let horizontal = (x * x + y * y).sqrt();
        if horizontal < 1e-6 {
            return Vec3::new(0.0, pitch, 0.0);
        }
        Vec3::new(0.0, pitch, (-y).atan2(-x).to_degrees())
    }

    /// Reject NaN/infinite values before they reach a renderer, where they
    /// become invisible objects and un-diagnosable blank frames.
    pub fn validate(&self) -> Result<(), String> {
        if !self.translation.is_finite() {
            return Err("translation must be finite".to_owned());
        }
        if !self.rotation_deg.is_finite() {
            return Err("rotation must be finite".to_owned());
        }
        if !self.scale.is_finite() {
            return Err("scale must be finite".to_owned());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_transform_is_identity() {
        let t = Transform::default();
        assert_eq!(t.translation, Vec3::ZERO);
        assert_eq!(t.rotation_deg, Vec3::ZERO);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn euler_rotation_matches_the_documented_order() {
        let assert_close = |actual: Vec3, expected: Vec3| {
            let close = (actual.x - expected.x).abs() < 1e-5
                && (actual.y - expected.y).abs() < 1e-5
                && (actual.z - expected.z).abs() < 1e-5;
            assert!(close, "expected {expected:?}, got {actual:?}");
        };

        // Yaw only: -Z turns to -X.
        let yaw = Transform {
            rotation_deg: Vec3::new(0.0, 90.0, 0.0),
            ..Transform::default()
        };
        assert_close(yaw.forward(), Vec3::new(-1.0, 0.0, 0.0));

        // Pitch only: -Z turns to -Y (looking down).
        let pitch = Transform {
            rotation_deg: Vec3::new(-90.0, 0.0, 0.0),
            ..Transform::default()
        };
        assert_close(pitch.forward(), Vec3::new(0.0, -1.0, 0.0));
        assert_close(pitch.up(), Vec3::new(0.0, 0.0, -1.0));

        // Pitch then yaw, the starter sun's rotation: the Y output must not be
        // contaminated by the yaw step (regression: a shadowed `x` did that).
        let sun = Transform {
            rotation_deg: Vec3::new(-50.0, 30.0, 0.0),
            ..Transform::default()
        };
        let forward = sun.forward();
        assert_close(
            forward,
            Vec3::new(
                -0.5 * 50f32.to_radians().cos(),
                -50f32.to_radians().sin(),
                -0.866_025_4 * 50f32.to_radians().cos(),
            ),
        );
    }

    #[test]
    fn non_finite_transforms_are_rejected() {
        let mut t = Transform::default();
        t.translation.x = f32::NAN;
        assert!(t.validate().is_err());
        t.translation.x = 0.0;
        t.scale.y = f32::INFINITY;
        assert!(t.validate().is_err());
    }

    #[test]
    fn looking_along_rotation_recovers_its_forward_axis() {
        let assert_forward = |forward: Vec3| {
            let rotation = Transform::rotation_looking_along(forward);
            let look = Transform {
                rotation_deg: rotation,
                ..Transform::default()
            };
            let got = look.forward();
            let length =
                (forward.x * forward.x + forward.y * forward.y + forward.z * forward.z).sqrt();
            let want = Vec3::new(forward.x / length, forward.y / length, forward.z / length);
            let close = (got.x - want.x).abs() < 1e-5
                && (got.y - want.y).abs() < 1e-5
                && (got.z - want.z).abs() < 1e-5;
            assert!(
                close,
                "forward {want:?}, recovered {got:?} via {rotation:?}"
            );
        };

        for forward in [
            Vec3::new(0.0, 0.0, -1.0),   // the model's neutral forward
            Vec3::new(0.0, 0.0, 1.0),    // straight up: degenerate yaw
            Vec3::new(-1.0, 0.0, 0.0),   // west
            Vec3::new(0.0, -1.0, 0.0),   // down
            Vec3::new(0.0, 1.0, 0.0),    // up
            Vec3::new(0.3, -0.4, 0.866), // arbitrary
            Vec3::new(2.0, 0.0, 0.0),    // non-unit input
        ] {
            assert_forward(forward);
        }

        // A zero direction cannot define a rotation; refuse silently.
        assert_eq!(Transform::rotation_looking_along(Vec3::ZERO), Vec3::ZERO);
    }
}
