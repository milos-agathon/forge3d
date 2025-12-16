// src/viewer/state/scene_state.rs
// Scene/object parameters for the Viewer

/// Scene/object parameters (runtime adjustable)
#[derive(Clone, Debug)]
pub struct SceneParams {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
    pub transform: glam::Mat4,
    pub transform_version: u64,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self {
            translation: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            transform: glam::Mat4::IDENTITY,
            transform_version: 0,
        }
    }
}

impl SceneParams {
    /// Update transform from components
    pub fn update_transform(&mut self) {
        self.transform = glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation,
            self.translation,
        );
    }

    /// Set translation and update transform
    pub fn set_translation(&mut self, t: glam::Vec3) {
        self.translation = t;
        self.update_transform();
        self.transform_version += 1;
    }

    /// Set rotation and update transform
    pub fn set_rotation(&mut self, r: glam::Quat) {
        self.rotation = r;
        self.update_transform();
        self.transform_version += 1;
    }

    /// Set scale and update transform
    pub fn set_scale(&mut self, s: glam::Vec3) {
        self.scale = s;
        self.update_transform();
        self.transform_version += 1;
    }

    /// Reset to identity
    pub fn reset(&mut self) {
        self.translation = glam::Vec3::ZERO;
        self.rotation = glam::Quat::IDENTITY;
        self.scale = glam::Vec3::ONE;
        self.transform = glam::Mat4::IDENTITY;
    }

    /// Check if transform is identity
    pub fn is_identity(&self) -> bool {
        self.translation == glam::Vec3::ZERO
            && self.rotation == glam::Quat::IDENTITY
            && self.scale == glam::Vec3::ONE
    }
}
