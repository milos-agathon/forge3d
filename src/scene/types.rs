#[derive(Debug, Clone)]
pub struct SceneGlobals {
    pub globals: crate::terrain::Globals,
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
}

impl Default for SceneGlobals {
    fn default() -> Self {
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(3.0, 2.0, 3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );
        let proj = crate::camera::perspective_wgpu(45f32.to_radians(), 4.0 / 3.0, 0.1, 100.0);
        Self {
            globals: crate::terrain::Globals::default(),
            view,
            proj,
        }
    }
}

pub(super) struct Text3DInstance {
    pub(super) vbuf: crate::core::resource_tracker::TrackedBuffer,
    pub(super) ibuf: crate::core::resource_tracker::TrackedBuffer,
    pub(super) index_count: u32,
    pub(super) vertex_count: u32,
    /// Absolute object origin. It remains f64 until the active camera anchor
    /// prepares the render model translation.
    pub(super) origin: glam::DVec3,
    /// Rotation and scale only; translation is derived from `origin`.
    pub(super) local_model: glam::Mat4,
    pub(super) model: glam::Mat4,
    pub(super) color: [f32; 4],
    pub(super) light_dir: [f32; 3],
    pub(super) light_intensity: f32,
    pub(super) metallic: f32,
    pub(super) roughness: f32,
}

pub(super) fn anchored_model(
    anchor: &crate::camera::Anchor,
    origin: glam::DVec3,
    local_model: glam::Mat4,
) -> glam::Mat4 {
    glam::Mat4::from_translation(anchor.model_offset(origin)) * local_model
}

// F16: GPU Instancing batch description
#[cfg(feature = "enable-gpu-instancing")]
pub(super) struct InstancedBatch {
    pub(super) vbuf: crate::core::resource_tracker::TrackedBuffer,
    pub(super) ibuf: crate::core::resource_tracker::TrackedBuffer,
    pub(super) instbuf: crate::core::resource_tracker::TrackedBuffer,
    pub(super) index_count: u32,
    pub(super) instance_count: u32,
    pub(super) color: [f32; 4],
    pub(super) light_dir: [f32; 3],
    pub(super) light_intensity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_object_origin_is_anchor_relative_at_earth_radius() {
        let origin = glam::DVec3::new(6_378_137.000_25, 2.0, -3.0);
        let mut anchor = crate::camera::Anchor::new();
        anchor.rebase_if_needed(glam::DVec3::new(6_378_137.0, 0.0, 0.0));
        let model = anchored_model(&anchor, origin, glam::Mat4::IDENTITY);
        assert!((model.w_axis.x - 0.000_25).abs() < 1e-6);
        assert_eq!(model.w_axis.y, 2.0);
        assert_eq!(model.w_axis.z, -3.0);
    }
}
