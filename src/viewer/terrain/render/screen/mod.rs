mod effects;
mod resources;
mod scene;
mod setup;

use super::*;

pub(super) struct ScreenRenderFlags {
    pub(super) use_pbr: bool,
    pub(super) needs_taa: bool,
    pub(super) needs_dof: bool,
    pub(super) needs_post_process: bool,
    pub(super) needs_volumetrics: bool,
    pub(super) needs_denoise: bool,
}

pub(super) struct ScreenRenderState {
    pub(super) view_mat: glam::Mat4,
    pub(super) proj: glam::Mat4,
    pub(super) view_proj: glam::Mat4,
    pub(super) view_proj_array: [[f32; 4]; 4],
    pub(super) sun_dir: glam::Vec3,
    pub(super) eye: glam::Vec3,
    pub(super) render_origin_span: [f32; 4],
    pub(super) h_range: f32,
    pub(super) shader_z_scale: f32,
    pub(super) cam_radius: f32,
    pub(super) vo_sun_dir: [f32; 3],
    pub(super) vo_lighting: [f32; 4],
}

impl ViewerTerrainScene {
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        selected_feature_id: u32,
        frame: crate::viewer::viewer_types::FrameCamera,
    ) -> bool {
        if self.terrain.is_none() {
            return false;
        }

        let flags = match self.prepare_screen_resources(width, height) {
            Ok(flags) => flags,
            Err(err) => {
                eprintln!("[terrain] screen resource preparation failed: {err:#}");
                return false;
            }
        };
        let state = self.build_screen_render_state(encoder, width, height, &flags, frame);

        let has_vector_overlays = self.prepare_screen_overlays();
        self.render_screen_scene_path(
            encoder,
            view,
            selected_feature_id,
            &flags,
            &state,
            has_vector_overlays,
        );
        self.apply_screen_effects(encoder, view, width, height, &flags, &state);

        true
    }
}
