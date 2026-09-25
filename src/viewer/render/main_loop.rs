// src/viewer/render/main_loop.rs
// Main render loop for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

mod finalize;
pub(crate) mod frame_anchor;
mod frame_anchor_stats;
mod frame_setup;
mod geometry;
mod postfx;
mod postfx_cache;
mod secondary;

use crate::viewer::Viewer;

#[derive(Clone, Copy)]
pub(super) struct RenderAvailability {
    pub(super) have_gi: bool,
    pub(super) have_pipe: bool,
    pub(super) have_cam: bool,
    pub(super) have_vb: bool,
    pub(super) have_z: bool,
    pub(super) have_bgl: bool,
}

impl RenderAvailability {
    pub(super) fn ready(self) -> bool {
        self.have_gi
            && self.have_pipe
            && self.have_cam
            && self.have_vb
            && self.have_z
            && self.have_bgl
    }
}

impl Viewer {
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.render_frame(None)
    }

    /// Shared frame pipeline for windowed and headless viewers.
    ///
    /// `timing` (CENSOR F-04): when supplied as `(one_shot, label)`, a loaded
    /// PBR scene brackets its GPU work in a certificate timing scope on the
    /// frame encoder (see `Viewer::render_pbr_scene_stage`).
    pub(crate) fn render_frame(
        &mut self,
        timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    ) -> Result<(), wgpu::SurfaceError> {
        self.prepare_frame_anchor();
        let (output, view, snapshot_dimensions, mut encoder) = self.prepare_render_frame()?;
        if self.pbr_scene.is_some() {
            if let Err(error) = self.render_pbr_scene_stage(&mut encoder, &view, timing) {
                eprintln!("[viewer] pbr_scene stage failed: {error}");
                self.command_error = Some(format!("pbr_scene_render_failed: {error}"));
            }
        } else {
            let availability = self.render_geometry_stage(&mut encoder, &view);
            encoder =
                self.render_secondary_paths(encoder, &view, snapshot_dimensions, availability);
        }
        self.finish_render_frame(encoder, output);
        Ok(())
    }

    /// Render one frame of a headless viewer (created via
    /// [`Viewer::new_headless`]). Errors when called on a windowed viewer.
    pub fn render_headless_frame(
        &mut self,
        timing: Option<(&mut crate::core::gpu_timing::OneShotTiming, &str)>,
    ) -> crate::core::error::RenderResult<()> {
        if self.window().is_some() {
            return Err(crate::core::error::RenderError::Render(
                "render_headless_frame requires a headless viewer".into(),
            ));
        }
        self.render_frame(timing).map_err(|e| {
            crate::core::error::RenderError::Render(format!("headless frame failed: {e}"))
        })?;
        if self.pbr_scene.is_some() {
            if let Some(error) = self.command_error.take() {
                return Err(crate::core::error::RenderError::Render(error));
            }
        }
        Ok(())
    }
}
