use crate::core::error::RenderResult;
use crate::viewer::render_targets::PreparedViewportTargets;
use crate::viewer::Viewer;
use winit::dpi::PhysicalSize;

impl Viewer {
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.failed_window_resize = None;
        if !self.render_targets_match(new_size.width, new_size.height) {
            if let Err(error) = self.resize_render_targets(new_size.width, new_size.height) {
                self.failed_window_resize = Some(new_size);
                eprintln!("Failed to resize render targets: {error}");
                return;
            }
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }

    /// Surface loss never retries a failed allocation or resets temporal history.
    pub(crate) fn recover_surface(&mut self) -> bool {
        if !self.window_can_render() {
            return false;
        }
        self.surface.configure(&self.device, &self.config);
        true
    }

    pub(crate) fn window_can_render(&self) -> bool {
        let size = self.window.inner_size();
        self.failed_window_resize != Some(size)
            && size.width != 0
            && size.height != 0
            && size.width == self.config.width
            && size.height == self.config.height
            && self.render_targets_match(size.width, size.height)
    }

    /// Prepare every size-dependent resource before replacing the live set.
    /// The surface size is separate so offscreen callers keep their window size.
    pub fn resize_render_targets(&mut self, width: u32, height: u32) -> RenderResult<()> {
        PreparedViewportTargets::prepare(self, width, height)?.install(self);
        Ok(())
    }
}
