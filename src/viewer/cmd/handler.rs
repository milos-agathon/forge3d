// src/viewer/cmd/handler.rs
// Thin command dispatcher for the interactive viewer.

use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn reject_command(&mut self, error: impl Into<String>) {
        if self.command_error.is_none() {
            self.command_error = Some(error.into());
        }
    }

    fn command_outcome(&mut self) -> Result<(), String> {
        self.command_error.take().map_or(Ok(()), Err)
    }

    fn finish_command(&mut self) -> Result<(), String> {
        self.command_outcome()?;
        self.applied_command_revision = self.applied_command_revision.wrapping_add(1);
        crate::viewer::event_loop::update_ipc_revision_stats(
            self.applied_command_revision,
            self.rendered_frame_revision,
        );
        Ok(())
    }

    pub(crate) fn handle_cmd(&mut self, cmd: ViewerCmd) -> Result<(), String> {
        self.command_error = None;
        if super::gi_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::scene_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::effects_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::terrain_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::vector_overlay_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::labels_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::scene_review_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::ipc_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }
        if super::pointcloud_command::handle_cmd(self, &cmd) {
            return self.finish_command();
        }

        self.handle_cmd_legacy(cmd);
        self.finish_command()
    }
}
