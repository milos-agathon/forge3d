// src/viewer/cmd/handler.rs
// Thin command dispatcher for the interactive viewer.

use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn handle_cmd(&mut self, cmd: ViewerCmd) {
        if super::gi_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::scene_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::effects_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::terrain_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::vector_overlay_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::labels_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::ipc_command::handle_cmd(self, &cmd) {
            return;
        }
        if super::pointcloud_command::handle_cmd(self, &cmd) {
            return;
        }

        self.handle_cmd_legacy(cmd);
    }
}
