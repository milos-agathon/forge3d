// src/viewer/cmd/legacy_handler.rs
// Fallback command handler for Viewer commands that intentionally stay local.

use crate::viewer::viewer_enums::ViewerCmd;
use crate::viewer::Viewer;

impl Viewer {
    pub(crate) fn handle_cmd_legacy(&mut self, cmd: ViewerCmd) {
        match cmd {
            ViewerCmd::Quit => {}
            _ => {}
        }
    }
}
