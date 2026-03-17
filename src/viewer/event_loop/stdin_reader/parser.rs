mod environment;
mod gi;
mod render;

use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_stdin_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if let Some(cmds) = gi::parse_gi_command(line) {
        return Some(cmds);
    }
    if let Some(cmds) = render::parse_render_command(line) {
        return Some(cmds);
    }
    if let Some(cmds) = environment::parse_environment_command(line) {
        return Some(cmds);
    }
    None
}
