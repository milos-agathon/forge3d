mod core;
mod ssao;
mod ssgi;
mod ssr;

use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_gi_command(line: &str) -> Option<Vec<ViewerCmd>> {
    core::parse_gi_core_command(line)
        .or_else(|| ssao::parse_ssao_commands(line))
        .or_else(|| ssgi::parse_ssgi_commands(line))
        .or_else(|| ssr::parse_ssr_commands(line))
}
