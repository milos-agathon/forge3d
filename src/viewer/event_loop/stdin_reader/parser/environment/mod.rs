mod atmosphere;
mod ibl;
mod lighting;
mod misc;
mod parse;

use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_environment_command(line: &str) -> Option<Vec<ViewerCmd>> {
    lighting::parse_brdf_and_lighting(line)
        .or_else(|| ibl::parse_ibl_command(line))
        .or_else(|| atmosphere::parse_atmosphere_command(line))
        .or_else(|| misc::parse_misc_command(line))
}
