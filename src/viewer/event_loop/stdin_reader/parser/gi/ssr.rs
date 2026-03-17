use crate::viewer::event_loop::stdin_reader::helpers::{parse_float_or_query, parse_u32_or_query};
use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_ssr_commands(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":ssr-max-steps") || line.starts_with("ssr-max-steps ") {
        return parse_u32_or_query(
            line,
            ViewerCmd::SetSsrMaxSteps,
            || ViewerCmd::QuerySsrMaxSteps,
            "ssr-max-steps <u32>",
        );
    }
    if line.starts_with(":ssr-thickness") || line.starts_with("ssr-thickness ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsrThickness,
            || ViewerCmd::QuerySsrThickness,
            "ssr-thickness <float>",
        );
    }
    if line == ":load-ssr-preset" || line == "load-ssr-preset" {
        return Some(vec![ViewerCmd::LoadSsrPreset]);
    }
    None
}
