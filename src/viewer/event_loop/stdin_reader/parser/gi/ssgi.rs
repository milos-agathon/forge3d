use crate::viewer::event_loop::stdin_reader::helpers::{
    parse_bool_or_query, parse_float_or_query, parse_u32_or_query,
};
use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_ssgi_commands(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":ssgi-steps") || line.starts_with("ssgi-steps ") {
        return parse_u32_or_query(
            line,
            ViewerCmd::SetSsgiSteps,
            || ViewerCmd::QuerySsgiSteps,
            "ssgi-steps <u32>",
        );
    }
    if line.starts_with(":ssgi-radius") || line.starts_with("ssgi-radius ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsgiRadius,
            || ViewerCmd::QuerySsgiRadius,
            "ssgi-radius <float>",
        );
    }
    if line.starts_with(":ssgi-half") || line.starts_with("ssgi-half ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetSsgiHalf,
            || ViewerCmd::QuerySsgiHalf,
            "ssgi-half <on|off>",
        );
    }
    if line.starts_with(":ssgi-temporal-alpha") || line.starts_with("ssgi-temporal-alpha ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsgiTemporalAlpha,
            || ViewerCmd::QuerySsgiTemporalAlpha,
            "ssgi-temporal-alpha <0..1>",
        );
    }
    if line.starts_with(":ssgi-temporal ") || line.starts_with("ssgi-temporal ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetSsgiTemporalEnabled,
            || ViewerCmd::QuerySsgiTemporalEnabled,
            "ssgi-temporal <on|off>",
        );
    }
    if line.starts_with(":ssgi-edges") || line.starts_with("ssgi-edges ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetSsgiEdges,
            || ViewerCmd::QuerySsgiEdges,
            "ssgi-edges <on|off>",
        );
    }
    if line.starts_with(":ssgi-upsigma")
        || line.starts_with("ssgi-upsigma ")
        || line.starts_with(":ssgi-upsample-sigma-depth")
        || line.starts_with("ssgi-upsample-sigma-depth ")
    {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsgiUpsampleSigmaDepth,
            || ViewerCmd::QuerySsgiUpsampleSigmaDepth,
            "ssgi-upsample-sigma-depth <float>",
        );
    }
    if line.starts_with(":ssgi-normexp")
        || line.starts_with("ssgi-normexp ")
        || line.starts_with(":ssgi-upsample-sigma-normal")
        || line.starts_with("ssgi-upsample-sigma-normal ")
    {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsgiUpsampleSigmaNormal,
            || ViewerCmd::QuerySsgiUpsampleSigmaNormal,
            "ssgi-upsample-sigma-normal <float>",
        );
    }
    None
}
