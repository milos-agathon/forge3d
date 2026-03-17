use crate::viewer::event_loop::stdin_reader::helpers::{
    parse_bool_or_query, parse_float_or_query, parse_u32_or_query,
};
use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_ssao_commands(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":ssao-radius") || line.starts_with("ssao-radius ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsaoRadius,
            || ViewerCmd::QuerySsaoRadius,
            "ssao-radius <float>",
        );
    }
    if line.starts_with(":ssao-intensity") || line.starts_with("ssao-intensity ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsaoIntensity,
            || ViewerCmd::QuerySsaoIntensity,
            "ssao-intensity <float>",
        );
    }
    if line.starts_with(":ssao-bias") || line.starts_with("ssao-bias ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsaoBias,
            || ViewerCmd::QuerySsaoBias,
            "ssao-bias <float>",
        );
    }
    if line.starts_with(":ssao-samples") || line.starts_with("ssao-samples ") {
        return parse_u32_or_query(
            line,
            ViewerCmd::SetSsaoSamples,
            || ViewerCmd::QuerySsaoSamples,
            "ssao-samples <u32>",
        );
    }
    if line.starts_with(":ssao-directions") || line.starts_with("ssao-directions ") {
        return parse_u32_or_query(
            line,
            ViewerCmd::SetSsaoDirections,
            || ViewerCmd::QuerySsaoDirections,
            "ssao-directions <u32>",
        );
    }
    if line.starts_with(":ssao-temporal-alpha") || line.starts_with("ssao-temporal-alpha ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsaoTemporalAlpha,
            || ViewerCmd::QuerySsaoTemporalAlpha,
            "ssao-temporal-alpha <0..1>",
        );
    }
    if line.starts_with(":ssao-temporal ") || line.starts_with("ssao-temporal ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetSsaoTemporalEnabled,
            || ViewerCmd::QuerySsaoTemporalEnabled,
            "ssao-temporal <on|off>",
        );
    }
    if line.starts_with(":ssao-blur") || line.starts_with("ssao-blur ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetAoBlur,
            || ViewerCmd::QuerySsaoBlur,
            "ssao-blur <on|off>",
        );
    }
    if line.starts_with(":ssao-composite") || line.starts_with("ssao-composite ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetSsaoComposite,
            || ViewerCmd::QuerySsaoComposite,
            "ssao-composite <on|off>",
        );
    }
    if line.starts_with(":ssao-mul") || line.starts_with("ssao-mul ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetSsaoCompositeMul,
            || ViewerCmd::QuerySsaoMul,
            "ssao-mul <0..1>",
        );
    }
    if line.starts_with(":ao-temporal-alpha") || line.starts_with("ao-temporal-alpha ") {
        return parse_float_or_query(
            line,
            ViewerCmd::SetAoTemporalAlpha,
            || ViewerCmd::QuerySsaoTemporalAlpha,
            "ao-temporal-alpha <0..1>",
        );
    }
    if line.starts_with(":ao-blur") || line.starts_with("ao-blur ") {
        return parse_bool_or_query(
            line,
            ViewerCmd::SetAoBlur,
            || ViewerCmd::QuerySsaoBlur,
            "ao-blur <on|off>",
        );
    }
    if line.starts_with(":ssao-technique") || line.starts_with("ssao-technique ") {
        return Some(if let Some(tok) = line.split_whitespace().nth(1) {
            let tech = match tok {
                "ssao" | "0" => 0u32,
                "gtao" | "1" => 1u32,
                _ => 0u32,
            };
            vec![ViewerCmd::SetSsaoTechnique(tech)]
        } else {
            vec![ViewerCmd::QuerySsaoTechnique]
        });
    }
    None
}
