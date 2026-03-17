use super::parse::{parse_f32, parse_u32};
use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_atmosphere_command(line: &str) -> Option<Vec<ViewerCmd>> {
    parse_sky_command(line).or_else(|| parse_fog_command(line))
}

fn parse_sky_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":sky ") || line == ":sky" || line.starts_with("sky ") {
        if let Some(arg) = line.split_whitespace().nth(1) {
            let cmds = match arg {
                "off" | "0" | "false" => vec![ViewerCmd::SkyToggle(false)],
                "on" | "1" | "true" => vec![ViewerCmd::SkyToggle(true)],
                "preetham" => vec![ViewerCmd::SkyToggle(true), ViewerCmd::SkySetModel(0)],
                "hosek-wilkie" | "hosekwilkie" | "hosek" | "hw" => {
                    vec![ViewerCmd::SkyToggle(true), ViewerCmd::SkySetModel(1)]
                }
                _ => {
                    println!(
                        "Unknown sky mode '{}', expected off|on|preetham|hosek-wilkie",
                        arg
                    );
                    vec![]
                }
            };
            return Some(cmds);
        }
        println!("Usage: :sky <off|on|preetham|hosek-wilkie>");
        return Some(vec![]);
    }
    if line.starts_with(":sky-turbidity") || line.starts_with("sky-turbidity ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::SkySetTurbidity)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":sky-ground") || line.starts_with("sky-ground ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::SkySetGround)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":sky-exposure") || line.starts_with("sky-exposure ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::SkySetExposure)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":sky-sun") || line.starts_with("sky-sun ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::SkySetSunIntensity)
                .into_iter()
                .collect(),
        );
    }
    None
}

fn parse_fog_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":fog ") || line == ":fog" || line.starts_with("fog ") {
        if let Some(arg) = line.split_whitespace().nth(1) {
            let on = matches!(arg, "on" | "1" | "true");
            return Some(vec![ViewerCmd::FogToggle(on)]);
        }
        println!("Usage: :fog <on|off>");
        return Some(vec![]);
    }
    if line.starts_with(":fog-density") || line.starts_with("fog-density ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::FogSetDensity)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-g") || line.starts_with("fog-g ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::FogSetG)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-steps") || line.starts_with("fog-steps ") {
        return Some(
            parse_u32(line)
                .map(ViewerCmd::FogSetSteps)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-shadow") || line.starts_with("fog-shadow ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| ViewerCmd::FogSetShadow(matches!(tok, "on" | "1" | "true")))
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-temporal") || line.starts_with("fog-temporal ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::FogSetTemporal)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-mode") || line.starts_with("fog-mode ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| {
                    let idx = match tok {
                        "raymarch" | "rm" | "0" => 0u32,
                        "froxels" | "fx" | "1" => 1u32,
                        _ => 0u32,
                    };
                    ViewerCmd::SetFogMode(idx)
                })
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-preset") || line.starts_with("fog-preset ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| {
                    let idx = match tok {
                        "low" | "0" => 0u32,
                        "med" | "medium" | "1" => 1u32,
                        _ => 2u32,
                    };
                    ViewerCmd::FogPreset(idx)
                })
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-half") || line.starts_with("fog-half ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| ViewerCmd::FogHalf(matches!(tok, "on" | "1" | "true")))
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-edges") || line.starts_with("fog-edges ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| ViewerCmd::FogEdges(matches!(tok, "on" | "1" | "true")))
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":fog-upsigma") || line.starts_with("fog-upsigma ") {
        return Some(
            parse_f32(line)
                .map(ViewerCmd::FogUpsigma)
                .into_iter()
                .collect(),
        );
    }
    None
}
