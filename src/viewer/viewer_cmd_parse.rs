// src/viewer/viewer_cmd_parse.rs
// Command string parsing for the interactive viewer
// RELEVANT FILES: src/viewer/mod.rs

use super::viewer_enums::{parse_gi_viz_mode_token, ViewerCmd, VizMode};

/// Parse a single command string into a ViewerCmd.
/// Returns None if the command is not recognized or requires special handling.
/// Returns Some(vec![cmds]) for commands that may expand to multiple commands.
pub fn parse_command_string(line: &str) -> Option<Vec<ViewerCmd>> {
    let l = line.trim().to_lowercase();
    if l.is_empty() {
        return None;
    }

    // GI seed
    if l.starts_with(":gi-seed") || l.starts_with("gi-seed ") {
        let mut it = l.split_whitespace();
        let _ = it.next();
        if let Some(val) = it.next().and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetGiSeed(val)]);
        }
        return Some(vec![ViewerCmd::QueryGiSeed]);
    }

    // GI toggle (can expand to multiple commands)
    if l.starts_with(":gi") || l.starts_with("gi ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() == 2 && toks[1] == "status" {
            return Some(vec![ViewerCmd::GiStatus]);
        }
        if toks.len() == 2 && toks[1] == "off" {
            return Some(vec![
                ViewerCmd::GiToggle("ssao", false),
                ViewerCmd::GiToggle("ssgi", false),
                ViewerCmd::GiToggle("ssr", false),
            ]);
        }
        if toks.len() >= 3 {
            let eff = match toks[1] {
                "ssao" | "ssgi" | "ssr" | "gtao" => toks[1],
                _ => return None,
            };
            let on = matches!(toks[2], "on" | "1" | "true");
            if eff == "gtao" {
                let mut cmds = vec![ViewerCmd::GiToggle("ssao", on)];
                if on {
                    cmds.push(ViewerCmd::SetSsaoTechnique(1));
                }
                return Some(cmds);
            }
            return Some(vec![ViewerCmd::GiToggle(
                match eff {
                    "ssao" => "ssao",
                    "ssgi" => "ssgi",
                    "ssr" => "ssr",
                    _ => "ssao",
                },
                on,
            )]);
        }
    }

    // Snapshot
    if l.starts_with(":snapshot") || l.starts_with("snapshot ") {
        let path = l.split_whitespace().nth(1).map(|s| s.to_string());
        return Some(vec![ViewerCmd::Snapshot(path)]);
    }

    // SSAO parameters
    if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoRadius(val)]);
        }
    }
    if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoIntensity(val)]);
        }
    }
    if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoBias(val)]);
        }
    }
    if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoSamples(val)]);
        }
    }
    if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoDirections(val)]);
        }
    }
    if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsaoTemporalAlpha(val)]);
        }
    }
    if l.starts_with(":ssao-temporal ") || l.starts_with("ssao-temporal ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let on = matches!(tok, "on" | "1" | "true");
            let off = matches!(tok, "off" | "0" | "false");
            if on || off {
                return Some(vec![ViewerCmd::SetSsaoTemporalEnabled(on)]);
            }
        }
    }
    if l.starts_with(":ssao-blur") || l.starts_with("ssao-blur ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetAoBlur(matches!(tok, "on" | "1" | "true"))]);
        }
    }
    if l.starts_with(":ssao-technique") || l.starts_with("ssao-technique ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let tech = match tok {
                "ssao" | "0" => 0,
                "gtao" | "1" => 1,
                _ => return None,
            };
            return Some(vec![ViewerCmd::SetSsaoTechnique(tech)]);
        }
    }

    // AO parameters
    if l.starts_with(":ao-temporal-alpha") || l.starts_with("ao-temporal-alpha ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetAoTemporalAlpha(val)]);
        }
    }
    if l.starts_with(":ao-blur") || l.starts_with("ao-blur ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetAoBlur(matches!(tok, "on" | "1" | "true"))]);
        }
    }

    // SSGI parameters
    if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetSsgiSteps(val)]);
        }
    }
    if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsgiRadius(val)]);
        }
    }
    if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetSsgiHalf(matches!(tok, "on" | "1" | "true"))]);
        }
    }
    if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsgiTemporalAlpha(val)]);
        }
    }
    if l.starts_with(":ssgi-temporal ") || l.starts_with("ssgi-temporal ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetSsgiTemporalEnabled(matches!(
                tok,
                "on" | "1" | "true"
            ))]);
        }
    }
    if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetSsgiEdges(matches!(tok, "on" | "1" | "true"))]);
        }
    }
    if l.starts_with(":ssgi-upsigma")
        || l.starts_with("ssgi-upsigma ")
        || l.starts_with(":ssgi-upsample-sigma-depth")
        || l.starts_with("ssgi-upsample-sigma-depth ")
    {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsgiUpsampleSigmaDepth(val)]);
        }
    }
    if l.starts_with(":ssgi-upsample-sigma-normal") || l.starts_with("ssgi-upsample-sigma-normal ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsgiUpsampleSigmaNormal(val)]);
        }
    }

    // SSR parameters
    if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetSsrMaxSteps(val)]);
        }
    }
    if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetSsrThickness(val)]);
        }
    }
    // SSR roughness threshold not currently exposed in ViewerCmd

    // Visualization
    if l.starts_with(":viz") || l.starts_with("viz ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let _mode = match tok {
                "material" => VizMode::Material,
                "normal" => VizMode::Normal,
                "depth" => VizMode::Depth,
                "gi" => VizMode::Gi,
                "lit" => VizMode::Lit,
                _ => return None,
            };
            return Some(vec![ViewerCmd::SetViz(tok.to_string())]);
        }
    }
    if l.starts_with(":viz-depth-max") || l.starts_with("viz-depth-max ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetVizDepthMax(val)]);
        }
    }
    if l.starts_with(":gi-viz") || l.starts_with("gi-viz ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            if let Some(mode) = parse_gi_viz_mode_token(tok) {
                return Some(vec![ViewerCmd::SetGiViz(mode)]);
            }
        }
    }

    // P5 captures
    if l.starts_with(":p5") || l.starts_with("p5 ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let cmd = match tok {
                "cornell" => ViewerCmd::CaptureP51Cornell,
                "grid" => ViewerCmd::CaptureP51Grid,
                "sweep" => ViewerCmd::CaptureP51Sweep,
                "ssgi-cornell" => ViewerCmd::CaptureP52SsgiCornell,
                "ssgi-temporal" => ViewerCmd::CaptureP52SsgiTemporal,
                "ssr-glossy" => ViewerCmd::CaptureP53SsrGlossy,
                "ssr-thickness" => ViewerCmd::CaptureP53SsrThickness,
                "gi-stack" => ViewerCmd::CaptureP54GiStack,
                _ => return None,
            };
            return Some(vec![cmd]);
        }
    }

    // HUD toggle
    if l.starts_with(":hud") || l.starts_with("hud ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::HudToggle(matches!(tok, "on" | "1" | "true"))]);
        }
    }

    // Quit
    if l == ":quit" || l == "quit" || l == ":exit" || l == "exit" {
        return Some(vec![ViewerCmd::Quit]);
    }

    None
}
