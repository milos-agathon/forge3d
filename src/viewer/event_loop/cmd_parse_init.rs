// src/viewer/event_loop/cmd_parse_init.rs
// Initial command parsing for viewer startup

use crate::viewer::viewer_enums::{parse_gi_viz_mode_token, ViewerCmd};

/// Parse initial commands from CLI into ViewerCmd list
/// This consolidates the duplicated command parsing logic from run_viewer
pub fn parse_initial_commands(cmds: &[String]) -> Vec<ViewerCmd> {
    let mut pending = Vec::new();

    for raw in cmds.iter() {
        let l = raw.trim().to_lowercase();
        if l.is_empty() {
            continue;
        }

        if let Some(cmd) = parse_single_initial_command(&l) {
            pending.extend(cmd);
        }
    }

    pending
}

fn parse_single_initial_command(l: &str) -> Option<Vec<ViewerCmd>> {
    // GI seed
    if l.starts_with(":gi-seed") || l.starts_with("gi-seed ") {
        let mut it = l.split_whitespace();
        let _ = it.next();
        if let Some(val) = it.next().and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::SetGiSeed(val)]);
        }
    }

    // GI toggle
    if l.starts_with(":gi") || l.starts_with("gi ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
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
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsaoRadius(val)]);
        }
    }
    if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsaoIntensity(val)]);
        }
    }
    if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsaoBias(val)]);
        }
    }
    if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsaoSamples(val)]);
        }
    }
    if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsaoDirections(val)]);
        }
    }
    if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
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
            return Some(vec![ViewerCmd::SetAoBlur(matches!(
                tok,
                "on" | "1" | "true"
            ))]);
        }
    }

    // SSGI parameters
    if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsgiSteps(val)]);
        }
    }
    if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsgiRadius(val)]);
        }
    }
    if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::SetSsgiHalf(matches!(
                tok,
                "on" | "1" | "true"
            ))]);
        }
    }

    // SSR parameters
    if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsrMaxSteps(val)]);
        }
    }
    if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetSsrThickness(val)]);
        }
    }

    // Load SSR preset
    if l == ":load-ssr-preset" || l == "load-ssr-preset" {
        return Some(vec![ViewerCmd::LoadSsrPreset]);
    }

    // P5 captures
    if l.starts_with(":p5") || l.starts_with("p5 ") {
        let sub = l.split_whitespace().nth(1).unwrap_or("");
        match sub {
            "cornell" => return Some(vec![ViewerCmd::CaptureP51Cornell]),
            "grid" => return Some(vec![ViewerCmd::CaptureP51Grid]),
            "sweep" => return Some(vec![ViewerCmd::CaptureP51Sweep]),
            "ssgi-cornell" => return Some(vec![ViewerCmd::CaptureP52SsgiCornell]),
            "ssgi-temporal" => return Some(vec![ViewerCmd::CaptureP52SsgiTemporal]),
            "ssr-glossy" => return Some(vec![ViewerCmd::CaptureP53SsrGlossy]),
            "ssr-thickness" => return Some(vec![ViewerCmd::CaptureP53SsrThickness]),
            "gi-stack" => return Some(vec![ViewerCmd::CaptureP54GiStack]),
            _ => {}
        }
    }

    // Load OBJ/glTF/Terrain
    if l.starts_with(":obj") || l.starts_with("obj ") {
        if let Some(path) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::LoadObj(path.to_string())]);
        }
    }
    if l.starts_with(":gltf") || l.starts_with("gltf ") {
        if let Some(path) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::LoadGltf(path.to_string())]);
        }
    }
    if l.starts_with(":terrain") || l.starts_with("terrain ") {
        if let Some(path) = l.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::LoadTerrain(path.to_string())]);
        }
    }

    // Visualization
    if l.starts_with(":viz") || l.starts_with("viz ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() >= 2 && toks[0] == "viz" && toks[1] == "gi" {
            if toks.len() == 2 {
                return Some(vec![ViewerCmd::QueryGiViz]);
            } else if let Some(m) = parse_gi_viz_mode_token(toks[2]) {
                return Some(vec![ViewerCmd::SetGiViz(m)]);
            }
        } else if toks.len() >= 2 {
            return Some(vec![ViewerCmd::SetViz(toks[1].to_string())]);
        }
    }

    // BRDF
    if l.starts_with(":brdf") || l.starts_with("brdf ") {
        if let Some(model) = l.split_whitespace().nth(1) {
            let idx = match model {
                "lambert" | "lam" => 0u32,
                "phong" => 1u32,
                "ggx" | "cooktorrance-ggx" | "cook-torrance-ggx" | "cooktorrance" | "ct-ggx" => {
                    4u32
                }
                "disney" | "disney-principled" | "principled" => 6u32,
                _ => 4u32,
            };
            return Some(vec![ViewerCmd::SetLitBrdf(idx)]);
        }
    }

    // Lit controls
    if l.starts_with(":lit-rough") || l.starts_with("lit-rough ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetLitRough(val)]);
        }
    }
    if l.starts_with(":lit-debug") || l.starts_with("lit-debug ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let mode = match tok {
                "rough" | "1" | "smoke" => 1u32,
                "ndf" | "2" => 2u32,
                _ => 0u32,
            };
            return Some(vec![ViewerCmd::SetLitDebug(mode)]);
        }
    }

    // Size and FOV
    if l.starts_with(":size") || l.starts_with("size ") {
        if let (Some(ws), Some(hs)) = (l.split_whitespace().nth(1), l.split_whitespace().nth(2)) {
            if let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) {
                return Some(vec![ViewerCmd::SetSize(w, h)]);
            }
        }
    }
    if l.starts_with(":fov") || l.starts_with("fov ") {
        if let Some(val) = l
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetFov(val)]);
        }
    }

    None
}
