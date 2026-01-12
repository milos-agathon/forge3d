// src/viewer/event_loop/stdin_reader.rs
// Stdin command reader thread for the interactive viewer
// Extracted from mod.rs as part of the viewer refactoring

use std::io::BufRead;
use winit::event_loop::EventLoopProxy;

use crate::viewer::viewer_enums::{parse_gi_viz_mode_token, ViewerCmd};

/// Spawn a thread that reads stdin and sends ViewerCmd events via the proxy
pub fn spawn_stdin_reader(proxy: EventLoopProxy<ViewerCmd>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let mut iter = stdin.lock().lines();
        while let Some(Ok(line)) = iter.next() {
            let l = line.trim().to_lowercase();
            if l.is_empty() {
                continue;
            }
            if let Some(cmds) = parse_stdin_command(&l) {
                for cmd in cmds {
                    let _ = proxy.send_event(cmd);
                }
            } else if l == ":quit" || l == "quit" || l == ":exit" || l == "exit" {
                let _ = proxy.send_event(ViewerCmd::Quit);
                break;
            } else {
                print_help();
            }
        }
    });
}

/// Parse a single stdin command and return ViewerCmd(s) or None for unknown
pub fn parse_stdin_command(l: &str) -> Option<Vec<ViewerCmd>> {
    // GI seed
    if l.starts_with(":gi-seed") || l.starts_with("gi-seed ") {
        let mut it = l.split_whitespace();
        let _ = it.next();
        if let Some(val_str) = it.next() {
            if let Ok(seed) = val_str.parse::<u32>() {
                return Some(vec![ViewerCmd::SetGiSeed(seed)]);
            } else {
                println!("Usage: :gi-seed <u32>");
                return Some(vec![]);
            }
        } else {
            return Some(vec![ViewerCmd::QueryGiSeed]);
        }
    }

    // GI toggle/status
    if l.starts_with(":gi") || l.starts_with("gi ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() == 2 && toks[1] == "status" {
            return Some(vec![ViewerCmd::GiStatus]);
        } else if toks.len() == 2 && toks[1] == "off" {
            return Some(vec![
                ViewerCmd::GiToggle("ssao", false),
                ViewerCmd::GiToggle("ssgi", false),
                ViewerCmd::GiToggle("ssr", false),
            ]);
        } else if toks.len() >= 3 {
            let eff = match toks[1] {
                "ssao" | "ssgi" | "ssr" | "gtao" => toks[1],
                _ => {
                    println!("Unknown effect '{}'", toks[1]);
                    return Some(vec![]);
                }
            };
            let on = match toks[2] {
                "on" | "1" | "true" => true,
                "off" | "0" | "false" => false,
                _ => {
                    println!("Unknown state '{}', expected on/off", toks[2]);
                    return Some(vec![]);
                }
            };
            if eff == "gtao" {
                let mut cmds = vec![ViewerCmd::GiToggle("ssao", on)];
                if on {
                    cmds.push(ViewerCmd::SetSsaoTechnique(1));
                }
                return Some(cmds);
            } else {
                return Some(vec![ViewerCmd::GiToggle(
                    match eff { "ssao" => "ssao", "ssgi" => "ssgi", "ssr" => "ssr", _ => "ssao" },
                    on,
                )]);
            }
        } else {
            println!("Usage: :gi <ssao|ssgi|ssr|off|status> [on|off]");
            return Some(vec![]);
        }
    }

    // GI weights
    if l.starts_with(":ao-weight") || l.starts_with("ao-weight ") {
        return parse_float_or_query(l, ViewerCmd::SetGiAoWeight, || ViewerCmd::QueryGiAoWeight, "ao-weight <float 0..1>");
    }
    if l.starts_with(":ssgi-weight") || l.starts_with("ssgi-weight ") {
        return parse_float_or_query(l, ViewerCmd::SetGiSsgiWeight, || ViewerCmd::QueryGiSsgiWeight, "ssgi-weight <float 0..1>");
    }
    if l.starts_with(":ssr-weight") || l.starts_with("ssr-weight ") {
        return parse_float_or_query(l, ViewerCmd::SetGiSsrWeight, || ViewerCmd::QueryGiSsrWeight, "ssr-weight <float 0..1>");
    }

    // Snapshot - supports "snap path.png" or "snap path.png 1920x1920"
    if l.starts_with(":snap") || l.starts_with("snap") || l.starts_with(":snapshot") || l.starts_with("snapshot") {
        let mut tokens = l.split_whitespace();
        let _ = tokens.next(); // skip command name
        let path = tokens.next().map(|s| s.to_string());
        
        // Check for size parameter (e.g., "1920x1920")
        if let Some(size_str) = tokens.next() {
            if let Some((w_str, h_str)) = size_str.split_once('x').or_else(|| size_str.split_once('X')) {
                if let (Ok(width), Ok(height)) = (w_str.parse::<u32>(), h_str.parse::<u32>()) {
                    return Some(vec![ViewerCmd::SnapshotWithSize {
                        path: path.unwrap_or_else(|| "snapshot.png".to_string()),
                        width: Some(width),
                        height: Some(height),
                    }]);
                }
            }
        }
        
        return Some(vec![ViewerCmd::Snapshot(path)]);
    }

    // SSAO parameters
    if l.starts_with(":ssao-radius") || l.starts_with("ssao-radius ") {
        return parse_float_or_query(l, ViewerCmd::SetSsaoRadius, || ViewerCmd::QuerySsaoRadius, "ssao-radius <float>");
    }
    if l.starts_with(":ssao-intensity") || l.starts_with("ssao-intensity ") {
        return parse_float_or_query(l, ViewerCmd::SetSsaoIntensity, || ViewerCmd::QuerySsaoIntensity, "ssao-intensity <float>");
    }
    if l.starts_with(":ssao-bias") || l.starts_with("ssao-bias ") {
        return parse_float_or_query(l, ViewerCmd::SetSsaoBias, || ViewerCmd::QuerySsaoBias, "ssao-bias <float>");
    }
    if l.starts_with(":ssao-samples") || l.starts_with("ssao-samples ") {
        return parse_u32_or_query(l, ViewerCmd::SetSsaoSamples, || ViewerCmd::QuerySsaoSamples, "ssao-samples <u32>");
    }
    if l.starts_with(":ssao-directions") || l.starts_with("ssao-directions ") {
        return parse_u32_or_query(l, ViewerCmd::SetSsaoDirections, || ViewerCmd::QuerySsaoDirections, "ssao-directions <u32>");
    }
    if l.starts_with(":ssao-temporal-alpha") || l.starts_with("ssao-temporal-alpha ") {
        return parse_float_or_query(l, ViewerCmd::SetSsaoTemporalAlpha, || ViewerCmd::QuerySsaoTemporalAlpha, "ssao-temporal-alpha <0..1>");
    }
    if l.starts_with(":ssao-temporal ") || l.starts_with("ssao-temporal ") {
        return parse_bool_or_query(l, ViewerCmd::SetSsaoTemporalEnabled, || ViewerCmd::QuerySsaoTemporalEnabled, "ssao-temporal <on|off>");
    }
    if l.starts_with(":ssao-blur") || l.starts_with("ssao-blur ") {
        return parse_bool_or_query(l, ViewerCmd::SetAoBlur, || ViewerCmd::QuerySsaoBlur, "ssao-blur <on|off>");
    }
    if l.starts_with(":ssao-composite") || l.starts_with("ssao-composite ") {
        return parse_bool_or_query(l, ViewerCmd::SetSsaoComposite, || ViewerCmd::QuerySsaoComposite, "ssao-composite <on|off>");
    }
    if l.starts_with(":ssao-mul") || l.starts_with("ssao-mul ") {
        return parse_float_or_query(l, ViewerCmd::SetSsaoCompositeMul, || ViewerCmd::QuerySsaoMul, "ssao-mul <0..1>");
    }
    if l.starts_with(":ao-temporal-alpha") || l.starts_with("ao-temporal-alpha ") {
        return parse_float_or_query(l, ViewerCmd::SetAoTemporalAlpha, || ViewerCmd::QuerySsaoTemporalAlpha, "ao-temporal-alpha <0..1>");
    }
    if l.starts_with(":ao-blur") || l.starts_with("ao-blur ") {
        return parse_bool_or_query(l, ViewerCmd::SetAoBlur, || ViewerCmd::QuerySsaoBlur, "ao-blur <on|off>");
    }
    if l.starts_with(":ssao-technique") || l.starts_with("ssao-technique ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let tech = match tok {
                "ssao" | "0" => 0u32,
                "gtao" | "1" => 1u32,
                _ => 0u32,
            };
            return Some(vec![ViewerCmd::SetSsaoTechnique(tech)]);
        } else {
            return Some(vec![ViewerCmd::QuerySsaoTechnique]);
        }
    }

    // SSGI parameters
    if l.starts_with(":ssgi-steps") || l.starts_with("ssgi-steps ") {
        return parse_u32_or_query(l, ViewerCmd::SetSsgiSteps, || ViewerCmd::QuerySsgiSteps, "ssgi-steps <u32>");
    }
    if l.starts_with(":ssgi-radius") || l.starts_with("ssgi-radius ") {
        return parse_float_or_query(l, ViewerCmd::SetSsgiRadius, || ViewerCmd::QuerySsgiRadius, "ssgi-radius <float>");
    }
    if l.starts_with(":ssgi-half") || l.starts_with("ssgi-half ") {
        return parse_bool_or_query(l, ViewerCmd::SetSsgiHalf, || ViewerCmd::QuerySsgiHalf, "ssgi-half <on|off>");
    }
    if l.starts_with(":ssgi-temporal-alpha") || l.starts_with("ssgi-temporal-alpha ") {
        return parse_float_or_query(l, ViewerCmd::SetSsgiTemporalAlpha, || ViewerCmd::QuerySsgiTemporalAlpha, "ssgi-temporal-alpha <0..1>");
    }
    if l.starts_with(":ssgi-temporal ") || l.starts_with("ssgi-temporal ") {
        return parse_bool_or_query(l, ViewerCmd::SetSsgiTemporalEnabled, || ViewerCmd::QuerySsgiTemporalEnabled, "ssgi-temporal <on|off>");
    }
    if l.starts_with(":ssgi-edges") || l.starts_with("ssgi-edges ") {
        return parse_bool_or_query(l, ViewerCmd::SetSsgiEdges, || ViewerCmd::QuerySsgiEdges, "ssgi-edges <on|off>");
    }
    if l.starts_with(":ssgi-upsigma") || l.starts_with("ssgi-upsigma ") 
        || l.starts_with(":ssgi-upsample-sigma-depth") || l.starts_with("ssgi-upsample-sigma-depth ")
    {
        return parse_float_or_query(l, ViewerCmd::SetSsgiUpsampleSigmaDepth, || ViewerCmd::QuerySsgiUpsampleSigmaDepth, "ssgi-upsample-sigma-depth <float>");
    }
    if l.starts_with(":ssgi-normexp") || l.starts_with("ssgi-normexp ")
        || l.starts_with(":ssgi-upsample-sigma-normal") || l.starts_with("ssgi-upsample-sigma-normal ")
    {
        return parse_float_or_query(l, ViewerCmd::SetSsgiUpsampleSigmaNormal, || ViewerCmd::QuerySsgiUpsampleSigmaNormal, "ssgi-upsample-sigma-normal <float>");
    }

    // SSR parameters
    if l.starts_with(":ssr-max-steps") || l.starts_with("ssr-max-steps ") {
        return parse_u32_or_query(l, ViewerCmd::SetSsrMaxSteps, || ViewerCmd::QuerySsrMaxSteps, "ssr-max-steps <u32>");
    }
    if l.starts_with(":ssr-thickness") || l.starts_with("ssr-thickness ") {
        return parse_float_or_query(l, ViewerCmd::SetSsrThickness, || ViewerCmd::QuerySsrThickness, "ssr-thickness <float>");
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
            _ => {
                println!("Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness|gi-stack>");
                return Some(vec![]);
            }
        }
    }

    // OBJ/glTF loading
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

    // Visualization
    if l.starts_with(":viz") || l.starts_with("viz ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() >= 2 && toks[0] == "viz" && toks[1] == "gi" {
            if toks.len() == 2 {
                return Some(vec![ViewerCmd::QueryGiViz]);
            } else if let Some(m) = parse_gi_viz_mode_token(toks[2]) {
                return Some(vec![ViewerCmd::SetGiViz(m)]);
            } else {
                println!("Unknown :viz gi mode '{}', expected one of none|composite|ao|ssgi|ssr", toks[2]);
                return Some(vec![]);
            }
        } else if toks.len() >= 2 {
            return Some(vec![ViewerCmd::SetViz(toks[1].to_string())]);
        }
    }

    // Viz depth max
    if l.starts_with(":viz-depth-max") || l.starts_with("viz-depth-max ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetVizDepthMax(val)]);
        }
    }

    // BRDF
    if l.starts_with(":brdf") || l.starts_with("brdf ") {
        if let Some(model) = l.split_whitespace().nth(1) {
            let idx = match model {
                "lambert" | "lam" => 0u32,
                "phong" => 1u32,
                "ggx" | "cooktorrance-ggx" | "cook-torrance-ggx" | "cooktorrance" | "ct-ggx" => 4u32,
                "disney" | "disney-principled" | "principled" => 6u32,
                _ => 4u32,
            };
            return Some(vec![ViewerCmd::SetLitBrdf(idx)]);
        }
    }

    // Lit controls
    if l.starts_with(":lit-sun") || l.starts_with("lit-sun ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetLitSun(val)]);
        }
    }
    if l.starts_with(":lit-ibl") || l.starts_with("lit-ibl ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SetLitIbl(val)]);
        }
    }
    if l.starts_with(":lit-rough") || l.starts_with("lit-rough ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
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

    // IBL controls
    if l.starts_with(":ibl") || l.starts_with("ibl ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() >= 2 {
            match toks[1] {
                "on" | "1" | "true" => return Some(vec![ViewerCmd::IblToggle(true)]),
                "off" | "0" | "false" => return Some(vec![ViewerCmd::IblToggle(false)]),
                "load" => {
                    if let Some(path) = toks.get(2) {
                        return Some(vec![ViewerCmd::LoadIbl(path.to_string())]);
                    }
                }
                "intensity" => {
                    if let Some(v) = toks.get(2).and_then(|s| s.parse::<f32>().ok()) {
                        return Some(vec![ViewerCmd::IblIntensity(v)]);
                    }
                }
                "rotate" => {
                    if let Some(v) = toks.get(2).and_then(|s| s.parse::<f32>().ok()) {
                        return Some(vec![ViewerCmd::IblRotate(v)]);
                    }
                }
                "cache" => {
                    let dir = toks.get(2).map(|s| s.to_string());
                    return Some(vec![ViewerCmd::IblCache(dir)]);
                }
                "res" => {
                    if let Some(v) = toks.get(2).and_then(|s| s.parse::<u32>().ok()) {
                        return Some(vec![ViewerCmd::IblRes(v)]);
                    }
                }
                _ => {
                    if toks[1].contains('.') || toks[1].starts_with('/') || toks[1].starts_with('\\') {
                        return Some(vec![ViewerCmd::LoadIbl(toks[1].to_string())]);
                    }
                }
            }
        }
    }

    // Sky controls
    if l.starts_with(":sky ") || l == ":sky" || l.starts_with("sky ") {
        if let Some(arg) = l.split_whitespace().nth(1) {
            match arg {
                "off" | "0" | "false" => return Some(vec![ViewerCmd::SkyToggle(false)]),
                "on" | "1" | "true" => return Some(vec![ViewerCmd::SkyToggle(true)]),
                "preetham" => return Some(vec![ViewerCmd::SkyToggle(true), ViewerCmd::SkySetModel(0)]),
                "hosek-wilkie" | "hosekwilkie" | "hosek" | "hw" => return Some(vec![ViewerCmd::SkyToggle(true), ViewerCmd::SkySetModel(1)]),
                _ => println!("Unknown sky mode '{}', expected off|on|preetham|hosek-wilkie", arg),
            }
        } else {
            println!("Usage: :sky <off|on|preetham|hosek-wilkie>");
        }
        return Some(vec![]);
    }
    if l.starts_with(":sky-turbidity") || l.starts_with("sky-turbidity ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SkySetTurbidity(val)]);
        }
    }
    if l.starts_with(":sky-ground") || l.starts_with("sky-ground ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SkySetGround(val)]);
        }
    }
    if l.starts_with(":sky-exposure") || l.starts_with("sky-exposure ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SkySetExposure(val)]);
        }
    }
    if l.starts_with(":sky-sun") || l.starts_with("sky-sun ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::SkySetSunIntensity(val)]);
        }
    }

    // Fog controls
    if l.starts_with(":fog ") || l == ":fog" || l.starts_with("fog ") {
        if let Some(arg) = l.split_whitespace().nth(1) {
            let on = matches!(arg, "on" | "1" | "true");
            return Some(vec![ViewerCmd::FogToggle(on)]);
        }
        println!("Usage: :fog <on|off>");
        return Some(vec![]);
    }
    if l.starts_with(":fog-density") || l.starts_with("fog-density ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::FogSetDensity(val)]);
        }
    }
    if l.starts_with(":fog-g") || l.starts_with("fog-g ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::FogSetG(val)]);
        }
    }
    if l.starts_with(":fog-steps") || l.starts_with("fog-steps ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<u32>().ok()) {
            return Some(vec![ViewerCmd::FogSetSteps(val)]);
        }
    }
    if l.starts_with(":fog-shadow") || l.starts_with("fog-shadow ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let on = matches!(tok, "on" | "1" | "true");
            return Some(vec![ViewerCmd::FogSetShadow(on)]);
        }
    }
    if l.starts_with(":fog-temporal") || l.starts_with("fog-temporal ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::FogSetTemporal(val)]);
        }
    }
    if l.starts_with(":fog-mode") || l.starts_with("fog-mode ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let idx = match tok { "raymarch" | "rm" | "0" => 0u32, "froxels" | "fx" | "1" => 1u32, _ => 0u32 };
            return Some(vec![ViewerCmd::SetFogMode(idx)]);
        }
    }
    if l.starts_with(":fog-preset") || l.starts_with("fog-preset ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let idx = match tok { "low" | "0" => 0u32, "med" | "medium" | "1" => 1u32, _ => 2u32 };
            return Some(vec![ViewerCmd::FogPreset(idx)]);
        }
    }
    if l.starts_with(":fog-half") || l.starts_with("fog-half ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let on = matches!(tok, "on" | "1" | "true");
            return Some(vec![ViewerCmd::FogHalf(on)]);
        }
    }
    if l.starts_with(":fog-edges") || l.starts_with("fog-edges ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let on = matches!(tok, "on" | "1" | "true");
            return Some(vec![ViewerCmd::FogEdges(on)]);
        }
    }
    if l.starts_with(":fog-upsigma") || l.starts_with("fog-upsigma ") {
        if let Some(val) = l.split_whitespace().nth(1).and_then(|s| s.parse::<f32>().ok()) {
            return Some(vec![ViewerCmd::FogUpsigma(val)]);
        }
    }

    // HUD
    if l.starts_with(":hud") || l.starts_with("hud ") {
        if let Some(tok) = l.split_whitespace().nth(1) {
            let on = matches!(tok, "on" | "1" | "true");
            return Some(vec![ViewerCmd::HudToggle(on)]);
        }
    }

    // P0.1/M1: OIT (Order-Independent Transparency)
    if l.starts_with(":oit") || l.starts_with("oit ") {
        let toks: Vec<&str> = l.trim_start_matches(":").split_whitespace().collect();
        if toks.len() >= 2 {
            let mode = toks[1].to_lowercase();
            let enabled = !matches!(mode.as_str(), "off" | "disabled" | "standard" | "0" | "false");
            let mode_str = match mode.as_str() {
                "on" | "1" | "true" | "auto" => "auto".to_string(),
                "wboit" => "wboit".to_string(),
                "dual_source" | "dualsource" => "dual_source".to_string(),
                "off" | "disabled" | "standard" | "0" | "false" => "standard".to_string(),
                other => other.to_string(),
            };
            return Some(vec![ViewerCmd::SetOitEnabled { enabled, mode: mode_str }]);
        } else {
            return Some(vec![ViewerCmd::GetOitMode]);
        }
    }

    None
}

// Helper to parse float value or query
fn parse_float_or_query<F, Q>(l: &str, set_cmd: F, query_cmd: Q, usage: &str) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(f32) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    let mut it = l.split_whitespace();
    let _ = it.next();
    if let Some(val_str) = it.next() {
        if let Ok(val) = val_str.parse::<f32>() {
            Some(vec![set_cmd(val)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

// Helper to parse u32 value or query
fn parse_u32_or_query<F, Q>(l: &str, set_cmd: F, query_cmd: Q, usage: &str) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(u32) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    let mut it = l.split_whitespace();
    let _ = it.next();
    if let Some(val_str) = it.next() {
        if let Ok(val) = val_str.parse::<u32>() {
            Some(vec![set_cmd(val)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

// Helper to parse bool value or query
fn parse_bool_or_query<F, Q>(l: &str, set_cmd: F, query_cmd: Q, usage: &str) -> Option<Vec<ViewerCmd>>
where
    F: FnOnce(bool) -> ViewerCmd,
    Q: FnOnce() -> ViewerCmd,
{
    if let Some(tok) = l.split_whitespace().nth(1) {
        let state = if tok.eq_ignore_ascii_case("on") || tok == "1" || tok.eq_ignore_ascii_case("true") {
            Some(true)
        } else if tok.eq_ignore_ascii_case("off") || tok == "0" || tok.eq_ignore_ascii_case("false") {
            Some(false)
        } else {
            None
        };
        if let Some(on) = state {
            Some(vec![set_cmd(on)])
        } else {
            println!("Usage: :{}", usage);
            Some(vec![])
        }
    } else {
        Some(vec![query_cmd()])
    }
}

fn print_help() {
    println!(
        "Commands:\n  :gi <ssao|ssgi|ssr> <on|off>\n  :viz <material|normal|depth|gi|lit>\n  :viz-depth-max <float>\n  :ibl <on|off|load <path>|intensity <f>|rotate <deg>|cache <dir>|res <u32>>\n  :brdf <lambert|phong|ggx|disney>\n  :snapshot [path]\n  :obj <path> | :gltf <path>\n  :sky off|on|preetham|hosek-wilkie | :sky-turbidity <f> | :sky-ground <f> | :sky-exposure <f> | :sky-sun <f>\n  :fog <on|off> | :fog-density <f> | :fog-g <f> | :fog-steps <u32> | :fog-shadow <on|off> | :fog-temporal <0..0.9> | :fog-mode <raymarch|froxels> | :fog-preset <low|med|high>\n  :oit <auto|wboit|dual_source|off> (Order-Independent Transparency)\n  Lit:  :lit-sun <float> | :lit-ibl <float>\n  SSAO: :ssao-technique <ssao|gtao> | :ssao-radius <f> | :ssao-intensity <f> | :ssao-composite <on|off> | :ssao-mul <0..1>\n  SSGI: :ssgi-steps <u32> | :ssgi-radius <f> | :ssgi-half <on|off> | :ssgi-temporal <on|off> | :ssgi-temporal-alpha <0..1> | :ssgi-edges <on|off> | :ssgi-upsample-sigma-depth <f> | :ssgi-upsample-sigma-normal <f>\n  SSR:  :ssr-max-steps <u32> | :ssr-thickness <f>\n  P5:   :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness>\n  :quit"
    );
}
