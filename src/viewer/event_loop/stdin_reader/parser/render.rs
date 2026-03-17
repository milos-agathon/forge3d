use crate::viewer::viewer_enums::{parse_gi_viz_mode_token, ViewerCmd};

pub(super) fn parse_render_command(line: &str) -> Option<Vec<ViewerCmd>> {
    parse_snapshot_command(line)
        .or_else(|| parse_p5_command(line))
        .or_else(|| parse_asset_command(line))
        .or_else(|| parse_visualization_command(line))
}

fn parse_snapshot_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if !(line.starts_with(":snap")
        || line.starts_with("snap")
        || line.starts_with(":snapshot")
        || line.starts_with("snapshot"))
    {
        return None;
    }

    let mut tokens = line.split_whitespace();
    let _ = tokens.next();
    let path = tokens.next().map(|s| s.to_string());

    if let Some(size_str) = tokens.next() {
        if let Some((w_str, h_str)) = size_str
            .split_once('x')
            .or_else(|| size_str.split_once('X'))
        {
            if let (Ok(width), Ok(height)) = (w_str.parse::<u32>(), h_str.parse::<u32>()) {
                return Some(vec![ViewerCmd::SnapshotWithSize {
                    path: path.unwrap_or_else(|| "snapshot.png".to_string()),
                    width: Some(width),
                    height: Some(height),
                }]);
            }
        }
    }

    Some(vec![ViewerCmd::Snapshot(path)])
}

fn parse_p5_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if !(line.starts_with(":p5") || line.starts_with("p5 ")) {
        return None;
    }

    let sub = line.split_whitespace().nth(1).unwrap_or("");
    Some(match sub {
        "cornell" => vec![ViewerCmd::CaptureP51Cornell],
        "grid" => vec![ViewerCmd::CaptureP51Grid],
        "sweep" => vec![ViewerCmd::CaptureP51Sweep],
        "ssgi-cornell" => vec![ViewerCmd::CaptureP52SsgiCornell],
        "ssgi-temporal" => vec![ViewerCmd::CaptureP52SsgiTemporal],
        "ssr-glossy" => vec![ViewerCmd::CaptureP53SsrGlossy],
        "ssr-thickness" => vec![ViewerCmd::CaptureP53SsrThickness],
        "gi-stack" => vec![ViewerCmd::CaptureP54GiStack],
        _ => {
            println!(
                "Usage: :p5 <cornell|grid|sweep|ssgi-cornell|ssgi-temporal|ssr-glossy|ssr-thickness|gi-stack>"
            );
            vec![]
        }
    })
}

fn parse_asset_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":obj") || line.starts_with("obj ") {
        if let Some(path) = line.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::LoadObj(path.to_string())]);
        }
        return Some(vec![]);
    }
    if line.starts_with(":gltf") || line.starts_with("gltf ") {
        if let Some(path) = line.split_whitespace().nth(1) {
            return Some(vec![ViewerCmd::LoadGltf(path.to_string())]);
        }
        return Some(vec![]);
    }
    None
}

fn parse_visualization_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":viz-depth-max") || line.starts_with("viz-depth-max ") {
        if let Some(val) = line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<f32>().ok())
        {
            return Some(vec![ViewerCmd::SetVizDepthMax(val)]);
        }
        return Some(vec![]);
    }

    if !(line.starts_with(":viz") || line.starts_with("viz ")) {
        return None;
    }

    let toks: Vec<&str> = line.trim_start_matches(':').split_whitespace().collect();
    if toks.len() >= 2 && toks[0] == "viz" && toks[1] == "gi" {
        if toks.len() == 2 {
            return Some(vec![ViewerCmd::QueryGiViz]);
        }
        if let Some(mode) = parse_gi_viz_mode_token(toks[2]) {
            return Some(vec![ViewerCmd::SetGiViz(mode)]);
        }
        println!(
            "Unknown :viz gi mode '{}', expected one of none|composite|ao|ssgi|ssr",
            toks[2]
        );
        return Some(vec![]);
    }

    if toks.len() >= 2 {
        return Some(vec![ViewerCmd::SetViz(toks[1].to_string())]);
    }

    Some(vec![])
}
