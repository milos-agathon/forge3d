use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_ibl_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if !(line.starts_with(":ibl") || line.starts_with("ibl ")) {
        return None;
    }

    let toks: Vec<&str> = line.trim_start_matches(':').split_whitespace().collect();
    if toks.len() < 2 {
        return Some(vec![]);
    }

    let cmd = match toks[1] {
        "on" | "1" | "true" => Some(ViewerCmd::IblToggle(true)),
        "off" | "0" | "false" => Some(ViewerCmd::IblToggle(false)),
        "load" => toks.get(2).map(|path| ViewerCmd::LoadIbl(path.to_string())),
        "intensity" => toks
            .get(2)
            .and_then(|s| s.parse::<f32>().ok())
            .map(ViewerCmd::IblIntensity),
        "rotate" => toks
            .get(2)
            .and_then(|s| s.parse::<f32>().ok())
            .map(ViewerCmd::IblRotate),
        "cache" => Some(ViewerCmd::IblCache(toks.get(2).map(|s| s.to_string()))),
        "res" => toks
            .get(2)
            .and_then(|s| s.parse::<u32>().ok())
            .map(ViewerCmd::IblRes),
        _ => {
            if toks[1].contains('.') || toks[1].starts_with('/') || toks[1].starts_with('\\') {
                Some(ViewerCmd::LoadIbl(toks[1].to_string()))
            } else {
                None
            }
        }
    };

    Some(cmd.into_iter().collect())
}
