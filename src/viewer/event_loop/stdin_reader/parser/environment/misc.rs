use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_misc_command(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":hud") || line.starts_with("hud ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| ViewerCmd::HudToggle(matches!(tok, "on" | "1" | "true")))
                .into_iter()
                .collect(),
        );
    }

    if !(line.starts_with(":oit") || line.starts_with("oit ")) {
        return None;
    }

    let toks: Vec<&str> = line.trim_start_matches(':').split_whitespace().collect();
    if toks.len() < 2 {
        return Some(vec![ViewerCmd::GetOitMode]);
    }

    let mode = toks[1].to_lowercase();
    let enabled = !matches!(
        mode.as_str(),
        "off" | "disabled" | "standard" | "0" | "false"
    );
    let mode = match mode.as_str() {
        "on" | "1" | "true" | "auto" => "auto".to_string(),
        "wboit" => "wboit".to_string(),
        "dual_source" | "dualsource" => "dual_source".to_string(),
        "off" | "disabled" | "standard" | "0" | "false" => "standard".to_string(),
        other => other.to_string(),
    };
    Some(vec![ViewerCmd::SetOitEnabled { enabled, mode }])
}
