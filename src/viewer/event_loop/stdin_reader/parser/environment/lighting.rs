use crate::viewer::viewer_enums::ViewerCmd;

pub(super) fn parse_brdf_and_lighting(line: &str) -> Option<Vec<ViewerCmd>> {
    if line.starts_with(":brdf") || line.starts_with("brdf ") {
        if let Some(model) = line.split_whitespace().nth(1) {
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
        return Some(vec![]);
    }

    if line.starts_with(":lit-sun") || line.starts_with("lit-sun ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f32>().ok())
                .map(ViewerCmd::SetLitSun)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":lit-ibl") || line.starts_with("lit-ibl ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f32>().ok())
                .map(ViewerCmd::SetLitIbl)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":lit-rough") || line.starts_with("lit-rough ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<f32>().ok())
                .map(ViewerCmd::SetLitRough)
                .into_iter()
                .collect(),
        );
    }
    if line.starts_with(":lit-debug") || line.starts_with("lit-debug ") {
        return Some(
            line.split_whitespace()
                .nth(1)
                .map(|tok| {
                    let mode = match tok {
                        "rough" | "1" | "smoke" => 1u32,
                        "ndf" | "2" => 2u32,
                        _ => 0u32,
                    };
                    ViewerCmd::SetLitDebug(mode)
                })
                .into_iter()
                .collect(),
        );
    }

    None
}
