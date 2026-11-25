// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = concat!(
    include_str!("../../shaders/includes/determinism.wgsl"),
    "\n",
    include_str!("../../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("shader_pbr/terrain_pbr.wgsl")
);

pub(crate) fn default_viewer_pcss_technique_params() -> [f32; 4] {
    [
        crate::shadows::DEFAULT_PCSS_BLOCKER_RADIUS_TEXELS,
        crate::shadows::DEFAULT_PCSS_FILTER_RADIUS_TEXELS,
        0.0005,
        crate::shadows::DEFAULT_PCSS_LIGHT_SIZE,
    ]
}

const SMOOTH_NORMALS_LINE: &str = "const SMOOTH_NORMALS: bool = false;";
const RIDGE_GAIN_LINE: &str = "const RIDGE_GAIN: f32 = 1.0;";

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.trim(), "1" | "true" | "on"))
        .unwrap_or(false)
}

fn env_ridge_gain() -> Option<f32> {
    std::env::var("FORGE3D_TERRAIN_RIDGE_GAIN")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|g| g.is_finite() && *g >= 0.0)
}

/// The terrain shader with its load-time switches applied.
///
/// `FORGE3D_TERRAIN_SMOOTH_NORMALS=1` blends per-texel normals bilinearly
/// (smooth shading; no per-texel streaks up steep walls).
/// `FORGE3D_TERRAIN_RIDGE_GAIN=<f32>` scales the screen-space ridge term in the
/// preserve-colours path (0 disables it). Unset, the source is unchanged.
pub fn terrain_pbr_shader_source() -> std::borrow::Cow<'static, str> {
    apply_switches(env_flag("FORGE3D_TERRAIN_SMOOTH_NORMALS"), env_ridge_gain())
}

fn apply_switches(smooth: bool, ridge: Option<f32>) -> std::borrow::Cow<'static, str> {
    if !smooth && ridge.is_none() {
        return std::borrow::Cow::Borrowed(TERRAIN_PBR_SHADER);
    }
    let mut src = TERRAIN_PBR_SHADER.to_string();
    if smooth {
        src = src.replacen(SMOOTH_NORMALS_LINE, "const SMOOTH_NORMALS: bool = true;", 1);
    }
    if let Some(gain) = ridge {
        src = src.replacen(
            RIDGE_GAIN_LINE,
            &format!("const RIDGE_GAIN: f32 = {gain:?};"),
            1,
        );
    }
    std::borrow::Cow::Owned(src)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(src)));
    }

    #[test]
    fn viewer_shader_has_valid_blocker_dependent_pcss_dispatch() {
        validate(TERRAIN_PBR_SHADER);

        assert!(TERRAIN_PBR_SHADER.contains("if (technique == 2u)"));
        assert!(TERRAIN_PBR_SHADER.contains("pcss_blocker_search("));
        assert!(TERRAIN_PBR_SHADER.contains("avg_blocker_depth"));
        assert!(TERRAIN_PBR_SHADER.contains("csm_uniforms.technique_params.w"));
    }

    #[test]
    fn viewer_uploads_nonzero_pcss_defaults() {
        assert_eq!(
            default_viewer_pcss_technique_params(),
            [6.0, 4.0, 0.0005, 1.0]
        );
    }

    #[test]
    fn switch_lines_exist_once() {
        for line in [SMOOTH_NORMALS_LINE, RIDGE_GAIN_LINE] {
            assert_eq!(TERRAIN_PBR_SHADER.matches(line).count(), 1, "{line}");
        }
    }

    #[test]
    fn switches_off_leave_shader_unchanged() {
        assert!(matches!(
            apply_switches(false, None),
            std::borrow::Cow::Borrowed(s) if s == TERRAIN_PBR_SHADER
        ));
    }

    #[test]
    fn switched_shader_is_valid() {
        let src = apply_switches(true, Some(0.0));
        assert!(src.contains("const SMOOTH_NORMALS: bool = true;"));
        assert!(src.contains("const RIDGE_GAIN: f32 = 0.0;"));
        validate(&src);
    }
}
