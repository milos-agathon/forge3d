// src/viewer/terrain/shader_pbr.rs
// Enhanced PBR-like shader for terrain rendering with better lighting

pub const TERRAIN_PBR_SHADER: &str = concat!(
    include_str!("../../shaders/includes/shadow_moments.wgsl"),
    "\n",
    include_str!("shader_pbr/terrain_pbr.wgsl")
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewer_shader_exposes_typed_pcss_penumbra_primitive() {
        let source = format!(
            "{TERRAIN_PBR_SHADER}
fn pcss_penumbra_probe() -> f32 {{
    return pcss_penumbra_size(0.8, 0.4, 1.0);
}}"
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    }
}
