//! Renderer-owned WGSL assembly shared with the static verifier.

fn strip_includes(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn hybrid_kernel() -> String {
    [
        include_str!("shaders/sdf_primitives.wgsl").to_string(),
        strip_includes(include_str!("shaders/sdf_operations.wgsl")),
        strip_includes(include_str!("shaders/hybrid_traversal.wgsl")),
        strip_includes(include_str!("shaders/hybrid_terrain_traversal.wgsl")),
        strip_includes(include_str!("shaders/hybrid_kernel.wgsl")),
    ]
    .join("\n")
}

pub(crate) fn terrain() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/includes/shadow_moments.wgsl").to_string(),
        include_str!("shaders/lights.wgsl").to_string(),
        include_str!("shaders/brdf/common.wgsl").to_string(),
        include_str!("shaders/brdf/lambert.wgsl").to_string(),
        include_str!("shaders/brdf/phong.wgsl").to_string(),
        include_str!("shaders/brdf/oren_nayar.wgsl").to_string(),
        include_str!("shaders/brdf/cook_torrance.wgsl").to_string(),
        include_str!("shaders/brdf/disney_principled.wgsl").to_string(),
        include_str!("shaders/brdf/ashikhmin_shirley.wgsl").to_string(),
        include_str!("shaders/brdf/ward.wgsl").to_string(),
        include_str!("shaders/brdf/toon.wgsl").to_string(),
        include_str!("shaders/brdf/minnaert.wgsl").to_string(),
        strip_includes(include_str!("shaders/brdf/dispatch.wgsl")),
        strip_includes(include_str!("shaders/lighting.wgsl")),
        include_str!("shaders/lighting_ibl.wgsl").to_string(),
        include_str!("shaders/terrain_noise.wgsl").to_string(),
        include_str!("shaders/terrain_probes.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        strip_includes(include_str!("shaders/terrain_pbr_pom.wgsl")),
    ]
    .join("\n")
}

pub(crate) fn terrain_shadow_depth() -> String {
    [
        include_str!("shaders/includes/determinism.wgsl").to_string(),
        include_str!("shaders/terrain_shadow_depth.wgsl").to_string(),
    ]
    .join("\n")
}

#[cfg(any(test, all(feature = "enable-pbr", feature = "enable-tbn")))]
pub(crate) fn pbr() -> String {
    let shadows = crate::shadows::CsmRenderer::shader_source().replace("@group(2)", "@group(3)");
    [
        shadows,
        include_str!("shaders/lights.wgsl").to_string(),
        include_str!("shaders/brdf/common.wgsl").to_string(),
        include_str!("shaders/brdf/lambert.wgsl").to_string(),
        include_str!("shaders/brdf/phong.wgsl").to_string(),
        include_str!("shaders/brdf/oren_nayar.wgsl").to_string(),
        include_str!("shaders/brdf/cook_torrance.wgsl").to_string(),
        include_str!("shaders/brdf/disney_principled.wgsl").to_string(),
        include_str!("shaders/brdf/ashikhmin_shirley.wgsl").to_string(),
        include_str!("shaders/brdf/ward.wgsl").to_string(),
        include_str!("shaders/brdf/toon.wgsl").to_string(),
        include_str!("shaders/brdf/minnaert.wgsl").to_string(),
        strip_includes(include_str!("shaders/brdf/dispatch.wgsl")),
        strip_includes(include_str!("shaders/lighting.wgsl")),
        include_str!("shaders/lighting_ibl.wgsl").to_string(),
        include_str!("shaders/includes/tonemap_common.wgsl").to_string(),
        strip_includes(include_str!("shaders/pbr.wgsl")),
    ]
    .join("\n")
}

#[cfg(test)]
pub(crate) fn assert_valid_wgsl(source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
}

#[cfg(test)]
pub(crate) fn assert_valid_wgsl_without_gpu(label: &str, source: &str) {
    assert_valid_wgsl(source);
    eprintln!("{label}: live GPU unavailable; validated the WGSL contract statically");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_renderer_sources_are_valid_wgsl() {
        for source in [hybrid_kernel(), terrain(), terrain_shadow_depth()] {
            assert_valid_wgsl(&source);
        }
    }

    #[test]
    fn actual_pbr_terrain_and_csm_assemblies_are_valid_wgsl() {
        for source in [
            pbr(),
            terrain(),
            terrain_shadow_depth(),
            crate::shadows::CsmRenderer::shader_source().to_owned(),
        ] {
            assert_valid_wgsl(&source);
        }
    }
}
