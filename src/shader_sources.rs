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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_renderer_sources_are_valid_wgsl() {
        let terrain = terrain();
        let bindless_terrain = terrain
            .replace(
                "var terrain_vt_atlas: texture_2d<f32>;",
                "var terrain_vt_atlas: binding_array<texture_2d<f32>>;",
            )
            .replace(
                "textureSampleLevel(terrain_vt_atlas, terrain_vt_sampler, atlas_uv, 0.0)",
                "textureSampleLevel(terrain_vt_atlas[family_slot], terrain_vt_sampler, atlas_uv, 0.0)",
            );
        for source in [
            hybrid_kernel(),
            terrain,
            bindless_terrain,
            include_str!("shaders/terrain_visbuffer_resolve.wgsl").to_string(),
        ] {
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}
