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

/// Descriptor-indexing variant of [`terrain`]: the single virtual-texture atlas
/// becomes a `binding_array` indexed by the family slot. Selected only when the
/// adapter grants `TEXTURE_COMPRESSION_BC`, `TEXTURE_BINDING_ARRAY` and
/// non-uniform indexing; otherwise the compatibility assembly in [`terrain`] is
/// compiled and a `terrain_vt_bindless_atlas` degradation is recorded.
///
/// This is the last source-level substitution in the renderer: the two atlas
/// forms cannot be expressed as one WGSL declaration. `terrain_atlas_variants_are_distinct`
/// fails if either substitution silently stops matching.
/// Number of atlas textures in the bindless `binding_array`.
///
/// The WGSL array must be SIZED and must match the `count` on the bind-group
/// layout entry exactly. An unsized `binding_array` against a layout declaring
/// a count is a descriptor-array mismatch that Vulkan faults on at draw time
/// (the device is lost, with no validation error first); DX12 tolerated it.
/// `terrain/renderer/bind_groups/layouts.rs` reads this same constant.
pub(crate) const VT_ATLAS_BINDING_COUNT: u32 = 3;

pub(crate) fn terrain_bindless() -> String {
    terrain()
        .replace(
            "var terrain_vt_atlas: texture_2d<f32>;",
            &format!(
                "var terrain_vt_atlas: binding_array<texture_2d<f32>, {VT_ATLAS_BINDING_COUNT}>;"
            ),
        )
        .replace(
            "textureSampleLevel(terrain_vt_atlas, terrain_vt_sampler, atlas_uv, 0.0)",
            "textureSampleLevel(terrain_vt_atlas[family_slot], terrain_vt_sampler, atlas_uv, 0.0)",
        )
}

fn terrain_base(bindless: bool) -> String {
    if bindless {
        terrain_bindless()
    } else {
        terrain()
    }
}

/// TESSELLA pass 1: the shared terrain module plus the visibility-write
/// fragment stage. Depth + R32Uint primitive identity only, no material work.
pub(crate) fn terrain_visbuffer_write(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visbuffer_write.wgsl").to_string(),
    ]
    .join("\n")
}

/// TESSELLA pass 2: the shared terrain module plus the full-screen material
/// resolve that decodes pass 1's identity and shades each visible pixel once.
pub(crate) fn terrain_visbuffer_resolve(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visibility_fullscreen.wgsl").to_string(),
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every source the renderer can hand to naga, in the exact form the
    /// pipeline cache builds it. Nothing here is reassembled by the test: a
    /// variant that only exists inside a test proves nothing about the device.
    fn every_renderer_source() -> Vec<(&'static str, String)> {
        let stats = include_str!("shaders/terrain_visbuffer_resolve.wgsl").to_string();
        vec![
            ("hybrid_kernel", hybrid_kernel()),
            ("terrain", terrain()),
            ("terrain_bindless", terrain_bindless()),
            ("visbuffer_write", terrain_visbuffer_write(false)),
            ("visbuffer_write_bindless", terrain_visbuffer_write(true)),
            ("visbuffer_resolve", terrain_visbuffer_resolve(false)),
            ("visbuffer_resolve_bindless", terrain_visbuffer_resolve(true)),
            ("visbuffer_stats", stats),
        ]
    }

    #[test]
    fn assembled_renderer_sources_are_valid_wgsl() {
        for (name, source) in every_renderer_source() {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|err| panic!("{name} failed to parse: {err:?}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|err| panic!("{name} failed validation: {err:?}"));
        }
    }

    /// The visibility packing is defined by real files now, not by rewriting
    /// the forward module's text at assembly time.
    #[test]
    fn visibility_passes_are_distinct_modules_with_one_packing() {
        let write = terrain_visbuffer_write(false);
        let resolve = terrain_visbuffer_resolve(false);
        assert!(write.contains("fn fs_visibility("));
        assert!(write.contains("fn terrain_visbuffer_pack("));
        assert!(!write.contains("fn fs_visibility_resolve_fullscreen("));
        assert!(resolve.contains("fn fs_visibility_resolve_fullscreen("));
        assert!(!resolve.contains("fn fs_visibility("));
        for bindless in [false, true] {
            assert_ne!(
                terrain_visbuffer_write(bindless),
                terrain_visbuffer_resolve(bindless),
                "the two visibility pipelines must not share one source"
            );
        }
        // The forward module carries the packed tile/LOD identity itself; the
        // stale 24|8 packing and its string rewrite are gone.
        let terrain = terrain();
        assert!(!terrain.contains("fn fs_visibility("));
        assert!(!terrain.contains("0x00ffffffu"));
        assert!(terrain.contains("out.tile_id = ((_tile_id_lod.y & 0xfu) << 12u)"));
    }

    /// Guards the one remaining source substitution: if either atlas pattern
    /// stops matching, `terrain_bindless()` silently degenerates into the
    /// compatibility source and the descriptor-indexing path stops existing.
    #[test]
    fn terrain_atlas_variants_are_distinct() {
        let fixed = terrain();
        let bindless = terrain_bindless();
        assert!(fixed.contains("var terrain_vt_atlas: texture_2d<f32>;"));
        assert!(!fixed.contains("binding_array"));
        assert!(!fixed.contains("terrain_vt_atlas[family_slot]"));
        // The array must be SIZED and must agree with the bind-group layout's
        // `count`; an unsized array against a counted layout entry is what
        // faulted the Vulkan device.
        assert!(bindless.contains(&format!(
            "var terrain_vt_atlas: binding_array<texture_2d<f32>, {VT_ATLAS_BINDING_COUNT}>;"
        )));
        assert!(!bindless.contains("binding_array<texture_2d<f32>>"));
        assert!(bindless.contains("terrain_vt_atlas[family_slot], terrain_vt_sampler"));
        assert_ne!(fixed, bindless);
    }
}
