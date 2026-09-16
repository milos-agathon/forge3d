//! Renderer-owned WGSL assembly shared with the static verifier.

fn strip_includes(source: &str) -> String {
    source
        .lines()
        .filter(|line| !line.trim_start().starts_with("#include"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// One constituent file of an assembled module. `strip` mirrors
/// [`strip_includes`]: legacy `#include` markers are dropped so the text is
/// valid WGSL. The determinism rewriter uses the path to map naga spans in
/// the assembled text back to the on-disk file.
#[derive(Clone, Copy)]
pub(crate) struct SourcePart {
    /// File path used by the test-only assembled-source mapper.
    #[allow(dead_code)]
    pub path: &'static str,
    pub text: &'static str,
    pub strip: bool,
}

/// Same-length textual rewrites a module applies to a part at assembly
/// time: (module, part path, from, to). File byte offsets survive because
/// every rewrite is length-preserving; the rewriter refuses edits that land
/// on a rewritten range, since the on-disk text differs there.
#[cfg(test)]
pub(crate) const MODULE_REWRITES: &[(&str, &str, &str, &str)] =
    &[("pbr", "src/shaders/shadows.wgsl", "@group(2)", "@group(3)")];

pub(crate) fn assemble_parts(parts: &[SourcePart]) -> String {
    parts
        .iter()
        .map(|part| {
            if part.strip {
                strip_includes(part.text)
            } else {
                part.text.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

const PART_DET: SourcePart = SourcePart {
    path: "src/shaders/includes/determinism.wgsl",
    text: include_str!("shaders/includes/determinism.wgsl"),
    strip: false,
};
#[cfg(test)]
const PART_TONEMAP_COMMON: SourcePart = SourcePart {
    path: "src/shaders/includes/tonemap_common.wgsl",
    text: include_str!("shaders/includes/tonemap_common.wgsl"),
    strip: false,
};

fn det_and(path: &'static str, text: &'static str) -> [SourcePart; 2] {
    [
        PART_DET,
        SourcePart {
            path,
            text,
            strip: false,
        },
    ]
}

pub(crate) fn hybrid_kernel() -> String {
    [
        include_str!("shaders/sdf_primitives.wgsl").to_string(),
        strip_includes(include_str!("shaders/sdf_operations.wgsl")),
        strip_includes(include_str!("shaders/hybrid_traversal.wgsl")),
        strip_includes(include_str!("shaders/hybrid_terrain_traversal.wgsl")),
        include_str!("shaders/atmosphere/prometheus_spectral_reference.wgsl").to_string(),
        strip_includes(include_str!("shaders/hybrid_kernel.wgsl")),
    ]
    .join("\n")
}

/// AETHER sky module: the established camera/sky ABI plus the spectral LUT
/// evaluator in its dedicated bind group. The legacy sky module remains a
/// separate source and cannot accidentally claim AETHER shader provenance.
pub(crate) fn aether_sky_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/sky.wgsl",
            text: include_str!("shaders/sky.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/atmosphere/evaluation_core.wgsl",
            text: include_str!("shaders/atmosphere/evaluation_core.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/atmosphere/scattering.wgsl",
            text: include_str!("shaders/atmosphere/scattering.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn aether_sky() -> String {
    assemble_parts(aether_sky_parts())
}

/// Legacy sky module: `sky.wgsl` over the determinism prelude. The file calls
/// `det_*` helpers, so a raw `include_str!` module fails naga resolution; this
/// is the production assembly for `terrain.sky.*` and `viewer.sky.*`.
pub(crate) fn sky_parts() -> Vec<SourcePart> {
    det_and("src/shaders/sky.wgsl", include_str!("shaders/sky.wgsl")).to_vec()
}

pub(crate) fn sky_module() -> String {
    assemble_parts(&sky_parts())
}

/// Display-only AETHER resolve. Atmosphere evaluation itself stays linear HDR
/// and reaches the existing tonemap operator unchanged through this assembly.
pub(crate) fn aether_blit_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_aether_blit.wgsl",
            text: include_str!("shaders/terrain_aether_blit.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn aether_blit() -> String {
    assemble_parts(aether_blit_parts())
}

/// Standalone PROMETHEUS aerial post. Keeping this source out of
/// `hybrid_kernel()` is the contract that the established traversal and
/// accumulation bind-group layouts remain byte-for-byte untouched.
pub(crate) fn prometheus_aerial_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/atmosphere/evaluation_core.wgsl",
            text: include_str!("shaders/atmosphere/evaluation_core.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/atmosphere/prometheus_aerial.wgsl",
            text: include_str!("shaders/atmosphere/prometheus_aerial.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn prometheus_aerial() -> String {
    assemble_parts(prometheus_aerial_parts())
}

pub(crate) fn terrain_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/atmosphere/evaluation_core.wgsl",
            text: include_str!("shaders/atmosphere/evaluation_core.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/shadow_moments.wgsl",
            text: include_str!("shaders/includes/shadow_moments.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/lights.wgsl",
            text: include_str!("shaders/lights.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/common.wgsl",
            text: include_str!("shaders/brdf/common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/lambert.wgsl",
            text: include_str!("shaders/brdf/lambert.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/phong.wgsl",
            text: include_str!("shaders/brdf/phong.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/oren_nayar.wgsl",
            text: include_str!("shaders/brdf/oren_nayar.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/cook_torrance.wgsl",
            text: include_str!("shaders/brdf/cook_torrance.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/disney_principled.wgsl",
            text: include_str!("shaders/brdf/disney_principled.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/ashikhmin_shirley.wgsl",
            text: include_str!("shaders/brdf/ashikhmin_shirley.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/ward.wgsl",
            text: include_str!("shaders/brdf/ward.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/toon.wgsl",
            text: include_str!("shaders/brdf/toon.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/minnaert.wgsl",
            text: include_str!("shaders/brdf/minnaert.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/dispatch.wgsl",
            text: include_str!("shaders/brdf/dispatch.wgsl"),
            strip: true,
        },
        SourcePart {
            path: "src/shaders/lighting.wgsl",
            text: include_str!("shaders/lighting.wgsl"),
            strip: true,
        },
        SourcePart {
            path: "src/shaders/lighting_ibl.wgsl",
            text: include_str!("shaders/lighting_ibl.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_noise.wgsl",
            text: include_str!("shaders/terrain_noise.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_probes.wgsl",
            text: include_str!("shaders/terrain_probes.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_pbr_pom.wgsl",
            text: include_str!("shaders/terrain_pbr_pom.wgsl"),
            strip: true,
        },
    ]
}

pub(crate) fn terrain() -> String {
    assemble_parts(terrain_parts())
}

pub(crate) fn terrain_shadow_depth_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_shadow_depth.wgsl",
            text: include_str!("shaders/terrain_shadow_depth.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn terrain_shadow_depth() -> String {
    assemble_parts(terrain_shadow_depth_parts())
}

/// IBL precompute: equirect -> cubemap conversion.
pub(crate) fn ibl_equirect_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/ibl_equirect.wgsl",
            text: include_str!("shaders/ibl_equirect.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn ibl_equirect() -> String {
    assemble_parts(ibl_equirect_parts())
}

/// IBL precompute: specular prefilter.
pub(crate) fn ibl_prefilter_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/ibl_prefilter.wgsl",
            text: include_str!("shaders/ibl_prefilter.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn ibl_prefilter() -> String {
    assemble_parts(ibl_prefilter_parts())
}

/// IBL precompute: BRDF integration LUT.
pub(crate) fn ibl_brdf_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/ibl_brdf.wgsl",
            text: include_str!("shaders/ibl_brdf.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn ibl_brdf() -> String {
    assemble_parts(ibl_brdf_parts())
}

/// HDR offscreen resolve tonemap (postprocess_tonemap entry).
#[cfg(any(test, feature = "enable-hdr-offscreen"))]
pub(crate) fn hdr_tonemap_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/postprocess_tonemap.wgsl",
            text: include_str!("shaders/postprocess_tonemap.wgsl"),
            strip: false,
        },
    ]
}

#[cfg(feature = "enable-hdr-offscreen")]
pub(crate) fn hdr_tonemap() -> String {
    assemble_parts(hdr_tonemap_parts())
}

/// Offline terrain renderer tonemap resolve.
pub(crate) fn offline_tonemap_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/tonemap_terrain_offline.wgsl",
            text: include_str!("shaders/tonemap_terrain_offline.wgsl"),
            strip: false,
        },
    ]
}

#[cfg(feature = "extension-module")]
pub(crate) fn offline_tonemap() -> String {
    assemble_parts(offline_tonemap_parts())
}

/// Viewshed analysis compute.
pub(crate) fn viewshed_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/terrain_viewshed.wgsl",
            text: include_str!("shaders/terrain_viewshed.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn viewshed() -> String {
    assemble_parts(viewshed_parts())
}

/// LIMES vector coverage binning compute.
pub(crate) fn vector_coverage_bin_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/vector_coverage_bin.wgsl",
            text: include_str!("shaders/vector_coverage_bin.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn vector_coverage_bin() -> String {
    assemble_parts(vector_coverage_bin_parts())
}

/// LIMES vector coverage rasterize.
pub(crate) fn vector_coverage_raster_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/vector_coverage_raster.wgsl",
            text: include_str!("shaders/vector_coverage_raster.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn vector_coverage_raster() -> String {
    assemble_parts(vector_coverage_raster_parts())
}

/// LIMES vector coverage resolve.
pub(crate) fn vector_coverage_resolve_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/vector_coverage_resolve.wgsl",
            text: include_str!("shaders/vector_coverage_resolve.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn vector_coverage_resolve() -> String {
    assemble_parts(vector_coverage_resolve_parts())
}

/// Every deterministic-path WGSL assembly is linted through
/// `deterministic_module_parts` (below): a deterministic-path module that
/// includes `determinism.wgsl` must appear there or it escapes the naga-IR
/// lint and the shader-proof accounting. Two includers are deliberately NOT
/// deterministic paths and are not linted: the interactive viewer terrain
/// shader (`viewer::terrain::shader_pbr`, which includes the prelude only so
/// `shadow_moments.wgsl` resolves its `det_*` calls) and the DUPLA
/// `dd_harness` substitution assemblies (`core::dd::gpu_exec`).

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
            "textureSampleLevel(terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)], terrain_vt_sampler, atlas_uv, 0.0)",
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
#[cfg(test)]
pub(crate) fn terrain_visbuffer_write_parts() -> Vec<SourcePart> {
    let mut parts = terrain_parts().to_vec();
    parts.push(SourcePart {
        path: "src/shaders/terrain_visbuffer_write.wgsl",
        text: include_str!("shaders/terrain_visbuffer_write.wgsl"),
        strip: false,
    });
    parts
}

pub(crate) fn terrain_visbuffer_write(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visbuffer_write.wgsl").to_string(),
    ]
    .join("\n")
}

/// TESSELLA pass 2: the shared terrain module plus the full-screen material
/// resolve that decodes pass 1's identity and shades each visible pixel once.
#[cfg(test)]
pub(crate) fn terrain_visbuffer_resolve_parts() -> Vec<SourcePart> {
    let mut parts = terrain_parts().to_vec();
    parts.push(SourcePart {
        path: "src/shaders/terrain_visibility_fullscreen.wgsl",
        text: include_str!("shaders/terrain_visibility_fullscreen.wgsl"),
        strip: false,
    });
    parts
}

pub(crate) fn terrain_visbuffer_resolve(bindless: bool) -> String {
    [
        terrain_base(bindless),
        include_str!("shaders/terrain_visibility_fullscreen.wgsl").to_string(),
    ]
    .join("\n")
}

/// File composition of [`pbr`]: the first three parts are what
/// `CsmRenderer::shader_source()` concatenates; its `@group(2)`->`@group(3)`
/// remap is byte-length-preserving, so file offsets still line up with the
/// assembled text.
#[cfg(test)]
pub(crate) fn pbr_parts() -> &'static [SourcePart] {
    &[
        SourcePart {
            path: "src/shaders/includes/determinism.wgsl",
            text: include_str!("shaders/includes/determinism.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/shadow_moments.wgsl",
            text: include_str!("shaders/includes/shadow_moments.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/shadows.wgsl",
            text: include_str!("shaders/shadows.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/lights.wgsl",
            text: include_str!("shaders/lights.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/common.wgsl",
            text: include_str!("shaders/brdf/common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/lambert.wgsl",
            text: include_str!("shaders/brdf/lambert.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/phong.wgsl",
            text: include_str!("shaders/brdf/phong.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/oren_nayar.wgsl",
            text: include_str!("shaders/brdf/oren_nayar.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/cook_torrance.wgsl",
            text: include_str!("shaders/brdf/cook_torrance.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/disney_principled.wgsl",
            text: include_str!("shaders/brdf/disney_principled.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/ashikhmin_shirley.wgsl",
            text: include_str!("shaders/brdf/ashikhmin_shirley.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/ward.wgsl",
            text: include_str!("shaders/brdf/ward.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/toon.wgsl",
            text: include_str!("shaders/brdf/toon.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/minnaert.wgsl",
            text: include_str!("shaders/brdf/minnaert.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/brdf/dispatch.wgsl",
            text: include_str!("shaders/brdf/dispatch.wgsl"),
            strip: true,
        },
        SourcePart {
            path: "src/shaders/lighting.wgsl",
            text: include_str!("shaders/lighting.wgsl"),
            strip: true,
        },
        SourcePart {
            path: "src/shaders/lighting_ibl.wgsl",
            text: include_str!("shaders/lighting_ibl.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/includes/tonemap_common.wgsl",
            text: include_str!("shaders/includes/tonemap_common.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/pbr.wgsl",
            text: include_str!("shaders/pbr.wgsl"),
            strip: true,
        },
    ]
}

/// File composition of the CSM module (same concatenation as
/// `crate::shadows::CSM_SHADER_SOURCE`).
#[cfg(test)]
pub(crate) fn csm_parts() -> &'static [SourcePart] {
    &[
        PART_DET,
        SourcePart {
            path: "src/shaders/includes/shadow_moments.wgsl",
            text: include_str!("shaders/includes/shadow_moments.wgsl"),
            strip: false,
        },
        SourcePart {
            path: "src/shaders/shadows.wgsl",
            text: include_str!("shaders/shadows.wgsl"),
            strip: false,
        },
    ]
}

/// File composition of the tone-mapping module (same concatenation as
/// `crate::pipeline::pbr::tone_map_shader_source()`).
#[cfg(test)]
pub(crate) fn tone_map_parts() -> &'static [SourcePart] {
    &[
        PART_DET,
        PART_TONEMAP_COMMON,
        SourcePart {
            path: "src/shaders/tone_map.wgsl",
            text: include_str!("shaders/tone_map.wgsl"),
            strip: false,
        },
    ]
}

pub(crate) fn det_probe_parts() -> Vec<SourcePart> {
    det_and(
        "src/shaders/det_probe.wgsl",
        include_str!("shaders/det_probe.wgsl"),
    )
    .to_vec()
}

pub(crate) fn det_raster_parts() -> Vec<SourcePart> {
    det_and(
        "src/shaders/det_raster.wgsl",
        include_str!("shaders/det_raster.wgsl"),
    )
    .to_vec()
}

/// Deterministic-path modules as (name, parts) for the span-to-file
/// rewriter. Deliberately excludes the derived variants whose byte offsets
/// do not line up with any single file (`terrain_bindless`,
/// `visbuffer_*_bindless`, and the `dd_harness_*` substitution assemblies);
/// their violations live in shared files that the listed modules already
/// cover.
#[cfg(test)]
pub(crate) fn deterministic_module_parts() -> Vec<(&'static str, Vec<SourcePart>)> {
    let mut modules: Vec<(&'static str, Vec<SourcePart>)> = vec![
        ("aether_sky", aether_sky_parts().to_vec()),
        ("aether_blit", aether_blit_parts().to_vec()),
        ("prometheus_aerial", prometheus_aerial_parts().to_vec()),
        ("terrain", terrain_parts().to_vec()),
        (
            "terrain_shadow_depth",
            terrain_shadow_depth_parts().to_vec(),
        ),
        ("visbuffer_write", terrain_visbuffer_write_parts()),
        ("visbuffer_resolve", terrain_visbuffer_resolve_parts()),
        ("ibl_equirect", ibl_equirect_parts().to_vec()),
        ("ibl_prefilter", ibl_prefilter_parts().to_vec()),
        ("ibl_brdf", ibl_brdf_parts().to_vec()),
        ("hdr_tonemap", hdr_tonemap_parts().to_vec()),
        ("offline_tonemap", offline_tonemap_parts().to_vec()),
        ("viewshed", viewshed_parts().to_vec()),
        ("vector_coverage_bin", vector_coverage_bin_parts().to_vec()),
        (
            "vector_coverage_raster",
            vector_coverage_raster_parts().to_vec(),
        ),
        (
            "vector_coverage_resolve",
            vector_coverage_resolve_parts().to_vec(),
        ),
        ("csm", csm_parts().to_vec()),
        ("tone_map", tone_map_parts().to_vec()),
        ("det_probe", det_probe_parts().to_vec()),
        ("det_raster", det_raster_parts().to_vec()),
    ];
    #[cfg(any(test, all(feature = "enable-pbr", feature = "enable-tbn")))]
    modules.push(("pbr", pbr_parts().to_vec()));
    modules
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
            ("aether_sky", aether_sky()),
            ("aether_blit", aether_blit()),
            ("prometheus_aerial", prometheus_aerial()),
            ("terrain", terrain()),
            ("terrain_bindless", terrain_bindless()),
            ("visbuffer_write", terrain_visbuffer_write(false)),
            ("visbuffer_write_bindless", terrain_visbuffer_write(true)),
            ("visbuffer_resolve", terrain_visbuffer_resolve(false)),
            (
                "visbuffer_resolve_bindless",
                terrain_visbuffer_resolve(true),
            ),
            ("visbuffer_stats", stats),
        ]
    }

    #[test]
    fn assembled_renderer_sources_are_valid_wgsl() {
        for (name, source) in every_renderer_source() {
            assert_valid_wgsl(&source);
            assert!(!source.is_empty(), "{name} assembled an empty shader");
        }
    }

    #[test]
    fn production_aether_consumers_share_one_evaluation_core() {
        let marker = "AETHER's single production LUT-evaluation core";
        let core = include_str!("shaders/atmosphere/evaluation_core.wgsl");
        assert!(!core.contains("@group("));
        assert!(!core.contains("@binding("));
        for (name, source) in [
            ("sky", aether_sky()),
            ("terrain", terrain()),
            ("prometheus", prometheus_aerial()),
        ] {
            assert_eq!(
                source.matches(marker).count(),
                1,
                "{name} must assemble exactly one AETHER evaluation core"
            );
            assert!(source.contains("AETHER_EVAL_WAVELENGTHS_NM"));
            assert!(source.contains("AETHER_EVAL_CIE_XYZ"));
            assert!(source.contains("fn aether_eval_xyz_to_rgb"));
            assert!(source.contains("fn aether_eval_mu_to_unit"));
            assert!(source.contains("fn aether_eval_nu_to_unit"));
            assert!(source.contains("fn aether_eval_sample_accumulated_scattering"));
            assert!(source.contains("fn aether_eval_segment_transmittance"));
        }

        let sky = include_str!("shaders/atmosphere/scattering.wgsl");
        let terrain_source = include_str!("shaders/terrain_pbr_pom.wgsl");
        let prometheus = include_str!("shaders/atmosphere/prometheus_aerial.wgsl");
        assert!(sky.contains("aether_eval_sample_accumulated_scattering("));
        assert!(terrain_source.contains("aether_eval_sample_accumulated_scattering("));
        assert!(terrain_source.contains("aether_eval_segment_transmittance("));
        assert!(prometheus.contains("aether_eval_sample_accumulated_scattering("));
        assert!(prometheus.contains("aether_eval_segment_transmittance("));

        for duplicate in [
            "AETHER_TERRAIN_WAVELENGTHS_NM",
            "AETHER_TERRAIN_CIE_XYZ",
            "AETHER_TERRAIN_OZONE_ABSORPTION",
            "fn aether_terrain_xyz_to_rgb",
            "fn aether_terrain_spectral_xyz",
            "fn aether_terrain_mu_to_unit",
            "fn aether_terrain_nu_to_unit",
            "fn aether_terrain_load_scattering",
            "AETHER_PT_WAVELENGTHS_NM",
            "AETHER_PT_CIE_XYZ",
            "AETHER_PT_OZONE_ABSORPTION",
            "fn prometheus_xyz_to_rgb",
            "fn prometheus_mu_to_unit",
            "fn prometheus_nu_to_unit",
            "fn prometheus_segment_transmittance",
            "fn prometheus_load_scattering_texel",
            "fn atmosphere_mu_to_unit",
            "fn atmosphere_relative_cosine_to_unit",
            "fn atmosphere_load_scattering",
        ] {
            assert!(
                !sky.contains(duplicate)
                    && !terrain_source.contains(duplicate)
                    && !prometheus.contains(duplicate),
                "production consumer retained divergent evaluator {duplicate}"
            );
        }

        let stochastic = include_str!("shaders/atmosphere/prometheus_spectral_reference.wgsl");
        assert!(!stochastic.contains(marker));
        assert!(!stochastic.contains("aether_eval_sample_accumulated_scattering"));
    }

    #[test]
    fn terrain_height_sampling_is_portable_manual_bilinear() {
        let source = terrain();
        assert!(source.contains("fn sample_height_bilinear_level("));
        assert_eq!(source.matches("textureLoad(height_tex").count(), 4);
        assert!(!source.contains("textureSample(height_tex"));
        assert!(!source.contains("textureSampleLevel(height_tex"));

        // Both the ordinary geometry path and every clipmap morph lookup must
        // share the same reconstruction instead of drifting by callsite.
        assert!(source.contains("let h_raw = det_barrier(sample_height_bilinear(uv));"));
        assert!(source.contains("let h_fine = sample_height_bilinear(uv);"));
        // The three offset taps barrier `coarse_base` before the add.
        assert_eq!(
            source.matches("sample_height_bilinear(coarse_base").count()
                + source
                    .matches("sample_height_bilinear(det_barrier2(coarse_base)")
                    .count(),
            4
        );

        // Resolve reconstructs the exact vertices emitted by the visibility
        // write pass. It must therefore reuse the same helper rather than
        // silently reintroducing filterable R32Float sampling.
        let resolve = terrain_visbuffer_resolve(false);
        assert!(!resolve.contains("textureSample(height_tex"));
        assert!(!resolve.contains("textureSampleLevel(height_tex"));
        assert_eq!(resolve.matches("textureLoad(height_tex").count(), 4);
        assert!(resolve.contains("let h_fine = sample_height_bilinear(uv);"));

        // The CSM caster and visible surface must agree between texel centres.
        let shadow = terrain_shadow_depth();
        assert!(shadow.contains("fn sample_height_bilinear("));
        assert_eq!(shadow.matches("textureLoad(height_tex").count(), 4);
        assert!(!shadow.contains("textureSample(height_tex"));
        assert!(!shadow.contains("textureSampleLevel(height_tex"));
        assert!(shadow.contains("let h_raw = det_barrier(sample_height_bilinear(uv));"));
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
        //
        // The retired mask is FORMATTED rather than written as a literal: this
        // file is itself scanned by `test_visibility_write_and_resolve_are_
        // separate_shader_sources`, so a literal here would be an occurrence of
        // the very constant the gate requires to be absent from the assembly.
        let retired_primitive_mask = format!("0x{:08x}u", 0x00ff_ffffu32);
        let terrain = terrain();
        assert!(!terrain.contains("fn fs_visibility("));
        assert!(!terrain.contains(&retired_primitive_mask));
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
        assert!(!fixed.contains("terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)]"));
        // The array must be SIZED and must agree with the bind-group layout's
        // `count`; an unsized array against a counted layout entry is what
        // faulted the Vulkan device.
        assert!(bindless.contains(&format!(
            "var terrain_vt_atlas: binding_array<texture_2d<f32>, {VT_ATLAS_BINDING_COUNT}>;"
        )));
        assert!(!bindless.contains("binding_array<texture_2d<f32>>"));
        assert!(bindless
            .contains("terrain_vt_atlas[terrain_vt_atlas_layer(family_slot)], terrain_vt_sampler"));
        assert!(bindless.contains("fn terrain_vt_atlas_layer(family_slot: u32) -> u32"));
        assert_ne!(fixed, bindless);
    }
}
