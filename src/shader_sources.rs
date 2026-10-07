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

/// Replace exactly one occurrence of `from`; a missing or duplicated anchor is
/// an assembly error (the base shader drifted), never a silent no-op.
#[cfg(feature = "splat-fusion")]
fn replace_once(
    text: &str,
    from: &str,
    to: &str,
    what: &str,
) -> Result<String, crate::core::error::RenderError> {
    match text.matches(from).count() {
        1 => Ok(text.replacen(from, to, 1)),
        found => Err(crate::core::error::RenderError::Render(format!(
            "fused kernel assembly: anchor `{what}` matched {found} times in the hybrid \
             kernel sources (expected exactly 1)"
        ))),
    }
}

/// Text between `start` (inclusive) and the first `end` after it (inclusive).
#[cfg(feature = "splat-fusion")]
fn extract_block(
    text: &str,
    start: &str,
    end: &str,
    what: &str,
) -> Result<String, crate::core::error::RenderError> {
    let missing = || {
        crate::core::error::RenderError::Render(format!(
            "fused kernel assembly: block `{what}` not found in its source shader"
        ))
    };
    let begin = text.find(start).ok_or_else(missing)?;
    let stop = text[begin..].find(end).ok_or_else(missing)? + begin + end.len();
    Ok(text[begin..stop].to_string())
}

/// Anchored edits that route `main_terrain` / `main_terrain_gbuffer` through
/// the fused occlusion model: (what, from, to). Each must match exactly once.
#[cfg(feature = "splat-fusion")]
const FUSED_TERRAIN_EDITS: &[(&str, &str, &str)] = &[
    // Every fused render is seamless (camera_flags = 1): spatial reuse is
    // self-only (pt_restir_spatial.wgsl) and temporal reuse is per pixel, so
    // the reservoir's sun-disc sample is tile-safe. Without this edit tiles
    // would shade from the disc centre only (hard shadows).
    (
        "seamless reservoir sun direction",
        concat!(
            "        if (uniforms.camera_flags == 0u && prev_valid) {\n",
            "            sun_dir = normalize(prev_r.sample.direction);",
        ),
        concat!(
            "        if (prev_valid) {\n",
            "            sun_dir = normalize(prev_r.sample.direction);",
        ),
    ),
    (
        "terrain shading normal seam",
        "fn terrain_normal_at(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {",
        "fn terrain_normal_at_base(p: vec3<f32>, cx: u32, cz: u32) -> vec3<f32> {",
    ),
    (
        "ReSTIR candidate target function",
        concat!(
            "            let ndotl = max(dot(n, wi), 0.0);\n",
            "            let target_pdf = select(0.0, 1.0, terrain_luminance(albedo * lighting.light_color * ndotl) > 0.0);\n",
            "            if (target_pdf > 0.0) {\n",
            "                cand.sample.position = hit.point;\n",
            "                cand.sample.light_index = 0u;\n",
            "                cand.sample.direction = wi;\n",
            "                cand.sample.intensity = terrain_luminance(lighting.light_color);\n",
            "                cand.sample.light_type = 1u;\n",
            "                cand.w_sum = cand.w_sum + target_pdf;\n",
            "                cand.m = cand.m + 1u;\n",
            "                cand.target_pdf = target_pdf;\n",
            "            }\n",
        ),
        "            fusion_restir_candidate(&cand, hit.point, n, albedo, wi, &st);\n",
    ),
    (
        "candidate reservoir publish",
        "    terrain_reservoirs_curr[pix] = cand;",
        "    fusion_restir_publish(&cand);\n    terrain_reservoirs_curr[pix] = cand;",
    ),
    (
        "sun shading visibility",
        concat!(
            "            var vis = 1.0;\n",
            "            if (lighting.shadows_enabled != 0u && intersect_shadow_ray(sray, 1e30)) {\n",
            "                vis = 0.0;\n",
            "            }\n",
        ),
        concat!(
            "            var vis = 1.0;\n",
            "            if (lighting.shadows_enabled != 0u) {\n",
            "                vis = fusion_shading_visibility(sray, prev_valid, prev_r.sample.intensity);\n",
            "            }\n",
        ),
    ),
    (
        "sun surface response",
        "            sun = albedo * sampled_spectrum * nd * vis;",
        "            sun = fusion_surface_response(n, -rd, sun_dir, albedo) * sampled_spectrum * nd * vis;",
    ),
    (
        "AOV centre ray",
        "        let chit = intersect_hybrid(cray);",
        concat!(
            "        fusion_deterministic = true;\n",
            "        let chit = intersect_hybrid(cray);\n",
            "        fusion_deterministic = false;",
        ),
    ),
    (
        "fused AOVs",
        concat!(
            "            textureStore(aov_visibility, coord, vec4<f32>(select(0.0, 1.0, is_hit), 0.0, 0.0, 1.0));\n",
            "        }\n",
        ),
        concat!(
            "            textureStore(aov_visibility, coord, vec4<f32>(select(0.0, 1.0, is_hit), 0.0, 0.0, 1.0));\n",
            "        }\n",
            "        fusion_write_aovs(coord, chit, calbedo, is_hit);\n",
        ),
    ),
    (
        "G-buffer centre ray",
        concat!(
            "    let ray = Ray(center_camera.origin, 1e-3, center_camera.direction, 1e30);\n",
            "\n",
            "    let hit = intersect_hybrid(ray);",
        ),
        concat!(
            "    let ray = Ray(center_camera.origin, 1e-3, center_camera.direction, 1e30);\n",
            "\n",
            "    fusion_deterministic = true;\n",
            "    let hit = intersect_hybrid(ray);",
        ),
    ),
];

/// The four traversal seams the fused wrappers replace (renamed `*_base`).
#[cfg(feature = "splat-fusion")]
const FUSED_TRAVERSAL_SEAMS: &[(&str, &str)] = &[
    (
        "fn intersect_hybrid(ray: Ray) -> HybridHitResult {",
        "fn intersect_hybrid_base(ray: Ray) -> HybridHitResult {",
    ),
    (
        "fn get_surface_properties(hit: HybridHitResult) -> vec3f {",
        "fn get_surface_properties_base(hit: HybridHitResult) -> vec3f {",
    ),
    (
        "fn intersect_shadow_ray(ray: Ray, max_distance: f32) -> bool {",
        "fn intersect_shadow_ray_base(ray: Ray, max_distance: f32) -> bool {",
    ),
    (
        "fn intersect_ibl_occlusion_ray(ray: Ray, max_distance: f32) -> bool {",
        "fn intersect_ibl_occlusion_ray_base(ray: Ray, max_distance: f32) -> bool {",
    ),
];

/// SPLAT-FUSED kernel: the hybrid ReSTIR kernel specialized for the fused
/// splat + LiDAR + terrain scene. It is the SAME integrator — every source
/// file of [`hybrid_kernel`] is reused verbatim — with:
///
/// * the four traversal seams (`intersect_hybrid`, `get_surface_properties`,
///   `intersect_shadow_ray`, `intersect_ibl_occlusion_ray`) renamed `*_base`
///   so the fused wrappers in `fusion/unified_occlusion.wgsl` take their
///   place for every existing caller;
/// * the reservoir candidate's target function, the sun shading visibility
///   and the surface response in `main_terrain` routed through
///   `shadow_transmittance` / `eval_brdf`;
/// * the analytic Gaussian kernel, the unified occlusion module and the
///   shared BRDF library appended.
///
/// The default [`hybrid_kernel`] text is untouched (its shader proofs and
/// runtime contracts keep applying); a base-shader edit that moves an anchor
/// fails here loudly instead of silently dropping the fused path.
#[cfg(feature = "splat-fusion")]
pub(crate) fn fused_kernel() -> Result<String, crate::core::error::RenderError> {
    // Checkouts may carry CRLF; anchors are written with LF.
    let lf = |text: &str| text.replace("\r\n", "\n");

    let mut traversal = strip_includes(&lf(include_str!("shaders/hybrid_traversal.wgsl")));
    for (from, to) in FUSED_TRAVERSAL_SEAMS {
        traversal = replace_once(&traversal, from, to, from)?;
    }
    let mut terrain = strip_includes(&lf(include_str!("shaders/hybrid_terrain_traversal.wgsl")));
    for (what, from, to) in FUSED_TERRAIN_EDITS {
        terrain = replace_once(&terrain, from, to, what)?;
    }

    // Shared BRDF library: the dispatcher constants + ShadingParamsGPU come
    // from lighting.wgsl and VolumetricParams from volumetric.wgsl, extracted
    // verbatim so the fused kernel reuses those layouts without pulling in
    // the raster pipelines' bind groups.
    let brdf_prelude = extract_block(
        &lf(include_str!("shaders/lighting.wgsl")),
        "const BRDF_LAMBERT: u32 = 0u;",
        "};",
        "BRDF constants + ShadingParamsGPU",
    )?;
    if !brdf_prelude.contains("struct ShadingParamsGPU") {
        return Err(crate::core::error::RenderError::Render(
            "fused kernel assembly: ShadingParamsGPU is no longer adjacent to the BRDF \
             constants in lighting.wgsl"
                .into(),
        ));
    }
    let volumetric_params = extract_block(
        &lf(include_str!("shaders/volumetric.wgsl")),
        "struct VolumetricParams {",
        "\n}",
        "VolumetricParams",
    )?;

    Ok([
        lf(include_str!("shaders/includes/determinism.wgsl")),
        // Same literal as lights.wgsl, which the BRDF library expects in scope.
        "const PI: f32 = 3.14159265359;".to_string(),
        brdf_prelude,
        volumetric_params,
        lf(include_str!("shaders/brdf/common.wgsl")),
        lf(include_str!("shaders/brdf/lambert.wgsl")),
        lf(include_str!("shaders/brdf/phong.wgsl")),
        lf(include_str!("shaders/brdf/oren_nayar.wgsl")),
        lf(include_str!("shaders/brdf/cook_torrance.wgsl")),
        lf(include_str!("shaders/brdf/disney_principled.wgsl")),
        lf(include_str!("shaders/brdf/ashikhmin_shirley.wgsl")),
        lf(include_str!("shaders/brdf/ward.wgsl")),
        lf(include_str!("shaders/brdf/toon.wgsl")),
        lf(include_str!("shaders/brdf/minnaert.wgsl")),
        strip_includes(&lf(include_str!("shaders/brdf/dispatch.wgsl"))),
        lf(include_str!("shaders/sdf_primitives.wgsl")),
        strip_includes(&lf(include_str!("shaders/sdf_operations.wgsl"))),
        traversal,
        terrain,
        lf(include_str!(
            "shaders/atmosphere/prometheus_spectral_reference.wgsl"
        )),
        strip_includes(&lf(include_str!("shaders/hybrid_kernel.wgsl"))),
        lf(include_str!("shaders/splat/gaussian_intersect.wgsl")),
        lf(include_str!("shaders/fusion/unified_occlusion.wgsl")),
    ]
    .join("\n"))
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

/// DIFFERENTIA inverse module: the forward hybrid kernel plus the three
/// reverse-mode files (loss adjoint, reverse shading, edge sampling). The
/// inverse entry points share every forward declaration and add their own
/// bindings at group 0 binding 2, group 2 bindings 11-12 and group 3
/// bindings 8-10.
#[cfg(feature = "enable-inverse-pt")]
pub(crate) fn inverse_kernel() -> String {
    [
        hybrid_kernel(),
        include_str!("shaders/pt_inverse_loss.wgsl").to_string(),
        include_str!("shaders/pt_inverse_shade.wgsl").to_string(),
        include_str!("shaders/pt_edge_sample.wgsl").to_string(),
    ]
    .join("\n")
}

pub(crate) fn terrain() -> String {
    assemble_parts(terrain_parts())
}

#[cfg(feature = "enable-globe")]
pub(crate) fn orbis_globe_background() -> &'static str {
    include_str!("shaders/orbis_globe_background.wgsl")
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

pub(crate) fn clipmap_lod_select_parts() -> [SourcePart; 2] {
    det_and(
        "src/shaders/clipmap_lod_select.wgsl",
        include_str!("shaders/clipmap_lod_select.wgsl"),
    )
}

pub(crate) fn clipmap_lod_select() -> String {
    assemble_parts(&clipmap_lod_select_parts())
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
        assert!(source
            .contains("let h_fine = clipmap_sample_height_level(uv, fine_level, height_dims);"));
        // The three offset taps barrier `level_base` before the add.
        assert_eq!(
            source.matches("sample_height_bilinear(level_base").count()
                + source
                    .matches("sample_height_bilinear(det_barrier2(level_base)")
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
        assert!(resolve
            .contains("let h_fine = clipmap_sample_height_level(uv, fine_level, height_dims);"));

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

    #[test]
    #[cfg(feature = "enable-inverse-pt")]
    fn inverse_kernel_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(&inverse_kernel()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[cfg(all(test, feature = "splat-fusion"))]
mod fused_kernel_tests {
    use super::*;

    #[test]
    fn fused_kernel_assembles_and_validates() {
        let source = fused_kernel().expect("fused kernel assembly");
        assert_valid_wgsl(&source);
        // Every entry point of the base integrator survives the assembly.
        for entry in [
            "fn main(",
            "fn main_terrain(",
            "fn main_terrain_gbuffer(",
            "fn main_terrain_publish(",
        ] {
            assert_eq!(source.matches(entry).count(), 1, "{entry}");
        }
    }

    #[test]
    fn default_hybrid_kernel_is_untouched_by_the_fused_specialization() {
        let base = hybrid_kernel();
        assert!(!base.contains("fusion_"));
        assert!(!base.contains("shadow_transmittance"));
        assert!(base.contains("fn intersect_hybrid(ray: Ray) -> HybridHitResult {"));
    }

    /// Text of `fn name(` up to the next top-level `fn` / `@compute`.
    fn body<'a>(source: &'a str, name: &str) -> &'a str {
        let start = source
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("missing fn {name}"));
        let rest = &source[start + 3..];
        let end = rest
            .find("\nfn ")
            .into_iter()
            .chain(rest.find("\n@compute"))
            .min()
            .unwrap_or(rest.len());
        &rest[..end]
    }

    #[test]
    fn reservoir_target_function_calls_shadow_transmittance() {
        let source = fused_kernel().unwrap();
        // main_terrain generates its candidate through the fused target...
        let main_terrain = body(&source, "main_terrain");
        assert!(
            main_terrain.contains("fusion_restir_candidate(&cand, hit.point, n, albedo, wi, &st);")
        );
        assert!(!main_terrain.contains("let target_pdf = select(0.0, 1.0"));
        // ...whose light-visibility term is the unified occlusion query...
        let candidate = body(&source, "fusion_restir_candidate");
        assert!(candidate.contains("visibility = shadow_transmittance(sray)"));
        assert!(candidate.contains("let target_pdf = floor_pdf + (1.0 - floor_pdf) * visibility;"));
        // ...and that query is the product of the three sub-occluders.
        let query = body(&source, "shadow_transmittance");
        assert!(query.contains("fusion_shadow_parts(ray, ray.tmax, true)"));
        assert!(query.contains("parts.x * parts.y * parts.z"));
        // Shading visibility and the surface response go through the same
        // query and the shared BRDF dispatcher.
        assert!(main_terrain.contains(
            "vis = fusion_shading_visibility(sray, prev_valid, prev_r.sample.intensity);"
        ));
        assert!(main_terrain.contains("fusion_surface_response(n, -rd, sun_dir, albedo)"));
        assert!(body(&source, "fusion_surface_response").contains("eval_brdf("));
        assert!(body(&source, "fusion_shading_visibility").contains("shadow_transmittance(sray)"));
    }

    #[test]
    fn traversal_uses_the_analytic_gaussian_kernel_not_a_rasterized_splat() {
        let source = fused_kernel().unwrap();
        assert!(body(&source, "fusion_splat_page_closest").contains("gaussian_closest_hit("));
        assert!(body(&source, "fusion_splat_page_shadow").contains("gaussian_any_hit("));
        let kernel = body(&source, "gaussian_response");
        assert!(kernel.contains("out.t_star = -dot(delta, sd) / out.a;"));
        assert!(kernel.contains("exp(-0.5 * out.g_star)"));
        // One structure, three leaf kinds, dispatched in a single traversal.
        let closest = body(&source, "fusion_closest_hit");
        for needle in [
            "kind == FUSION_KIND_TERRAIN",
            "terrain_trace(",
            "fusion_splat_page_closest(",
            "fusion_point_page_closest(",
        ] {
            assert!(closest.contains(needle), "{needle}");
        }
        let gaussian = include_str!("shaders/splat/gaussian_intersect.wgsl");
        for projected in ["textureSample", "@vertex", "@fragment", "@group"] {
            assert!(!gaussian.contains(projected), "{projected}");
        }
    }

    #[test]
    fn fused_kernel_uses_the_reservoir_sun_sample_in_seamless_mode() {
        let source = fused_kernel().unwrap();
        let main_terrain = body(&source, "main_terrain");
        assert!(main_terrain.contains(
            "if (prev_valid) {\n            sun_dir = normalize(prev_r.sample.direction);"
        ));
        assert!(!main_terrain.contains("camera_flags == 0u && prev_valid"));
        assert!(hybrid_kernel().contains("camera_flags == 0u && prev_valid"));
    }

    #[test]
    fn fused_kernel_overrides_the_terrain_shading_normal() {
        let source = fused_kernel().unwrap();
        assert_eq!(source.matches("fn terrain_normal_at_base(").count(), 1);
        assert_eq!(source.matches("fn terrain_normal_at(").count(), 1);
        let fused = body(&source, "terrain_normal_at");
        assert!(fused.contains("terrain_normal_at_base(p, cx, cz)"));
        assert!(fused.contains("FUSION_FLAG_SMOOTH_TERRAIN"));
        let base = hybrid_kernel();
        assert!(base.contains("fn terrain_normal_at("));
        assert!(!base.contains("terrain_normal_at_base"));
    }

    #[test]
    fn a_moved_anchor_is_a_loud_assembly_error() {
        let error = replace_once("abc", "x", "y", "probe").unwrap_err();
        assert!(format!("{error}").contains("matched 0 times"));
        let error = replace_once("xx", "x", "y", "probe").unwrap_err();
        assert!(format!("{error}").contains("matched 2 times"));
        assert!(extract_block("abc", "x", "c", "probe").is_err());
        assert_eq!(extract_block("abc", "b", "c", "probe").unwrap(), "bc");
    }
}

/// Drift gate for the four WGSL copies of `ShadowCascade` + `CsmUniforms`.
///
/// They cannot be collapsed into one definition yet: the assembled terrain
/// source is hash-pinned (`PINNED_TERRAIN_SOURCE_HASH`), so injecting a shared
/// prelude changes that hash and needs owner approval. Until then this test
/// fails if any copy's fields (name, type, order) diverge, which is the
/// back-and-forth this lock exists to stop.
#[cfg(test)]
mod csm_single_source_tests {
    const COPIES: &[(&str, &str)] = &[
        ("shadows.wgsl", include_str!("shaders/shadows.wgsl")),
        (
            "mesh_instanced.wgsl",
            include_str!("shaders/mesh_instanced.wgsl"),
        ),
        (
            "terrain_pbr_pom.wgsl",
            include_str!("shaders/terrain_pbr_pom.wgsl"),
        ),
        (
            "terrain_pbr.wgsl",
            include_str!("viewer/terrain/shader_pbr/terrain_pbr.wgsl"),
        ),
    ];

    fn struct_fields(source: &str, name: &str) -> Vec<String> {
        let marker = format!("struct {name}");
        let start = source
            .find(&marker)
            .unwrap_or_else(|| panic!("struct {name} not found"));
        let open = source[start..].find('{').expect("open brace") + start;
        let close = source[open..].find('}').expect("close brace") + open;
        source[open + 1..close]
            .lines()
            .map(|line| line.split("//").next().unwrap_or("").trim())
            .filter(|line| !line.is_empty())
            .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
            .collect()
    }

    #[test]
    fn csm_uniforms_wgsl_copies_are_identical() {
        for name in ["ShadowCascade", "CsmUniforms"] {
            let reference = struct_fields(COPIES[0].1, name);
            assert!(!reference.is_empty(), "{name} has no fields");
            for (file, text) in COPIES {
                assert_eq!(
                    struct_fields(text, name),
                    reference,
                    "{name} drifted in {file}; the WGSL copies must stay identical",
                );
            }
        }
    }
}
