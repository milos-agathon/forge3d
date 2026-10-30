//! CONCORDIA specializes the authoritative forward shaders. The ordinary
//! terrain assembly and its pinned determinism hash remain byte-identical.
use anyhow::{bail, Result};

fn replace_once(source: String, before: &str, after: &str) -> Result<String> {
    let count = source.matches(before).count();
    if count != 1 {
        bail!("CONCORDIA shader contract: expected one {before:?}, found {count}");
    }
    Ok(source.replacen(before, after, 1))
}

pub(crate) fn terrain_source(bindless: bool) -> Result<String> {
    let mut source = if bindless {
        crate::shader_sources::terrain_bindless()
    } else {
        crate::shader_sources::terrain()
    };
    source = replace_once(
        source,
        "    @location(4) source_id : u32,",
        concat!(
            "    @location(4) source_id : u32,\n",
            "    @location(6) concordia_normal_roughness : vec4<f32>,\n",
            "    @location(7) concordia_metallic : f32,"
        ),
    )?;
    source = replace_once(
        source,
        "    out.aov_albedo = vec4<f32>(albedo, 1.0);",
        concat!(
            "    out.aov_albedo = vec4<f32>(albedo, 1.0);\n",
            "    out.concordia_normal_roughness = vec4<f32>(select(n_dd, shading_normal, is_water), roughness);\n",
            "    out.concordia_metallic = metallic;"
        ),
    )?;
    source = replace_once(
        source,
        "let shadow_z = h_curved * h_exag;",
        "let shadow_z = (h_curved - 0.5) * h_range * h_exag;",
    )?;
    source = replace_once(
        source,
        "shaded = apply_atmospheric_fog(",
        "shaded = concordia_defer_atmosphere(",
    )?;
    source.push_str(r#"
fn concordia_defer_atmosphere(color: vec3<f32>, world: vec3<f32>, camera: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> { return color; }
"#);
    source = replace_once(source,
        "let combined_shadow = shadow_clamped * sun_vis_clamped;",
        "let combined_shadow = shadow_clamped * sun_vis_clamped * concordia_cloud_visibility(input.world_position);",
    )?;
    source.push_str(include_str!("../../../shaders/concordia_surface.wgsl"));
    source.push_str(
        r#"
@fragment
fn fs_concordia(input: VertexOutput) -> ConcordiaSurface {
    det_seed(input.clip_position.x);
    let shaded = forward_shade_with_feedback(input);
    return concordia_surface(input.world_position,
        shaded.concordia_normal_roughness.xyz,
        shaded.concordia_normal_roughness.w,
        shaded.aov_albedo.rgb, shaded.concordia_metallic, shaded.color, input.clip_position, 1u);
}
"#,
    );
    Ok(source)
}

pub(crate) fn scatter_source() -> Result<String> {
    let source = include_str!("../../../shaders/mesh_instanced.wgsl").replace("\r\n", "\n");
    let mut source = replace_once(
        source,
        "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {",
        "fn concordia_scatter_shade(in: VsOut) -> vec4<f32> {",
    )?;
    source = replace_once(
        source,
        "let ndotl = max(dot(n, -l), 0.0);",
        "let ndotl = max(dot(n, l), 0.0);",
    )?;
    source = replace_once(source,
        "let direct_shadow = mix(0.20, 1.0, shadow_visibility);",
        "let direct_shadow = mix(0.20, 1.0, shadow_visibility) * concordia_cloud_visibility(in.world_pos);",
    )?;
    source = replace_once(
        source,
        "let shaded = lit * (1.0 - contact);",
        "let shaded = lit * bitcast<f32>(concordia_frame.effects.w);",
    )?;
    source.push_str(
        &include_str!("../../../shaders/concordia_surface.wgsl").replace("@group(7)", "@group(3)"),
    );
    source.push_str(
        r#"
@fragment
fn fs_concordia(in: VsOut) -> ConcordiaSurface {
    let shaded = concordia_scatter_shade(in);
    // The scalar scatter material is an opaque dielectric. Its published
    // color is linear base color; it has no authored metallic/roughness map.
    return concordia_surface(in.world_pos, in.n_ws, 1.0,
        U.color.rgb, 0.0, shaded, in.pos, 2u);
}
"#,
    );
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_scatter_coordinate_shader_validates() {
        let source = include_str!("../../../shaders/mesh_instanced.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("ordinary scatter shader validates");
    }

    #[test]
    fn concordia_terrain_and_scatter_have_the_same_geometry_outputs() {
        let terrain = terrain_source(false).expect("terrain source anchors");
        let scatter = scatter_source().expect("scatter source anchors");
        for (name, source) in [("terrain", terrain), ("scatter", scatter)] {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            let entry = module
                .entry_points
                .iter()
                .find(|entry| entry.name == "fs_concordia")
                .expect("shared geometry entry point");
            let result = entry.function.result.as_ref().expect("surface output");
            let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner else {
                panic!("{name} must return the shared surface struct")
            };
            assert_eq!(
                members.len(),
                6,
                "{name} has HDR, four GBuffer outputs and surface kind"
            );
            assert_eq!(members[3].name.as_deref(), Some("depth"));
            assert_eq!(members[4].name.as_deref(), Some("velocity"));
        }
    }
}

#[cfg(test)]
mod composition_tests {
    #[test]
    fn concordia_hdr_and_label_composition_modules_validate() {
        for (name, source, entries) in [
            (
                "HDR",
                include_str!("../../../shaders/concordia_compose.wgsl"),
                5,
            ),
            (
                "label fade",
                include_str!("../../../shaders/concordia_label_fade.wgsl"),
                1,
            ),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
            assert_eq!(module.entry_points.len(), entries);
        }
    }
}

#[cfg(test)]
mod cloud_contract_tests {
    #[test]
    fn concordia_existing_cloud_generator_validates() {
        let source = include_str!("../../../shaders/cloud_shadows.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{error:?}"));
        assert!(module
            .entry_points
            .iter()
            .any(|entry| entry.name == "cs_generate_cloud_shadows"));
    }
}

/// Use the existing terrain atmosphere evaluator as a full-frame HDR post.
/// Forward geometry defers this function so terrain and buildings share it.
pub(crate) fn atmosphere_source() -> Result<String> {
    let mut source = crate::shader_sources::terrain();
    source = replace_once(
        source,
        "fog_uniforms.sky_params0.x < 0.5 || water_reflection_uniforms.enable_flags.y > 0.5",
        "fog_uniforms.sky_params0.x < 0.5",
    )?;
    source.push_str(include_str!("../../../shaders/concordia_surface.wgsl"));
    source.push_str(
        r#"
@group(7) @binding(3) var concordia_atmosphere_input: texture_2d<f32>;
@group(7) @binding(4) var concordia_atmosphere_depth: texture_2d<f32>;
@group(7) @binding(5) var concordia_atmosphere_output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_concordia_atmosphere(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(concordia_atmosphere_input);
    if (any(gid.xy >= dims)) { return; }
    det_seed(f32(gid.x)+0.5);
    let color = textureLoad(concordia_atmosphere_input, gid.xy, 0);
    let depth = textureLoad(concordia_atmosphere_depth, gid.xy, 0).r;
    if (depth <= 0.0) {
        textureStore(concordia_atmosphere_output, gid.xy, color);
        return;
    }
    let uv = (vec2<f32>(gid.xy)+0.5) / vec2<f32>(dims);
    let ndc = uv*vec2<f32>(2.0,-2.0)+vec2<f32>(-1.0,1.0);
    let near_h = concordia_frame.inv_proj*vec4<f32>(ndc,0.0,1.0);
    let far_h = concordia_frame.inv_proj*vec4<f32>(ndc,1.0,1.0);
    let near = near_h.xyz/near_h.w;
    let far = far_h.xyz/far_h.w;
    let view = mix(near,far,(depth+near.z)/(near.z-far.z));
    let world = (concordia_frame.inv_view*vec4<f32>(view,1.0)).xyz;
    let eye = concordia_frame.inv_view[3].xyz;
    let pixel = vec2<f32>(gid.xy)+0.5;
    let transported = apply_atmospheric_fog(color.rgb, world, eye, pixel);
    let inscatter = apply_atmospheric_fog(vec3<f32>(0.0), world, eye, pixel);
    let cloud = concordia_cloud_visibility(world);
    textureStore(concordia_atmosphere_output, gid.xy,
        vec4<f32>(max(transported-inscatter+inscatter*cloud,vec3<f32>(0.0)),color.a));
}
"#,
    );
    Ok(source)
}

#[cfg(test)]
mod shared_depth_tests {
    #[test]
    fn concordia_shared_depth_and_atmosphere_bindings_validate() {
        let modules = [
            (
                "authoritative height horizon",
                include_str!("../../../shaders/heightfield_ao.wgsl").to_owned(),
            ),
            (
                "partitioned GTAO",
                crate::core::screen_space_effects::SsaoRenderer::shared_contacts_source()
                    .expect("contact GTAO anchors"),
            ),
            (
                "authoritative GTAO",
                crate::core::screen_space_effects::GTAO_SHADER_SRC.to_owned(),
            ),
            (
                "linear HZB",
                crate::core::screen_space_effects::hzb::linear_depth_source(),
            ),
            (
                "cartography",
                crate::core::overlays::shared_depth_source().expect("overlay anchors"),
            ),
            (
                "atmosphere",
                super::atmosphere_source().expect("atmosphere anchors"),
            ),
            ("froxels", super::froxel_source().expect("froxel anchors")),
            (
                "transparent points",
                crate::vector::PointRenderer::shared_depth_source().expect("point anchors"),
            ),
        ];
        for (name, source) in modules {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        }
    }
}

pub(crate) fn froxel_source() -> Result<String> {
    let mut source = include_str!("../../../shaders/volumetric.wgsl").replace("\r\n", "\n");
    source = replace_once(
        source,
        "    let height = world_pos.y;",
        "    let height = world_pos.z - bitcast<f32>(params._pad0);",
    )?;
    source = replace_once(source, "    let view_ray = normalize(vec3<f32>(froxel_ndc.xy, 1.0));",
        "    let uv = vec2<f32>((froxel_ndc.x+1.0)*0.5,(froxel_ndc.y+1.0)*0.5);\n    let view_ray = normalize(reconstruct_world_pos(uv,1.0)-camera.eye_position);")?;
    source = source.replace(
        "sample_shadow(world_pos);",
        "sample_shadow(world_pos) * concordia_cloud_visibility(world_pos);",
    );
    source = replace_once(
        source,
        "    let depth = light_ndc.z * 0.5 + 0.5;",
        "    let depth = light_ndc.z;",
    )?;
    source = replace_once(
        source,
        "    let uv = light_ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);",
        "    let uv = light_ndc.xy * vec2<f32>(0.5,-0.5) + vec2<f32>(0.5, 0.5);",
    )?;
    let begin = source
        .find("fn reconstruct_world_pos(")
        .ok_or_else(|| anyhow::anyhow!("fog projection anchor missing"))?;
    let end = source[begin..]
        .find("\n}\n")
        .map(|offset| begin + offset + 3)
        .ok_or_else(|| anyhow::anyhow!("fog projection end missing"))?;
    source.replace_range(
        begin..end,
        r#"fn reconstruct_world_pos(uv: vec2<f32>, linear: f32) -> vec3<f32> {
    let ndc = uv*vec2<f32>(2.0,-2.0)+vec2<f32>(-1.0,1.0);
    let a_h = camera.inv_proj*vec4<f32>(ndc,0.0,1.0);
    let b_h = camera.inv_proj*vec4<f32>(ndc,1.0,1.0);
    let a = a_h.xyz/a_h.w;
    let b = b_h.xyz/b_h.w;
    let depth = select(linear,camera.far,linear <= 0.0);
    let view = mix(a,b,(depth+a.z)/(a.z-b.z));
    return (camera.inv_view*vec4<f32>(view,1.0)).xyz;
}"#,
    );
    source = source.replace("textureSampleLevel(depth_texture, depth_sampler, uv, 0.0).r",
        "textureLoad(depth_texture, min(vec2<u32>(uv*vec2<f32>(textureDimensions(depth_texture))),textureDimensions(depth_texture)-1u),0).r");
    source.push_str(
        &include_str!("../../../shaders/concordia_surface.wgsl").replace("@group(7)", "@group(4)"),
    );
    Ok(source)
}

/// Specialize the authoritative horizon kernel only for shared-frame AO.
pub(crate) fn height_ao_source() -> Result<String> {
    let mut source = include_str!("../../../shaders/heightfield_ao.wgsl").replace("\r\n", "\n");
    source = replace_once(
        source,
        r###"    // Use textureLoad since R32Float doesn't support filtering
    let tex_size = vec2<f32>(u_ao.params2.z, u_ao.params2.w);
    let pixel = vec2<i32>(uv * tex_size);
    let clamped_pixel = clamp(pixel, vec2<i32>(0), vec2<i32>(tex_size) - vec2<i32>(1));
    let h = textureLoad(height_tex, clamped_pixel, 0).r;
    // Height textures store raw elevation values. Convert them to the same
    // scene-space Y used by the terrain render path: (h - height_min) * z_scale.
    return (h - u_ao.params1.w) * u_ao.params1.z;
"###,
        r###"    // R32Float is unfilterable. Match the continuous DEM surface explicitly
    // instead of introducing nearest-texel steps into its horizon.
    let dims = vec2<i32>(textureDimensions(height_tex));
    let coordinate = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * vec2<f32>(dims-vec2<i32>(1));
    let base = vec2<i32>(floor(coordinate));
    let next = min(base+vec2<i32>(1), dims-vec2<i32>(1));
    let t = fract(coordinate);
    let a = mix(textureLoad(height_tex,base,0).r,textureLoad(height_tex,vec2<i32>(next.x,base.y),0).r,t.x);
    let b = mix(textureLoad(height_tex,vec2<i32>(base.x,next.y),0).r,textureLoad(height_tex,next,0).r,t.x);
    return (mix(a,b,t.y)-u_ao.params1.w)*u_ao.params1.z;
"###,
    )?;
    source = replace_once(
        source,
        r###"    let max_distance = u_ao.params0.z;
    let spacing = vec2<f32>(u_ao.params1.x, u_ao.params1.y);
    
    // Convert max_distance from world units to UV space
    let height_tex_size = vec2<f32>(u_ao.params2.z, u_ao.params2.w);
    let world_extent = spacing * height_tex_size;
    let max_uv_distance = max_distance / max(world_extent.x, world_extent.y);
    let step_uv = max_uv_distance / f32(steps);
    
    var total_occlusion: f32 = 0.0;
    
    // March in multiple directions around the horizon
    for (var d: u32 = 0u; d < directions; d = d + 1u) {
        let angle = (f32(d) / f32(directions)) * 2.0 * PI;
        let dir = vec2<f32>(cos(angle), sin(angle));
        
        var max_tan_angle: f32 = -999.0;
        
        // March along this direction
        for (var s: u32 = 1u; s <= steps; s = s + 1u) {
            let sample_uv = center_uv + dir * step_uv * f32(s);
            
            // Skip if outside [0,1] bounds
            if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                break;
            }
            
            let sample_height = sample_height(sample_uv);
            let height_diff = sample_height - center_height;
            let horizontal_dist = length((sample_uv - center_uv) * world_extent);
            
            // Compute tangent of elevation angle
            if (horizontal_dist > 0.001) {
                let tan_angle = height_diff / horizontal_dist;
                max_tan_angle = max(max_tan_angle, tan_angle);
            }
"###,
        r###"    let world_extent = u_ao.params1.xy*u_ao.params2.zw;
    let texel = vec2<f32>(1.0)/max(u_ao.params2.zw-vec2<f32>(1.0),vec2<f32>(1.0));
    let low_uv=max(center_uv-texel,vec2<f32>(0.0));
    let high_uv=min(center_uv+texel,vec2<f32>(1.0));
    let dx=(sample_height(vec2<f32>(high_uv.x,center_uv.y))-sample_height(vec2<f32>(low_uv.x,center_uv.y)))
        /((high_uv.x-low_uv.x)*world_extent.x);
    let dy=(sample_height(vec2<f32>(center_uv.x,high_uv.y))-sample_height(vec2<f32>(center_uv.x,low_uv.y)))
        /((high_uv.y-low_uv.y)*world_extent.y);
    let normal = normalize(vec3<f32>(-dx,-dy,1.0));
    var visibility = 0.0;
    for (var d=0u;d<directions;d++) {
        let angle=f32(d)/f32(directions)*2.0*PI;
        let direction=vec2<f32>(cos(angle),sin(angle));
        var horizon=-PI*0.5;
        for (var step=1u;step<=steps;step++) {
            let distance=u_ao.params0.z*f32(step)/f32(steps);
            let sample_uv=center_uv+direction*distance/world_extent;
            if (any(sample_uv<vec2<f32>(0.0)) || any(sample_uv>vec2<f32>(1.0))) { break; }
            horizon=max(horizon,atan2(sample_height(sample_uv)-center_height,distance));
"###,
    )?;
    source = replace_once(
        source,
        r###"        
        // Convert max tangent angle to occlusion factor
        // Positive tan_angle means terrain is above us → occlusion
        // Use atan to get angle, then map to [0,1]
        if (max_tan_angle > -999.0) {
            let horizon_angle = atan(max_tan_angle);
            // Normalize: 0 at horizon (angle=0), 1 at 90° above (angle=PI/2)
            let occlusion = clamp(horizon_angle / (PI * 0.5), 0.0, 1.0);
            total_occlusion = total_occlusion + occlusion;
        }
"###,
        r###"        // Integrate max(dot(n,w),0) dOmega above the height horizon. For
        // elevation e, n.w=n.z*sin(e)+q*cos(e), dOmega=cos(e) de dAzimuth.
        // Clipping to the normal hemisphere makes an unobstructed tilted
        // plane exactly the identity, rather than mistaking its slope for AO.
        let q=dot(normal.xy,direction);
        let lower=max(horizon,atan2(-q,normal.z));
        let integral=0.5*normal.z*(1.0-sin(lower)*sin(lower))
            +q*(PI*0.25-0.5*lower-0.25*sin(2.0*lower));
        visibility+=2.0*integral/f32(directions);
"###,
    )?;
    source = replace_once(
        source,
        r###"    
    // Average occlusion across all directions
    let avg_occlusion = total_occlusion / f32(directions);
    
    // Return visibility (1 - occlusion), so 1.0 = fully visible, 0.0 = fully occluded
    return 1.0 - avg_occlusion;
"###,
        r###"    return clamp(visibility,0.0,1.0);
"###,
    )?;
    source = replace_once(
        source,
        r###"    let final_ao = mix(1.0, ao, strength);
"###,
        r###"    let final_ao = clamp(mix(1.0, ao, strength),0.0,1.0);
"###,
    )?;
    Ok(source)
}

#[cfg(test)]
mod specialized_effect_tests {
    #[test]
    fn concordia_specialized_effect_sources_validate() {
        use crate::core::screen_space_effects::{concordia_reconstruction, SsaoRenderer};
        let mut sources = vec![
            ("height AO", super::height_ao_source().unwrap()),
            ("GTAO", SsaoRenderer::concordia_gtao_source().unwrap()),
        ];
        for (name, source, camera) in [
            (
                "SSGI",
                include_str!("../../../shaders/ssgi/trace.wgsl"),
                "uCam",
            ),
            (
                "SSR trace",
                include_str!("../../../shaders/ssr/trace.wgsl"),
                "camera",
            ),
            (
                "SSR shade",
                include_str!("../../../shaders/ssr/shade.wgsl"),
                "camera",
            ),
            (
                "SSR fallback",
                include_str!("../../../shaders/ssr/fallback_env.wgsl"),
                "camera",
            ),
        ] {
            sources.push((
                name,
                concordia_reconstruction(source, camera, "reconstruct_view_position").unwrap(),
            ));
        }
        for (name, source) in sources {
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        }
    }
}
