// TESSELLA visibility-only shader extension. Runtime assembly appends this to
// the proven terrain shader and upgrades the visibility ID packing to preserve
// 16 bits each for tile/LOD identity and primitive identity.

@group(7) @binding(0)
var terrain_visibility_ids: texture_2d<u32>;

@group(7) @binding(1)
var terrain_visibility_depth: texture_depth_2d;

struct VisibilityClipmapVertex {
    position: vec2<f32>,
    uv: vec2<f32>,
    morph_data: vec2<f32>,
}

struct VisibilityDrawTemplate {
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    tile_id: u32,
}

struct VisibilityResolveMeta {
    variant_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(7) @binding(2)
var<storage, read> terrain_visibility_vertices: array<VisibilityClipmapVertex>;

@group(7) @binding(3)
var<storage, read> terrain_visibility_indices: array<u32>;

@group(7) @binding(4)
var<storage, read> terrain_visibility_templates: array<VisibilityDrawTemplate>;

@group(7) @binding(5)
var<uniform> terrain_visibility_meta: VisibilityResolveMeta;

struct VisibilityFullscreenOutput {
    @builtin(position) clip_position: vec4<f32>,
}

struct VisibilityReconstructedVertex {
    clip: vec4<f32>,
    world: vec3<f32>,
    uv: vec2<f32>,
}

fn visibility_reconstruct_vertex(index: u32) -> VisibilityReconstructedVertex {
    let source = terrain_visibility_vertices[index];
    let uv = clamp(source.uv, vec2<f32>(0.0), vec2<f32>(1.0));
    let h_raw = textureSampleLevel(height_tex, height_samp, uv, 0.0).r;
    let t_geom = get_height_geom_t(h_raw);
    let h_min = u_shading.clamp0.x;
    let h_max = u_shading.clamp0.y;
    let h_disp = det_fma(apply_height_curve01(t_geom), h_max - h_min, h_min);
    let h_exag = u_terrain.spacing_h_exag.z;
    let h_center = (h_min + h_max) * 0.5;
    let skirt_offset = select(
        0.0,
        u_terrain.camera_mode_params.y * 0.001,
        source.morph_data.x < 0.0,
    );
    let centered = vec3<f32>(
        source.position,
        (h_disp - h_center - skirt_offset) * h_exag,
    );
    var out: VisibilityReconstructedVertex;
    out.clip = det_mat4_mul_vec4(
        u_terrain.proj,
        det_mat4_mul_vec4(
            u_terrain.view,
            vec4<f32>(centered, 1.0),
        ),
    );
    out.world = vec3<f32>(
        source.position,
        (h_disp - skirt_offset) * h_exag,
    );
    out.uv = uv;
    return out;
}

fn visibility_barycentrics(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let denominator = v0.x * v1.y - v1.x * v0.y;
    let inverse_denominator = 1.0 / select(
        1e-8,
        denominator,
        abs(denominator) > 1e-8,
    );
    let y = (v2.x * v1.y - v1.x * v2.y) * inverse_denominator;
    let z = (v0.x * v2.y - v2.x * v0.y) * inverse_denominator;
    return vec3<f32>(1.0 - y - z, y, z);
}

@vertex
fn vs_visibility_fullscreen(
    @builtin(vertex_index) vertex_index: u32,
) -> VisibilityFullscreenOutput {
    var out: VisibilityFullscreenOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.clip_position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_visibility_resolve_fullscreen(
    input: VisibilityFullscreenOutput,
) -> FragmentOutput {
    let pixel = vec2<i32>(input.clip_position.xy);
    let encoded = textureLoad(terrain_visibility_ids, pixel, 0).x;
    if (encoded == 0u) {
        discard;
    }
    let depth = textureLoad(terrain_visibility_depth, pixel, 0);
    if (depth >= 1.0) {
        discard;
    }
    let dimensions = vec2<f32>(textureDimensions(terrain_visibility_ids));
    let uv_screen = (input.clip_position.xy + vec2<f32>(0.5)) / dimensions;
    let ndc_xy = vec2<f32>(
        uv_screen.x * 2.0 - 1.0,
        1.0 - uv_screen.y * 2.0,
    );
    let payload = encoded - 1u;
    let tile_lod_id = payload >> 16u;
    let primitive = payload & 0xffffu;
    let selected_lod = (tile_lod_id >> 12u) & 0xfu;
    let tile_index = tile_lod_id & 0xfffu;
    let draw_template = terrain_visibility_templates[
        tile_index * terrain_visibility_meta.variant_count + selected_lod
    ];
    let first = draw_template.first_index + primitive * 3u;
    let v0 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first]) + draw_template.base_vertex),
    );
    let v1 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first + 1u]) + draw_template.base_vertex),
    );
    let v2 = visibility_reconstruct_vertex(
        u32(i32(terrain_visibility_indices[first + 2u]) + draw_template.base_vertex),
    );
    let bary_screen = visibility_barycentrics(
        ndc_xy,
        v0.clip.xy / v0.clip.w,
        v1.clip.xy / v1.clip.w,
        v2.clip.xy / v2.clip.w,
    );
    let perspective = bary_screen / vec3<f32>(v0.clip.w, v1.clip.w, v2.clip.w);
    let bary = perspective / max(
        perspective.x + perspective.y + perspective.z,
        1e-8,
    );
    var surface: VertexOutput;
    surface.clip_position = input.clip_position;
    surface.world_position =
        v0.world * bary.x + v1.world * bary.y + v2.world * bary.z;
    surface.world_normal = vec3<f32>(0.0, 0.0, 1.0);
    surface.tex_coord = v0.uv * bary.x + v1.uv * bary.y + v2.uv * bary.z;
    surface.tile_id = tile_lod_id;
    let out = shade_main(surface);
    atomicAdd(&terrain_frame_counters.material_invocations, 1u);
    terrain_vt_write_surface_feedback(surface.tex_coord, 0u);
    if (terrain_vt_uniforms.config2.w != 0u) {
        if (terrain_vt_enabled()
            && terrain_vt_family_enabled(TERRAIN_VT_FAMILY_ALBEDO)
            && out.source_id == 0u) {
            atomicAdd(&terrain_frame_counters.fallback_texels, 1u);
        }
    }
    return out;
}
