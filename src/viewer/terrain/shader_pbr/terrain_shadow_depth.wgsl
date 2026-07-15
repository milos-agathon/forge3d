// Viewer-specific depth-only terrain rendering for CSM shadow passes.
// Unlike the shared offscreen terrain shader, the interactive viewer supports
// an anchor-relative origin and a non-square physical geospatial span.

// Size: 128 bytes - must match viewer::terrain::render::ShadowPassUniforms.
struct ShadowPassUniforms {
    light_view_proj: mat4x4<f32>,
    render_origin_span: vec4<f32>,
    terrain_params: vec4<f32>,
    grid_params: vec4<f32>,
    height_curve: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> u_shadow: ShadowPassUniforms;

@group(0) @binding(1)
var height_tex: texture_2d<f32>;

@group(0) @binding(2)
var height_samp: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
}

@vertex
fn vs_shadow(@builtin(vertex_index) vertex_id: u32) -> VertexOutput {
    var out: VertexOutput;

    let height_min = u_shadow.terrain_params.x;
    let z_scale = u_shadow.terrain_params.w;
    let grid_res = u32(u_shadow.grid_params.x);
    let quads_per_row = grid_res - 1u;

    let triangle_idx = vertex_id / 3u;
    let vertex_in_tri = vertex_id % 3u;
    let quad_idx = triangle_idx / 2u;
    let tri_in_quad = triangle_idx % 2u;
    let quad_x = quad_idx % quads_per_row;
    let quad_y = quad_idx / quads_per_row;

    var dx: u32;
    var dy: u32;
    if (tri_in_quad == 0u) {
        if (vertex_in_tri == 0u) { dx = 0u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 0u; }
        else { dx = 0u; dy = 1u; }
    } else {
        if (vertex_in_tri == 0u) { dx = 1u; dy = 0u; }
        else if (vertex_in_tri == 1u) { dx = 1u; dy = 1u; }
        else { dx = 0u; dy = 1u; }
    }

    let uv = vec2<f32>(
        f32(quad_x + dx) / f32(grid_res - 1u),
        f32(quad_y + dy) / f32(grid_res - 1u)
    );
    let tex_dims = textureDimensions(height_tex, 0);
    let texel = vec2<i32>(uv * vec2<f32>(tex_dims));
    let texel_clamped = clamp(texel, vec2<i32>(0), vec2<i32>(tex_dims) - vec2<i32>(1));
    let h_raw = textureLoad(height_tex, texel_clamped, 0).r;

    let world_pos = vec3<f32>(
        u_shadow.render_origin_span.x + uv.x * u_shadow.render_origin_span.z,
        (h_raw - height_min) * z_scale,
        u_shadow.render_origin_span.y + uv.y * u_shadow.render_origin_span.w
    );
    out.clip_position = u_shadow.light_view_proj * vec4<f32>(world_pos, 1.0);
    return out;
}

@fragment
fn fs_shadow() {
}
