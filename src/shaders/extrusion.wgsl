// shaders/extrusion.wgsl
// Compute shader for GPU polygon prism extrusion
// Exists to parallelize mesh generation and mirror CPU reference output
// RELEVANT FILES: src/vector/gpu_extrusion.rs, src/vector/extrusion.rs, src/vector/api.rs, docs/api/polygon_extrusion.md

struct PolygonMeta {
    base_vertex_offset: u32,
    base_vertex_count: u32,
    base_index_offset: u32,
    base_index_count: u32,
    ring_offset: u32,
    ring_count: u32,
    output_vertex_offset: u32,
    output_index_offset: u32,
    bbox_min: vec2<f32>,
    bbox_scale: vec2<f32>,
};

struct RingVertexPacked {
    position: vec2<f32>,
    u_coord: f32,
    _pad: f32,
};

struct ExtrusionParams {
    height: f32,
};

@group(0) @binding(0) var<storage, read> metas: array<PolygonMeta>;
@group(0) @binding(1) var<storage, read> base_vertices: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> base_indices: array<u32>;
@group(0) @binding(3) var<storage, read> ring_vertices: array<RingVertexPacked>;
@group(0) @binding(4) var<storage, read_write> out_positions: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_normals: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> out_uvs: array<vec2<f32>>;
@group(0) @binding(8) var<uniform> params: ExtrusionParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let polygon_index = global_id.x;
    if (polygon_index >= arrayLength(&metas)) {
        return;
    }

    let polygon_meta = metas[polygon_index];
    let base_vertex_offset = polygon_meta.base_vertex_offset;
    let base_vertex_count = polygon_meta.base_vertex_count;
    let base_index_offset = polygon_meta.base_index_offset;
    let base_index_count = polygon_meta.base_index_count;
    let ring_offset = polygon_meta.ring_offset;
    let ring_count = polygon_meta.ring_count;

    let vertex_offset = polygon_meta.output_vertex_offset;
    let index_offset = polygon_meta.output_index_offset;

    let bottom_vertex_offset = vertex_offset;
    let top_vertex_offset = vertex_offset + base_vertex_count;
    let side_vertex_offset = top_vertex_offset + base_vertex_count;

    let bottom_index_offset = index_offset;
    let top_index_offset = index_offset + base_index_count;
    let side_index_offset = top_index_offset + base_index_count;

    let bbox_min = polygon_meta.bbox_min;
    let bbox_scale = polygon_meta.bbox_scale;
    let height = params.height;

    for (var i: u32 = 0u; i < base_vertex_count; i = i + 1u) {
        let base = base_vertices[base_vertex_offset + i];
        var uv = vec2<f32>(0.0, 0.0);
        if (bbox_scale.x != 0.0) {
            uv.x = (base.x - bbox_min.x) * bbox_scale.x;
        }
        if (bbox_scale.y != 0.0) {
            uv.y = (base.y - bbox_min.y) * bbox_scale.y;
        }

        out_positions[bottom_vertex_offset + i] = vec4<f32>(base.x, 0.0, base.y, 1.0);
        out_positions[top_vertex_offset + i] = vec4<f32>(base.x, height, base.y, 1.0);

        out_normals[bottom_vertex_offset + i] = vec4<f32>(0.0, -1.0, 0.0, 0.0);
        out_normals[top_vertex_offset + i] = vec4<f32>(0.0, 1.0, 0.0, 0.0);

        out_uvs[bottom_vertex_offset + i] = uv;
        out_uvs[top_vertex_offset + i] = uv;
    }

    for (var tri: u32 = 0u; tri < base_index_count / 3u; tri = tri + 1u) {
        let idx0 = base_indices[base_index_offset + tri * 3u];
        let idx1 = base_indices[base_index_offset + tri * 3u + 1u];
        let idx2 = base_indices[base_index_offset + tri * 3u + 2u];

        out_indices[bottom_index_offset + tri * 3u] = bottom_vertex_offset + idx0;
        out_indices[bottom_index_offset + tri * 3u + 1u] = bottom_vertex_offset + idx2;
        out_indices[bottom_index_offset + tri * 3u + 2u] = bottom_vertex_offset + idx1;

        out_indices[top_index_offset + tri * 3u] = top_vertex_offset + idx0;
        out_indices[top_index_offset + tri * 3u + 1u] = top_vertex_offset + idx1;
        out_indices[top_index_offset + tri * 3u + 2u] = top_vertex_offset + idx2;
    }

    var side_vertex_cursor = side_vertex_offset;
    var side_index_cursor = side_index_offset;

    for (var i: u32 = 0u; i < ring_count; i = i + 1u) {
        let current = ring_vertices[ring_offset + i];
        let next = ring_vertices[ring_offset + ((i + 1u) % ring_count)];

        let curr_pos = current.position;
        let next_pos = next.position;
        let edge = next_pos - curr_pos;
        let segment_len = length(edge);

        var normal = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (segment_len > 1e-6) {
            let normal2 = vec2<f32>(edge.y, -edge.x) / segment_len;
            normal = vec4<f32>(normal2.x, 0.0, normal2.y, 0.0);
        }

        var u_next = next.u_coord;
        if ((i + 1u) % ring_count == 0u || u_next < current.u_coord) {
            u_next = 1.0;
        }

        out_positions[side_vertex_cursor] = vec4<f32>(curr_pos.x, 0.0, curr_pos.y, 1.0);
        out_positions[side_vertex_cursor + 1u] = vec4<f32>(next_pos.x, 0.0, next_pos.y, 1.0);
        out_positions[side_vertex_cursor + 2u] = vec4<f32>(curr_pos.x, height, curr_pos.y, 1.0);
        out_positions[side_vertex_cursor + 3u] = vec4<f32>(next_pos.x, height, next_pos.y, 1.0);

        out_normals[side_vertex_cursor] = normal;
        out_normals[side_vertex_cursor + 1u] = normal;
        out_normals[side_vertex_cursor + 2u] = normal;
        out_normals[side_vertex_cursor + 3u] = normal;

        out_uvs[side_vertex_cursor] = vec2<f32>(current.u_coord, 0.0);
        out_uvs[side_vertex_cursor + 1u] = vec2<f32>(u_next, 0.0);
        out_uvs[side_vertex_cursor + 2u] = vec2<f32>(current.u_coord, 1.0);
        out_uvs[side_vertex_cursor + 3u] = vec2<f32>(u_next, 1.0);

        out_indices[side_index_cursor] = side_vertex_cursor;
        out_indices[side_index_cursor + 1u] = side_vertex_cursor + 2u;
        out_indices[side_index_cursor + 2u] = side_vertex_cursor + 1u;
        out_indices[side_index_cursor + 3u] = side_vertex_cursor + 2u;
        out_indices[side_index_cursor + 4u] = side_vertex_cursor + 3u;
        out_indices[side_index_cursor + 5u] = side_vertex_cursor + 1u;

        side_vertex_cursor = side_vertex_cursor + 4u;
        side_index_cursor = side_index_cursor + 6u;
    }
}
