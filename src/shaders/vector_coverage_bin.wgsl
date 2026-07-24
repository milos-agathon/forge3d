// LIMES 16x16 analytic-coverage tile binning.
//
// One invocation owns one stable primitive record. Atomic insertion order is
// deliberately NOT a numerical contract; vector_coverage_raster.wgsl restores
// stable primitive-ID order before it accumulates coverage.

struct PrimitiveRecord {
    geometry: vec4<f32>,
    bounds: vec4<f32>,
    metadata: vec4<u32>,
}

struct BinParams {
    extent_tiles: vec4<u32>,
    layers_capacity: vec4<u32>,
}

struct AtomicWords {
    values: array<atomic<u32>>,
}

struct Words {
    values: array<u32>,
}

@group(0) @binding(0) var<storage, read> primitives: array<PrimitiveRecord>;
@group(0) @binding(1) var<uniform> params: BinParams;
@group(0) @binding(2) var<storage, read_write> tile_counts: AtomicWords;
@group(0) @binding(3) var<storage, read_write> tile_indices: Words;
@group(0) @binding(4) var<storage, read_write> overflow: AtomicWords;

fn clamped_tile(value: f32, tile_extent: u32) -> u32 {
    let pixel = u32(max(value, 0.0));
    return min(pixel / params.layers_capacity.w, tile_extent - 1u);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let primitive_index = gid.x;
    let primitive_count = params.layers_capacity.z;
    if primitive_index >= primitive_count {
        return;
    }

    let primitive = primitives[primitive_index];
    let width = f32(params.extent_tiles.x);
    let height = f32(params.extent_tiles.y);
    if primitive.bounds.z < 0.0 || primitive.bounds.w < 0.0 ||
       primitive.bounds.x >= width || primitive.bounds.y >= height {
        return;
    }

    let tile_columns = params.extent_tiles.z;
    let tile_rows = params.extent_tiles.w;
    let tile_count = tile_columns * tile_rows;
    let layer = primitive.metadata.y;
    let capacity = params.layers_capacity.y;
    let tx0 = clamped_tile(primitive.bounds.x, tile_columns);
    let ty0 = clamped_tile(primitive.bounds.y, tile_rows);
    let tx1 = clamped_tile(primitive.bounds.z, tile_columns);
    let ty1 = clamped_tile(primitive.bounds.w, tile_rows);

    var ty = ty0;
    loop {
        var tx = tx0;
        loop {
            let tile = layer * tile_count + ty * tile_columns + tx;
            let slot = atomicAdd(&tile_counts.values[tile], 1u);
            if slot < capacity {
                tile_indices.values[tile * capacity + slot] = primitive_index;
            } else {
                // The CPU measures the exact bound before allocating. Any hit
                // here is a contract mismatch and must become a structured
                // error after readback; never silently truncate.
                atomicStore(&overflow.values[0], 1u);
            }
            if tx == tx1 {
                break;
            }
            tx += 1u;
        }
        if ty == ty1 {
            break;
        }
        ty += 1u;
    }
}
