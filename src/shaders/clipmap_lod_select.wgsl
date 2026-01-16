// P2.3: GPU LOD selection compute shader with frustum culling.
//
// This compute shader performs per-tile frustum culling and LOD selection
// on the GPU, outputting a compact list of visible tiles with optimal LOD levels.
//
// Workgroup size: 64 threads (8x8 tile grid per workgroup)

struct LodSelectParams {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    frustum_planes: array<vec4<f32>, 6>,  // left, right, bottom, top, near, far
    lod_params: vec4<f32>,    // x=pixel_error_budget, y=viewport_height, z=fov_y, w=max_lod
    terrain_params: vec4<f32>, // x=terrain_width, y=tile_size, z=num_tiles_x, w=num_tiles_y
}

struct TileInfo {
    tile_id: u32,       // packed: lod(8) | x(12) | y(12)
    bounds_min: vec2<f32>,
    bounds_max: vec2<f32>,
    distance: f32,
    selected_lod: u32,
    visible: u32,       // 0 = culled, 1 = visible
    _pad: u32,
}

struct OutputHeader {
    visible_count: atomic<u32>,
    total_triangles: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: LodSelectParams;
@group(0) @binding(1) var<storage, read> input_tiles: array<TileInfo>;
@group(0) @binding(2) var<storage, read_write> output_tiles: array<TileInfo>;
@group(0) @binding(3) var<storage, read_write> output_header: OutputHeader;

// Pack tile ID from components
fn pack_tile_id(lod: u32, x: u32, y: u32) -> u32 {
    return (lod << 24u) | ((x & 0xFFFu) << 12u) | (y & 0xFFFu);
}

// Unpack tile ID to components
fn unpack_tile_id(packed: u32) -> vec3<u32> {
    let lod = packed >> 24u;
    let x = (packed >> 12u) & 0xFFFu;
    let y = packed & 0xFFFu;
    return vec3<u32>(lod, x, y);
}

// Test if a point is inside a plane (positive half-space)
fn point_in_plane(point: vec3<f32>, plane: vec4<f32>) -> bool {
    return dot(plane.xyz, point) + plane.w >= 0.0;
}

// Test if an AABB is visible against frustum planes
fn frustum_cull_aabb(bounds_min: vec2<f32>, bounds_max: vec2<f32>, height_min: f32, height_max: f32) -> bool {
    // Test each frustum plane
    for (var i = 0u; i < 6u; i++) {
        let plane = params.frustum_planes[i];
        
        // Find the positive vertex (furthest along plane normal)
        var p_vertex = vec3<f32>(bounds_min.x, height_min, bounds_min.y);
        if plane.x >= 0.0 { p_vertex.x = bounds_max.x; }
        if plane.y >= 0.0 { p_vertex.y = height_max; }
        if plane.z >= 0.0 { p_vertex.z = bounds_max.y; }
        
        // If positive vertex is outside plane, AABB is culled
        if !point_in_plane(p_vertex, plane) {
            return false;
        }
    }
    return true;
}

// Calculate screen-space error for LOD selection
fn calculate_screen_space_error(distance: f32, tile_size: f32, lod: u32) -> f32 {
    let pixel_error_budget = params.lod_params.x;
    let viewport_height = params.lod_params.y;
    let fov_y = params.lod_params.z;
    
    // Avoid division by very small distances
    let safe_distance = max(distance, 0.1);
    
    // Calculate projected size
    let half_fov = fov_y * 0.5;
    let pixels_per_unit = (viewport_height * 0.5) / (safe_distance * tan(half_fov));
    let projected_size = tile_size * pixels_per_unit;
    
    // Error increases with LOD level (coarser = more error)
    let lod_scale = 1.0 / f32(1u << lod);
    let error = projected_size * lod_scale;
    
    return error;
}

// Select optimal LOD for a tile based on distance
fn select_lod(distance: f32, tile_size: f32) -> u32 {
    let max_lod = u32(params.lod_params.w);
    let pixel_error_budget = params.lod_params.x;
    
    // Start from highest detail (LOD 0) and find first that fits budget
    for (var lod = 0u; lod <= max_lod; lod++) {
        let error = calculate_screen_space_error(distance, tile_size, lod);
        if error <= pixel_error_budget {
            return lod;
        }
    }
    
    return max_lod;
}

// Calculate triangle count for a tile at given LOD
fn tile_triangle_count(base_triangles: u32, lod: u32) -> u32 {
    // Triangle count reduces by 4x per LOD level
    let reduction = 1u << (lod * 2u);
    return base_triangles / max(reduction, 1u);
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_index = global_id.x;
    let num_tiles = u32(params.terrain_params.z * params.terrain_params.w);
    
    if tile_index >= num_tiles {
        return;
    }
    
    var tile = input_tiles[tile_index];
    
    // Calculate tile center and distance to camera
    let tile_center = (tile.bounds_min + tile.bounds_max) * 0.5;
    let camera_pos_2d = params.camera_pos.xz;
    let distance = length(tile_center - camera_pos_2d);
    tile.distance = distance;
    
    // Frustum culling (assume flat terrain for now, use 0-1000m height range)
    let visible = frustum_cull_aabb(tile.bounds_min, tile.bounds_max, 0.0, 1000.0);
    tile.visible = select(0u, 1u, visible);
    
    if visible {
        // Select optimal LOD
        let tile_size = params.terrain_params.y;
        let selected_lod = select_lod(distance, tile_size);
        tile.selected_lod = selected_lod;
        
        // Append to output using atomic counter
        let output_idx = atomicAdd(&output_header.visible_count, 1u);
        output_tiles[output_idx] = tile;
        
        // Accumulate triangle count
        let base_triangles = 128u * 128u * 2u;  // Assuming 128x128 tile resolution
        let tri_count = tile_triangle_count(base_triangles, selected_lod);
        atomicAdd(&output_header.total_triangles, tri_count);
    }
}

// Secondary pass: sort visible tiles by distance (optional, for front-to-back rendering)
@compute @workgroup_size(64, 1, 1)
fn cs_sort(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Simple bubble sort pass - would be replaced with parallel sort for production
    let idx = global_id.x;
    let count = atomicLoad(&output_header.visible_count);
    
    if idx >= count - 1u {
        return;
    }
    
    // Compare adjacent tiles and swap if needed (distance-based)
    let tile_a = output_tiles[idx];
    let tile_b = output_tiles[idx + 1u];
    
    if tile_a.distance > tile_b.distance {
        output_tiles[idx] = tile_b;
        output_tiles[idx + 1u] = tile_a;
    }
}
