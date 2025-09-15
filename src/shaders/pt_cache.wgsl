// Scene cache compute shader for path tracing acceleration
//
// Bind Groups:
// Group 0: Scene data (BVH, materials, textures)
// Group 1: Cache validation and statistics
//
// Formats:
// - BVH: R32G32B32A32_FLOAT (node data)
// - Materials: R32G32B32A32_FLOAT (material properties)
// - Cache stats: R32_UINT (counters)
//
// Workgroup size: 64 threads per workgroup

// Cache validation data
struct CacheValidation {
    scene_hash: u32,
    bvh_hash: u32,
    material_hash: u32,
    texture_hash: u32,
}

// Cache statistics
struct CacheStats {
    total_entries: u32,
    complete_entries: u32,
    hit_count: u32,
    miss_count: u32,
}

// Bind group 0: Scene data
@group(0) @binding(0) var<storage, read> bvh_nodes: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> material_data: array<vec4<f32>>;
@group(0) @binding(2) var texture_sampler: sampler;
@group(0) @binding(3) var texture_array: texture_2d_array<f32>;

// Bind group 1: Cache data
@group(1) @binding(0) var<storage, read_write> cache_validation: CacheValidation;
@group(1) @binding(1) var<storage, read_write> cache_stats: CacheStats;
@group(1) @binding(2) var<storage, read_write> cache_hashes: array<u32>;

// Hash computation for cache validation
fn compute_hash(data: ptr<storage, array<vec4<f32>>, read>, count: u32) -> u32 {
    var hash: u32 = 2166136261u; // FNV-1a offset basis

    for (var i: u32 = 0u; i < count; i++) {
        let value = (*data)[i];

        // Hash each component
        hash = hash ^ bitcast<u32>(value.x);
        hash = hash * 16777619u; // FNV-1a prime

        hash = hash ^ bitcast<u32>(value.y);
        hash = hash * 16777619u;

        hash = hash ^ bitcast<u32>(value.z);
        hash = hash * 16777619u;

        hash = hash ^ bitcast<u32>(value.w);
        hash = hash * 16777619u;
    }

    return hash;
}

// Validate cached scene data
@compute @workgroup_size(64, 1, 1)
fn cs_validate_cache(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;

    // Only first thread computes hashes to avoid race conditions
    if (thread_id == 0u) {
        let bvh_count = arrayLength(&bvh_nodes);
        let material_count = arrayLength(&material_data);

        // Compute current scene hashes
        let new_bvh_hash = compute_hash(&bvh_nodes, bvh_count);
        let new_material_hash = compute_hash(&material_data, material_count);

        // Simple texture hash (just texture array size for now)
        let texture_dims = textureDimensions(texture_array);
        let new_texture_hash = texture_dims.x ^ (texture_dims.y << 16u);

        // Combine into scene hash
        let new_scene_hash = new_bvh_hash ^ new_material_hash ^ new_texture_hash;

        // Check if cache is valid
        let cache_valid = (cache_validation.scene_hash == new_scene_hash) &&
                         (cache_validation.bvh_hash == new_bvh_hash) &&
                         (cache_validation.material_hash == new_material_hash) &&
                         (cache_validation.texture_hash == new_texture_hash);

        if (cache_valid) {
            cache_stats.hit_count++;
        } else {
            cache_stats.miss_count++;

            // Update validation hashes
            cache_validation.scene_hash = new_scene_hash;
            cache_validation.bvh_hash = new_bvh_hash;
            cache_validation.material_hash = new_material_hash;
            cache_validation.texture_hash = new_texture_hash;
        }
    }
}

// Cache warmup compute shader
@compute @workgroup_size(64, 1, 1)
fn cs_cache_warmup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let bvh_count = arrayLength(&bvh_nodes);

    if (thread_id >= bvh_count) {
        return;
    }

    // Pre-load BVH nodes into cache by accessing them
    let node = bvh_nodes[thread_id];

    // Simple operation to ensure the data is actually loaded
    let checksum = node.x + node.y + node.z + node.w;

    // Store checksum in cache hash array (if space available)
    if (thread_id < arrayLength(&cache_hashes)) {
        cache_hashes[thread_id] = bitcast<u32>(checksum);
    }
}

// Cache statistics reset
@compute @workgroup_size(1, 1, 1)
fn cs_reset_cache_stats(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        cache_stats.total_entries = 0u;
        cache_stats.complete_entries = 0u;
        cache_stats.hit_count = 0u;
        cache_stats.miss_count = 0u;

        cache_validation.scene_hash = 0u;
        cache_validation.bvh_hash = 0u;
        cache_validation.material_hash = 0u;
        cache_validation.texture_hash = 0u;
    }
}