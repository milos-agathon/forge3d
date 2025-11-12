// HZB (Hierarchical Z-Buffer) build shader
// Generates a min-depth mipmap pyramid for accelerated occlusion queries (P5)

// ============================================================================
// Copy pass: Depth texture -> R32Float mip 0
// ============================================================================

@group(0) @binding(0) var depth_in: texture_depth_2d;
@group(0) @binding(1) var hzb_out: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_in);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }
    
    let depth = textureLoad(depth_in, vec2<u32>(gid.xy), 0);
    textureStore(hzb_out, gid.xy, vec4<f32>(depth, 0.0, 0.0, 0.0));
}

// ============================================================================
// Downsample pass: R32Float mip N -> R32Float mip N+1
// ============================================================================

struct DownsampleParams {
    reversed_z: u32,  // 1 if using reversed-Z, 0 otherwise
}

@group(0) @binding(0) var hzb_src: texture_2d<f32>;
@group(0) @binding(1) var hzb_dst: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: DownsampleParams;

// Min-depth reduction (conservative occlusion test)
// For reversed-Z (far=0.0, near=1.0), we want max depth
// For standard-Z (near=0.0, far=1.0), we want min depth
fn reduce_depth(d0: f32, d1: f32, d2: f32, d3: f32, reversed: bool) -> f32 {
    if (reversed) {
        // Reversed-Z: further objects have smaller depth, so use max for conservative test
        return max(max(d0, d1), max(d2, d3));
    } else {
        // Standard-Z: further objects have larger depth, so use min for conservative test
        return min(min(d0, d1), min(d2, d3));
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_dims = textureDimensions(hzb_dst);
    if (gid.x >= dst_dims.x || gid.y >= dst_dims.y) {
        return;
    }
    
    let src_xy = gid.xy * 2u;
    let src_dims = textureDimensions(hzb_src);
    
    // Sample 2x2 quad from source mip
    let d0 = textureLoad(hzb_src, src_xy + vec2<u32>(0u, 0u), 0).r;
    let d1 = textureLoad(hzb_src, min(src_xy + vec2<u32>(1u, 0u), src_dims - 1u), 0).r;
    let d2 = textureLoad(hzb_src, min(src_xy + vec2<u32>(0u, 1u), src_dims - 1u), 0).r;
    let d3 = textureLoad(hzb_src, min(src_xy + vec2<u32>(1u, 1u), src_dims - 1u), 0).r;
    
    let reversed = params.reversed_z != 0u;
    let min_depth = reduce_depth(d0, d1, d2, d3, reversed);
    
    textureStore(hzb_dst, gid.xy, vec4<f32>(min_depth, 0.0, 0.0, 0.0));
}
