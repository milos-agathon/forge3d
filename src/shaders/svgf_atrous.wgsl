// src/shaders/svgf_atrous.wgsl
// WGSL compute pass stub for edge-aware A-trous filtering guided by albedo/normal/depth
// Documents bind layout for future GPU implementation while remaining a no-op here
// RELEVANT FILES:src/shaders/svgf_variance.wgsl,src/denoise/svgf/pipelines.rs,python/forge3d/denoise.py

// Bind group 0:
//  binding 0: color_in        (texture_2d<f16/f32>, sampled)
//  binding 1: albedo          (texture_2d<f16/f32>, sampled)
//  binding 2: normal          (texture_2d<f16/f32>, sampled)
//  binding 3: depth           (texture_2d<f32>, sampled)
//  binding 4: color_out       (texture_storage_2d<rgba16float, write>)
//  binding 5: params          (uniform)

struct Params {
  width: u32,
  height: u32,
  step: u32,         // a-trous step
  iterations: u32,   // total iterations
};

@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  // Stub only
}

