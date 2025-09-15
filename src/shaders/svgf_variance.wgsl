// src/shaders/svgf_variance.wgsl
// WGSL compute pass stub for SVGF temporal accumulation and variance estimation
// Exists to define resource layout and future compute entry for variance/moments update
// RELEVANT FILES:src/shaders/svgf_reproject.wgsl,src/shaders/svgf_atrous.wgsl,src/denoise/svgf/pipelines.rs

// Bind group 0:
//  binding 0: radiance_curr   (texture_2d<f16/f32>, sampled)
//  binding 1: moments_prev    (texture_storage_2d<rg16float, read>)
//  binding 2: moments_curr    (texture_storage_2d<rg16float, write>)
//  binding 3: variance_out    (texture_storage_2d<r16float,  write>)
//  binding 4: params          (uniform)

struct Params {
  width: u32,
  height: u32,
  alpha: f32,   // temporal alpha
  pad:   f32,
};

@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  // Stub only
}

