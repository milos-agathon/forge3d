// src/shaders/fog_upsample.wgsl
// Simple bilinear upsample from half-resolution fog into full-resolution fog.
// Optional bilateral depth-aware upsample guided by full-resolution depth.

struct UpsampleParams {
  sigma: f32,          // Depth sigma for bilateral weighting (e.g., 0.02)
  use_bilateral: u32,  // 1 = enable bilateral weighting, 0 = plain bilinear
  _pad: vec2<f32>,
};

@group(0) @binding(0) var src_tex : texture_2d<f32>;
@group(0) @binding(1) var src_samp : sampler;
@group(0) @binding(2) var dst_tex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var full_depth : texture_2d<f32>;
@group(0) @binding(4) var depth_samp : sampler;
@group(0) @binding(5) var<uniform> params : UpsampleParams;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(dst_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
  // Source is half resolution; sample at half scale
  let src_uv = uv * 0.5;

  if (params.use_bilateral == 0u) {
    // Fast bilinear
    let fog = textureSampleLevel(src_tex, src_samp, src_uv, 0.0);
    textureStore(dst_tex, vec2<i32>(gid.xy), fog);
    return;
  }

  // Depth-aware upsample guided by full-resolution depth
  // Reference depth at the full-resolution pixel (explicit LOD for compute)
  let d_ref = textureSampleLevel(full_depth, depth_samp, uv, 0.0).r;

  // Sample the four nearest low-res texels with bilinear base weights
  let tex_size = vec2<f32>(textureDimensions(src_tex));
  let texel = 1.0 / tex_size;
  let base = src_uv * tex_size - 0.5;
  let i0 = floor(base);
  let f = fract(base);

  let uv00 = (i0 + vec2<f32>(0.0, 0.0) + 0.5) * texel;
  let uv10 = (i0 + vec2<f32>(1.0, 0.0) + 0.5) * texel;
  let uv01 = (i0 + vec2<f32>(0.0, 1.0) + 0.5) * texel;
  let uv11 = (i0 + vec2<f32>(1.0, 1.0) + 0.5) * texel;

  // Map those low-res centers back to full-res uv to fetch guiding depths
  let uv00_full = uv00 * 2.0;
  let uv10_full = uv10 * 2.0;
  let uv01_full = uv01 * 2.0;
  let uv11_full = uv11 * 2.0;

  let d00 = textureSampleLevel(full_depth, depth_samp, uv00_full, 0.0).r;
  let d10 = textureSampleLevel(full_depth, depth_samp, uv10_full, 0.0).r;
  let d01 = textureSampleLevel(full_depth, depth_samp, uv01_full, 0.0).r;
  let d11 = textureSampleLevel(full_depth, depth_samp, uv11_full, 0.0).r;

  // Bilinear weights
  var w00 = (1.0 - f.x) * (1.0 - f.y);
  var w10 = f.x * (1.0 - f.y);
  var w01 = (1.0 - f.x) * f.y;
  var w11 = f.x * f.y;

  // Depth weights
  let sigma = max(params.sigma, 1e-6);
  let w_d00 = exp(-abs(d_ref - d00) / sigma);
  let w_d10 = exp(-abs(d_ref - d10) / sigma);
  let w_d01 = exp(-abs(d_ref - d01) / sigma);
  let w_d11 = exp(-abs(d_ref - d11) / sigma);

  w00 = w00 * w_d00;
  w10 = w10 * w_d10;
  w01 = w01 * w_d01;
  w11 = w11 * w_d11;

  let sum_w = max(w00 + w10 + w01 + w11, 1e-6);

  // Fetch low-res fog and combine with normalized weights
  let f00 = textureSampleLevel(src_tex, src_samp, uv00, 0.0);
  let f10 = textureSampleLevel(src_tex, src_samp, uv10, 0.0);
  let f01 = textureSampleLevel(src_tex, src_samp, uv01, 0.0);
  let f11 = textureSampleLevel(src_tex, src_samp, uv11, 0.0);

  let fog = (f00 * w00 + f10 * w10 + f01 * w01 + f11 * w11) / sum_w;
  textureStore(dst_tex, vec2<i32>(gid.xy), fog);
}
