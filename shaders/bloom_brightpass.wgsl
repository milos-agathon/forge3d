// Bloom bright-pass filter (placeholder implementation)
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;

struct Params { threshold: f32, _pad0: vec3<f32> };
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
  let c = textureSampleLevel(src_tex, samp, uv, 0.0);
  let luma = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
  let outc = select(vec4<f32>(0.0), vec4<f32>(c.rgb, 1.0), luma > params.threshold);
  textureStore(dst_tex, vec2<i32>(gid.xy), outc);
}

