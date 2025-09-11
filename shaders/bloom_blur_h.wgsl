// Bloom blur horizontal pass (placeholder)
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let px = 1.0 / f32(dims.x);
  let uv0 = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(dims);
  var sum = vec3<f32>(0.0);
  let w: array<f32,5> = array<f32,5>(0.204164, 0.304005, 0.093913, 0.017, 0.004918);
  for (var i:i32 = -4; i <= 4; i++) {
    let uv = uv0 + vec2<f32>(f32(i)*px, 0.0);
    sum += textureSampleLevel(src_tex, samp, uv, 0.0).rgb * w[abs(i)];
  }
  textureStore(dst_tex, vec2<i32>(gid.xy), vec4<f32>(sum, 1.0));
}

