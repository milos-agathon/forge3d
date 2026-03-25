struct DepthCopyParams {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var depth_scalar : texture_2d<f32>;
@group(0) @binding(1) var depth_rgba : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params : DepthCopyParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) {
    return;
  }

  let coords = vec2<i32>(gid.xy);
  let depth = textureLoad(depth_scalar, coords, 0).x;
  textureStore(depth_rgba, coords, vec4<f32>(depth, 0.0, 0.0, 1.0));
}
