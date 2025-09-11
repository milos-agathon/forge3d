// Terrain impostor atlas shader (scaffold)
// Placeholder WGSL to reserve entry points for atlas generation.

@group(0) @binding(0) var dst_tex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst_tex);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  // Write a debug pattern
  let c = vec4<f32>(f32(gid.x % 256u)/255.0, f32(gid.y % 256u)/255.0, 0.0, 1.0);
  textureStore(dst_tex, vec2<i32>(gid.xy), c);
}

