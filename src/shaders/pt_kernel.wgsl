// src/shaders/pt_kernel.wgsl
// Path tracing compute kernel scaffold with AOV storage targets and bindings.
// Exists to document/declare AOV outputs and uniform/state layout for the GPU path.
// RELEVANT FILES:src/path_tracing/compute.rs,src/path_tracing/aov.rs,python/forge3d/path_tracing.py

// Bind Group 0 (Uniforms): width, height, frame_index, aov_flags; camera params; exposure; seed_hi/seed_lo
// Bind Group 1 (Scene): readonly storage (materials, spheres/triangles, etc.)
// Bind Group 2 (Accum/State): storage buffers (if using accumulation)
// Bind Group 3 (AOV outputs): storage textures per AOV (enabled by aov_flags)

struct Uniforms {
  width: u32,
  height: u32,
  frame_index: u32,
  aov_flags: u32,
  cam_origin: vec3<f32>, cam_fov_y: f32,
  cam_right: vec3<f32>, cam_aspect: f32,
  cam_up: vec3<f32>, cam_exposure: f32,
  cam_forward: vec3<f32>, seed_hi: u32,
  seed_lo: u32,
  _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> ubo: Uniforms;

// AOV outputs (each is optional, enabled via bits in ubo.aov_flags)
// Bit layout (LSB..): 0=albedo,1=normal,2=depth,3=direct,4=indirect,5=emission,6=visibility
@group(3) @binding(0) var aov_albedo: texture_storage_2d<rgba16float, write>;
@group(3) @binding(1) var aov_normal: texture_storage_2d<rgba16float, write>;
@group(3) @binding(2) var aov_depth: texture_storage_2d<r32float, write>;
@group(3) @binding(3) var aov_direct: texture_storage_2d<rgba16float, write>;
@group(3) @binding(4) var aov_indirect: texture_storage_2d<rgba16float, write>;
@group(3) @binding(5) var aov_emission: texture_storage_2d<rgba16float, write>;
@group(3) @binding(6) var aov_visibility: texture_storage_2d<r8unorm, write>;

fn aov_enabled(bit: u32) -> bool {
  return (ubo.aov_flags & (1u << bit)) != 0u;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= ubo.width || gid.y >= ubo.height) { return; }
  let xy = vec2<i32>(i32(gid.x), i32(gid.y));
  let fx = f32(gid.x) / max(1.0, f32(ubo.width) - 1.0);
  let fy = f32(gid.y) / max(1.0, f32(ubo.height) - 1.0);

  // Simple synthetic values for scaffolding; real tracer writes physically based results.
  if (aov_enabled(0u)) { textureStore(aov_albedo, xy, vec4<f16>(f16(fx), f16(fy), f16(0.5), f16(1.0))); }
  if (aov_enabled(1u)) { textureStore(aov_normal, xy, vec4<f16>(f16(2.0*fx-1.0), f16(2.0*fy-1.0), f16(1.0), f16(1.0))); }
  if (aov_enabled(2u)) { textureStore(aov_depth, xy, vec4<f32>(1.0 - fx, 0.0, 0.0, 0.0)); }
  if (aov_enabled(3u)) { textureStore(aov_direct, xy, vec4<f16>(f16(fx), f16(fx*0.8), f16(fx*0.6), f16(1.0))); }
  if (aov_enabled(4u)) { textureStore(aov_indirect, xy, vec4<f16>(f16(fy*0.7), f16(fy), f16(fy*0.9), f16(1.0))); }
  if (aov_enabled(5u)) { textureStore(aov_emission, xy, vec4<f16>(f16(0.2), f16(0.1), f16(0.05), f16(1.0))); }
  if (aov_enabled(6u)) { textureStore(aov_visibility, xy, vec4<f32>(select(0.0, 1.0, fx+fy > 0.7), 0.0, 0.0, 0.0)); }
}
