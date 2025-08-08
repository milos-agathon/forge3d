// Terrain uniforms + LUT colormap sampling with simple tonemap.
// Size matches Rust's TerrainUniforms: 64 + 64 + 16 + 16 + 16 = 176 bytes.

struct TerrainUniforms {
  view : mat4x4<f32>,
  proj : mat4x4<f32>,
  // xyz = sun_dir, w = exposure
  sun_exposure : vec4<f32>,
  // x = spacing, y = h_range, z = exaggeration, w = 0
  spacing_h_exag_pad : vec4<f32>,
  // final lane to reach 176 bytes
  _pad_tail : vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms : TerrainUniforms;

@group(0) @binding(1)
var lut_tex : texture_2d<f32>;

@group(0) @binding(2)
var lut_samp : sampler;

struct VSIn {
  @location(0) pos : vec3<f32>,
  @location(1) nrm : vec3<f32>,
};

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0) height : f32,
  @location(1) ndotl  : f32,
};

@vertex
fn vs_main(v: VSIn) -> VSOut {
  var out : VSOut;

  let world = vec4<f32>(v.pos, 1.0);
  out.clip = uniforms.proj * uniforms.view * world;

  // Normalize height to ~[0,1]
  let hr = max(uniforms.spacing_h_exag_pad.y, 1e-6);
  let h  = v.pos.y * uniforms.spacing_h_exag_pad.z; // exaggeration
  let h_norm = 0.5 + 0.5 * (h / (0.5 * hr));
  out.height = clamp(h_norm, 0.0, 1.0);

  // Simple N·L
  let L = normalize(uniforms.sun_exposure.xyz);
  out.ndotl = max(dot(normalize(v.nrm), L), 0.0);

  return out;
}

fn reinhard_tonemap(c: vec3<f32>, exposure: f32) -> vec3<f32> {
  let x = c * exposure;
  return x / (vec3<f32>(1.0) + x);
}

@fragment
fn fs_main(inp: VSOut) -> @location(0) vec4<f32> {
  // Sample 256x1 LUT (created as texture_2d)
  let uv = vec2<f32>(inp.height, 0.5);
  var col = textureSampleLevel(lut_tex, lut_samp, uv, 0.0).rgb;

  // Cheap diffuse to avoid flat images in tests
  let lit = mix(0.15, 1.0, inp.ndotl) * col;

  let exposure = max(uniforms.sun_exposure.w, 1e-3);
  let mapped = reinhard_tonemap(lit, exposure);
  return vec4<f32>(mapped, 1.0);
}
