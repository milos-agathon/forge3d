// T3.1 Terrain shader (vertex-focused). WGPU clip space, std140-compatible UBO.
// Layout must match Rust `TerrainUniforms` (total 176 bytes).

// Small epsilon used for safe divides in normal correction.
const EPS: f32 = 1e-8;

struct TerrainUniforms {
  view: mat4x4<f32>,                // offset 0,  size 64
  proj: mat4x4<f32>,                // offset 64, size 64
  sun_exposure: vec4<f32>,          // offset 128 (sun_dir.xyz, exposure)
  spacing_h_exag_pad: vec4<f32>,    // offset 144 (spacing, h_range, exaggeration, 0)
  _pad_tail: vec4<f32>,             // offset 160 (padding lane)
}
@group(0) @binding(0)
var<uniform> ubo: TerrainUniforms;

// Prefixed with underscores to avoid "unused binding" warnings while keeping the layout stable for future tasks.
@group(0) @binding(1) var _lut_tex: texture_2d<f32>;
@group(0) @binding(2) var _lut_samp: sampler;

struct VSIn {
  @location(0) pos : vec3<f32>,   // model-space position
  @location(1) nrm : vec3<f32>,   // model-space normal
}

struct VSOut {
  @builtin(position) position : vec4<f32>, // clip-space position  
  @location(0) uv : vec2<f32>,             // texture coordinates for height sampling
  @location(1) world_pos : vec3<f32>,      // world-space (for lighting/colormap later)
  @location(2) height    : f32,            // world-space height (y) — handy for color mapping
}

// Build the inverse-transpose of a diagonal scale matrix quickly.
// Model scale is S = diag(spacing, exaggeration, spacing) with no rotation/translation.
// For normals, we use N = normalize((S^{-1})^T * n) = normalize(vec3(n.x/spacing, n.y/exag, n.z/spacing)).
fn correct_normal_for_scale(n: vec3<f32>, spacing: f32, exag: f32) -> vec3<f32> {
  // Guard tiny denominators to avoid infinities if future code passes ~0 values.
  let s = max(spacing, EPS);
  let e = max(exag,    EPS);
  let nx = n.x / s;
  let ny = n.y / e;
  let nz = n.z / s;
  return normalize(vec3<f32>(nx, ny, nz));
}

@vertex
fn vs_main(in: VSIn) -> VSOut {
  let spacing = ubo.spacing_h_exag_pad.x;
  let exag    = ubo.spacing_h_exag_pad.z;

  // Model → World: non-uniform scale
  let world = vec3<f32>(in.pos.x * spacing,
                        in.pos.y * exag,
                        in.pos.z * spacing);

  // View/Projection (WGPU clip space provided by host)
  let view_pos = ubo.view * vec4<f32>(world, 1.0);
  let clip     = ubo.proj * view_pos;

  // Generate UV coordinates from model position (assuming model spans [0,1] range)
  let uv = vec2<f32>(in.pos.x, in.pos.z);

  var out: VSOut;
  out.position  = clip;
  out.uv        = uv;
  out.world_pos = world;
  out.height    = world.y;
  return out;
}

// T32-BEGIN:fs
// Terrain fragment shader — linear lighting + Reinhard tonemap to sRGB target.
// Contract (from T2.x): `@group(0) @binding(0) var<uniform> globals : Globals;`
// Bindings (terrain path): group(1)=height R32F + sampler, group(2)=LUT RGBA8UnormSrgb + sampler.

// External declarations expected (do NOT redeclare `struct Globals` here).
@group(1) @binding(0) var heightTex  : texture_2d<f32>;
@group(1) @binding(1) var heightSamp : sampler;
@group(2) @binding(0) var lutTex     : texture_2d<f32>;
@group(2) @binding(1) var lutSamp    : sampler;

fn vf_reinhard(x: vec3<f32>) -> vec3<f32> { return x / (vec3<f32>(1.0) + x); }
fn safe_normalize(v: vec3<f32>) -> vec3<f32> { let m = max(length(v), 1e-8); return v / m; }

@fragment
fn fs_main(
  @location(0) uv : vec2<f32>
) -> @location(0) vec4<f32> {
  // Texture size → robust UV step (avoid div-by-zero at tiny dims)
  let dims = vec2<f32>(textureDimensions(heightTex));
  let duv  = 1.0 / max(dims - vec2<f32>(1.0), vec2<f32>(1.0));

  // Heights with forward differences (per roadmap)
  let h  : f32 = textureSampleLevel(heightTex, heightSamp, uv, 0.0).r;
  let hx : f32 = textureSampleLevel(heightTex, heightSamp, uv + vec2<f32>(duv.x, 0.0), 0.0).r;
  let hy : f32 = textureSampleLevel(heightTex, heightSamp, uv + vec2<f32>(0.0, duv.y), 0.0).r;

  // Numeric guards from Globals
  let dx = max(globals.spacing.x, 1e-8);
  let dy = max(globals.spacing.y, 1e-8);
  let ex = max(globals.exaggeration, 1e-8);

  // Tangents in world units and normal
  let dpx = vec3<f32>(dx, 0.0, (hx - h) * ex);
  let dpy = vec3<f32>(0.0, dy, (hy - h) * ex);
  let n   = safe_normalize(cross(dpy, dpx));  // right-handed; +Y up

  // Lighting (linear)
  let L        = safe_normalize(globals.sun_dir);
  let lambert  = max(dot(n, L), 0.0);
  let ambient  = 0.12;
  let exposure = max(globals.exposure, 0.0);

  // Height normalization → LUT (linear sample from sRGB LUT texture)
  let denom = max(globals.h_range.y - globals.h_range.x, 1e-6);
  let t     = clamp((h - globals.h_range.x) / denom, 0.0, 1.0);
  let albedo = textureSampleLevel(lutTex, lutSamp, vec2<f32>(t, 0.5), 0.0).rgb;

  // Tone map in linear; write to sRGB target (hardware encodes)
  let lit    = albedo * (ambient + lambert) * exposure;
  let mapped = vf_reinhard(lit);
  return vec4<f32>(mapped, 1.0);
}
// T32-END:fs
