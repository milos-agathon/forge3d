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
  @location(0) world_pos : vec3<f32>,      // world-space (for lighting/colormap later)
  @location(1) world_nrm : vec3<f32>,      // world-space normal (corrected for non-uniform scale)
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

  var out: VSOut;
  out.position  = clip;
  out.world_pos = world;
  out.world_nrm = correct_normal_for_scale(in.nrm, spacing, exag);
  out.height    = world.y;
  return out;
}

// Minimal fragment stage for T3.1 to keep pipeline compiling.
// Renders a simple lambert-ish grayscale using sun_dir⋅normal.
// T3.2+ will replace this with proper colormap + lighting.
struct FSOut { @location(0) color : vec4<f32>, }

@fragment
fn fs_main(in: VSOut) -> FSOut {
  let sun_dir = normalize(ubo.sun_exposure.xyz);
  let ndotl   = max(dot(normalize(in.world_nrm), sun_dir), 0.0);
  let exposure = ubo.sun_exposure.w;

  // Simple shaded grayscale; no colormap sampling yet (reserved for T3.2).
  let shade = pow(ndotl, 0.8) * exposure;
  var out: FSOut;
  out.color = vec4<f32>(shade, shade, shade, 1.0);
  return out;
}
