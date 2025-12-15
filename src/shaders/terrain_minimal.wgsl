// Minimal terrain shader variant for devices with 4 bind groups.
// Layout: group(0)=globals, group(1)=height+sampler, group(2)=lut+sampler, group(3)=tile uniforms + page table + tile slot + mosaic params

struct Globals {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
  sun_exposure: vec4<f32>,
  spacing_h_exag_pad: vec4<f32>,
  _pad_tail: vec4<f32>,
};
@group(0) @binding(0) var<uniform> globals : Globals;

@group(1) @binding(0) var height_tex  : texture_2d<f32>;
@group(1) @binding(1) var height_samp : sampler;

@group(2) @binding(0) var lut_tex  : texture_2d<f32>;
@group(2) @binding(1) var lut_samp : sampler;

struct TileUniforms {
  world_remap: vec4<f32>,
};
@group(3) @binding(0) var<uniform> tile : TileUniforms;
// E1: Page table binding (read-only), and E1b per-draw uniforms
struct PageTableEntry { lod: u32, x: u32, y: u32, _pad0: u32, sx: u32, sy: u32, slot: u32, _pad1: u32 };
@group(3) @binding(1) var<storage, read> PageTable : array<PageTableEntry>;
struct TileSlot { lod: u32, x: u32, y: u32, slot: u32 };
struct MosaicParams { inv_tiles_x: f32, inv_tiles_y: f32, tiles_x: u32, tiles_y: u32 };
@group(3) @binding(2) var<uniform> TileSlotU : TileSlot;
@group(3) @binding(3) var<uniform> MParams : MosaicParams;

struct VsIn { @location(0) pos_xy: vec2<f32>, @location(1) uv: vec2<f32> };
struct VsOut { @builtin(position) clip_pos: vec4<f32>, @location(0) uv: vec2<f32>, @location(1) height: f32, @location(2) xz: vec2<f32> };
struct FsOut { @location(0) color: vec4<f32>, @location(1) normal_depth: vec4<f32> };

fn analytic_height(x: f32, z: f32) -> f32 { return sin(x * 1.3) * 0.25 + cos(z * 1.1) * 0.25; }

@vertex
fn vs_main(in: VsIn) -> VsOut {
  let spacing      = max(globals.spacing_h_exag_pad.x, 1e-8);
  let exaggeration = globals.spacing_h_exag_pad.z;
  // Compute uv_tile via slot and mosaic params
  let inv = vec2<f32>(MParams.inv_tiles_x, MParams.inv_tiles_y);
  let tiles_x = max(MParams.tiles_x, 1u);
  let sx = f32(TileSlotU.slot % tiles_x);
  let sy = f32(TileSlotU.slot / tiles_x);
  let base = vec2<f32>(sx, sy) * inv;
  let uv_tile = clamp(in.uv * inv + base, vec2<f32>(0.0), vec2<f32>(1.0));
  var h_tex = textureSampleLevel(height_tex, height_samp, uv_tile, 0.0).r;
  let morph = clamp(globals._pad_tail.x, 0.0, 1.0);
  let coarse_factor = max(globals._pad_tail.y, 1.0);
  if (morph < 1.0) {
    let tex_dims = vec2<f32>(textureDimensions(height_tex, 0));
    let step = vec2<f32>(coarse_factor) / max(tex_dims, vec2<f32>(1.0));
    let uv_q = (floor(uv_tile / step) + 0.5) * step;
    let uv_qc = clamp(uv_q, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let h_coarse = textureSampleLevel(height_tex, height_samp, uv_qc, 0.0).r;
    h_tex = mix(h_coarse, h_tex, morph);
  }
  let h_ana = analytic_height(in.pos_xy.x, in.pos_xy.y);
  var h = h_tex + h_ana;
  let skirt_depth = max(globals._pad_tail.z, 0.0);
  let is_skirt = (in.uv.x < 0.0) || (in.uv.x > 1.0) || (in.uv.y < 0.0) || (in.uv.y > 1.0);
  if (is_skirt) {
    h = h - skirt_depth;
  }
  let px = in.pos_xy.x * tile.world_remap.x + tile.world_remap.z;
  let pz = in.pos_xy.y * tile.world_remap.y + tile.world_remap.w;
  let world = vec3<f32>(px, h * exaggeration, pz);
  var out: VsOut;
  out.clip_pos = globals.proj * (globals.view * vec4<f32>(world, 1.0));
  out.uv = in.uv;
  out.height = h;
  out.xz = in.pos_xy;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
  let spacing = max(globals.spacing_h_exag_pad.x, 1e-8);
  let h_range = max(globals.spacing_h_exag_pad.y, 1e-8);
  let inv = vec2<f32>(MParams.inv_tiles_x, MParams.inv_tiles_y);
  let tiles_x = max(MParams.tiles_x, 1u);
  let sx = f32(TileSlotU.slot % tiles_x);
  let sy = f32(TileSlotU.slot / tiles_x);
  let base = vec2<f32>(sx, sy) * inv;
  let uv_tile = clamp(in.uv * inv + base, vec2<f32>(0.0), vec2<f32>(1.0));
  let tex_dims = vec2<f32>(textureDimensions(height_tex, 0));
  let texel = vec2<f32>(1.0) / tex_dims;
  let h_left = textureSampleLevel(height_tex, height_samp, clamp(uv_tile + vec2<f32>(-texel.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
  let h_right = textureSampleLevel(height_tex, height_samp, clamp(uv_tile + vec2<f32>(texel.x, 0.0), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
  let h_down = textureSampleLevel(height_tex, height_samp, clamp(uv_tile + vec2<f32>(0.0, -texel.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
  let h_up = textureSampleLevel(height_tex, height_samp, clamp(uv_tile + vec2<f32>(0.0, texel.y), vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).r;
  let spacing_step_x = max(spacing * texel.x, 1e-5);
  let spacing_step_z = max(spacing * texel.y, 1e-5);
  let grad_x = ((h_right - h_left) * globals.spacing_h_exag_pad.z) / (2.0 * spacing_step_x);
  let grad_z = ((h_up - h_down) * globals.spacing_h_exag_pad.z) / (2.0 * spacing_step_z);
  let normal_ws = normalize(vec3<f32>(-grad_x, 1.0, -grad_z));
  let palette_index = globals.spacing_h_exag_pad.w;
  let v_coord = (palette_index + 0.5) / vec2<f32>(textureDimensions(lut_tex, 0)).y;
  let lut_color = textureSampleLevel(lut_tex, lut_samp, vec2<f32>(clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0), v_coord), 0.0);
  let lit = lut_color.rgb * (0.5 + 0.5 * max(dot(normal_ws, normalize(globals.sun_exposure.xyz)), 0.0));
  let world = vec3<f32>(in.xz.x * spacing, in.height * globals.spacing_h_exag_pad.z, in.xz.y * spacing);
  let view_pos = (globals.view * vec4<f32>(world, 1.0)).xyz;
  let linear_depth = max(-view_pos.z, 0.0);
  var out: FsOut;
  out.color = vec4<f32>(pow(lit / (1.0 + lit), vec3<f32>(1.0/2.2)), 1.0);
  out.normal_depth = vec4<f32>(normal_ws * 0.5 + vec3<f32>(0.5), linear_depth);
  // E1: no-op read from PageTable to keep binding live
  if (arrayLength(&PageTable) > 0u) {
    let _pt_dbg = f32(PageTable[0u].slot) * 0.0;
    out.color = out.color + vec4<f32>(0.0) * _pt_dbg;
  }
  return out;
}
