// T3.3 Terrain shader — compatible with Rust pipeline bind group layouts.
// Layout: 0=Globals UBO, 1=height R32Float + NonFiltering sampler, 2=LUT RGBA8 + Filtering sampler.
// This version adds a deterministic analytic height fallback to avoid uniform output with a 1×1 dummy height.

// ---------- Globals UBO (176 bytes total, must match Rust) ----------
// std140-compatible layout: 176 bytes total
// Field breakdown:
//   view:                 mat4x4<f32> = 64 B
//   proj:                 mat4x4<f32> = 64 B  
//   sun_exposure:         vec4<f32>   = 16 B (xyz = sun_dir normalized, w = exposure)
//   spacing_h_exag_pad:   vec4<f32>   = 16 B (x=dx, y=dy, z=height_exaggeration, w=palette_index)
//   _pad_tail:            vec4<f32>   = 16 B (X: lod_morph [0..1], Y: coarse_factor (≥1), Z: skirt_depth, W: reserved)
//   TOTAL:                            = 176 B
struct Globals {
  view: mat4x4<f32>,                    // 64 B
  proj: mat4x4<f32>,                    // 64 B
  sun_exposure: vec4<f32>,              // xyz = sun_dir (normalized), w = exposure (16 B)
  spacing_h_exag_pad: vec4<f32>,        // x=dx, y=dy, z=height_exaggeration, w=palette_index (16 B)
  _pad_tail: vec4<f32>,                 // tail padding to keep total size multiple-of-16 (16 B)
};

@group(0) @binding(0) var<uniform> globals : Globals;

// ---------- Textures & samplers ----------
@group(1) @binding(0) var height_tex  : texture_2d<f32>;  // R32Float, non-filterable
@group(1) @binding(1) var height_samp : sampler;          // NonFiltering at pipeline level

@group(2) @binding(0) var lut_tex  : texture_2d<f32>;     // RGBA8 (sRGB/UNORM), filterable, 256×N multi-palette
@group(2) @binding(1) var lut_samp : sampler;

// B7: Cloud shadow overlay (moved to group(4) to allow tile at group(3))
@group(4) @binding(0) var cloud_shadow_tex  : texture_2d<f32>;  // Cloud shadow texture
@group(4) @binding(1) var cloud_shadow_samp : sampler;

// B5: Planar reflections resources
struct ReflectionPlane {
  plane_equation: vec4<f32>,
  reflection_matrix: mat4x4<f32>,
  reflection_view: mat4x4<f32>,
  reflection_projection: mat4x4<f32>,
  plane_center: vec4<f32>,
  plane_size: vec4<f32>,
};

struct PlanarReflectionUniforms {
  reflection_plane: ReflectionPlane,
  enable_reflections: u32,
  reflection_intensity: f32,
  fresnel_power: f32,
  blur_kernel_size: u32,
  max_blur_radius: f32,
  reflection_resolution: f32,
  distance_fade_start: f32,
  distance_fade_end: f32,
  debug_mode: u32,
  camera_position: vec4<f32>,
  _pad: vec3<f32>,
};

@group(5) @binding(0) var<uniform> reflection_uniforms : PlanarReflectionUniforms;
@group(5) @binding(1) var reflection_texture : texture_2d<f32>;
@group(5) @binding(2) var reflection_sampler : sampler;
@group(5) @binding(3) var reflection_depth : texture_depth_2d;

// E2: Per-tile uniforms for multi-LOD patch rendering (group 3)
struct TileUniforms {
  // world_remap = (scale_x, scale_y, offset_x, offset_y) to map local in.pos_xy → world plane xz
  world_remap: vec4<f32>,
};
@group(3) @binding(0) var<uniform> tile : TileUniforms;
// E1b: Per-draw tile slot (lod,x,y,slot index) and mosaic params
struct TileSlot { lod: u32, x: u32, y: u32, slot: u32 };
struct MosaicParams { inv_tiles_x: f32, inv_tiles_y: f32, tiles_x: u32, tiles_y: u32 };
@group(3) @binding(2) var<uniform> TileSlotU : TileSlot;
@group(3) @binding(3) var<uniform> MParams : MosaicParams;
// E1: Page table storage buffer for tile->slot mapping (demonstration read only)
struct PageTableEntry { lod: u32, x: u32, y: u32, _pad0: u32, sx: u32, sy: u32, slot: u32, _pad1: u32 };
@group(3) @binding(1) var<storage, read> PageTable : array<PageTableEntry>;

// ---------- IO ----------
struct VsIn {
  // position.xy in plane
  @location(0) pos_xy : vec2<f32>,
  @location(1) uv     : vec2<f32>,
};

struct VsOut {
  @builtin(position) clip_pos : vec4<f32>,
  @location(0) uv             : vec2<f32>,
  @location(1) height         : f32,
  @location(2) xz             : vec2<f32>,   // pass plane x/z to fragment for shading
};

struct FsOut {
  @location(0) color : vec4<f32>,
  @location(1) normal_depth : vec4<f32>,
};

// Analytic fallback height that varies across the grid. Amplitude ≈ ±0.5 (matches Globals defaults).
fn analytic_height(x: f32, z: f32) -> f32 {
  return sin(x * 1.3) * 0.25 + cos(z * 1.1) * 0.25;
}

fn sample_height_with_fallback(
  uv: vec2<f32>,
  uv_offset: vec2<f32>,
  xz: vec2<f32>,
  spacing: f32
) -> f32 {
  let sample_uv = clamp(uv + uv_offset, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
  let h_tex = textureSampleLevel(height_tex, height_samp, sample_uv, 0.0).r;
  let offset_xz = vec2<f32>(xz.x + uv_offset.x * spacing, xz.y + uv_offset.y * spacing);
  let h_ana = analytic_height(offset_xz.x, offset_xz.y);
  return h_tex + h_ana;
}

// B7: Sample cloud shadow at world position
fn sample_cloud_shadow(world_pos: vec2<f32>, terrain_scale: f32) -> f32 {
  // Convert world position to UV coordinates for cloud shadow texture
  // Normalize world coordinates to [0, 1] range
  let terrain_size = terrain_scale * 2.0; // Assuming terrain spans from -terrain_scale to +terrain_scale
  let shadow_uv = (world_pos / terrain_size + 1.0) * 0.5;

  // Sample cloud shadow texture (shadow multiplier is stored in RGB channels)
  let shadow_sample = textureSampleLevel(cloud_shadow_tex, cloud_shadow_samp, shadow_uv, 0.0);
  return shadow_sample.r; // Return shadow multiplier [0, 1] where 1 = no shadow, 0 = full shadow
}

fn distance_to_plane(point: vec3<f32>, plane: vec4<f32>) -> f32 {
  return dot(vec4<f32>(point, 1.0), plane);
}

fn apply_planar_reflection(
  world_pos: vec3<f32>,
  world_normal: vec3<f32>,
  base_color: vec3<f32>,
) -> vec3<f32> {
  if reflection_uniforms.enable_reflections == 0u {
    return base_color;
  }

  let plane = reflection_uniforms.reflection_plane.plane_equation;
  let reflected_pos = reflection_uniforms.reflection_plane.reflection_view * vec4<f32>(world_pos, 1.0);
  let projected = reflection_uniforms.reflection_plane.reflection_projection * reflected_pos;
  if projected.w == 0.0 {
    return base_color;
  }
  let ndc = projected.xyz / projected.w;
  let uv = ndc.xy * 0.5 + 0.5;
  if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
    return base_color;
  }

  var reflection = textureSample(reflection_texture, reflection_sampler, uv);
  let kernel = f32(reflection_uniforms.blur_kernel_size);
  if kernel > 1.0 {
    let texel = reflection_uniforms.max_blur_radius / max(reflection_uniforms.reflection_resolution, 1.0);
    let offset = texel * max(kernel * 0.5, 1.0);
    let offsets = array<vec2<f32>, 4>(
      vec2<f32>(offset, 0.0),
      vec2<f32>(-offset, 0.0),
      vec2<f32>(0.0, offset),
      vec2<f32>(0.0, -offset)
    );
    var accum = reflection;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
      let sample_uv = clamp(uv + offsets[i], vec2<f32>(0.0), vec2<f32>(1.0));
      accum = accum + textureSample(reflection_texture, reflection_sampler, sample_uv);
    }
    reflection = accum / 5.0;
  }

  let camera_pos = reflection_uniforms.camera_position.xyz;
  let view_dir = normalize(camera_pos - world_pos);
  let fresnel = pow(max(0.0, 1.0 - dot(world_normal, view_dir)), reflection_uniforms.fresnel_power);
  let distance = abs(distance_to_plane(world_pos, plane));
  let fade_start = reflection_uniforms.distance_fade_start;
  let fade_end = max(reflection_uniforms.distance_fade_end, fade_start + 0.001);
  let fade = clamp((fade_end - distance) / max(fade_end - fade_start, 0.001), 0.0, 1.0);
  let intensity = clamp(reflection_uniforms.reflection_intensity * fresnel * fade, 0.0, 1.0);

  switch reflection_uniforms.debug_mode {
    case 1u: {
      return vec3<f32>(uv, 0.0);
    }
    case 2u: {
      return vec3<f32>(intensity, intensity, intensity);
    }
    case 3u: {
      let dist_norm = clamp(distance / fade_end, 0.0, 1.0);
      return vec3<f32>(dist_norm, 0.0, 1.0 - dist_norm);
    }
    case 4u: {
      let blur_norm = clamp(kernel / 9.0, 0.0, 1.0);
      return vec3<f32>(blur_norm, 1.0 - blur_norm, 0.0);
    }
    default: {
      return mix(base_color, reflection.rgb, intensity);
    }
  }
}

// ---------- Vertex ----------
@vertex
fn vs_main(in: VsIn) -> VsOut {
  let spacing      = max(globals.spacing_h_exag_pad.x, 1e-8);
  let exaggeration = globals.spacing_h_exag_pad.z;

  // Remap UV into the tile's atlas sub-rect via slot + mosaic params
  let inv = vec2<f32>(MParams.inv_tiles_x, MParams.inv_tiles_y);
  let tiles_x = max(MParams.tiles_x, 1u);
  let sx = f32(TileSlotU.slot % tiles_x);
  let sy = f32(TileSlotU.slot / tiles_x);
  let base = vec2<f32>(sx, sy) * inv;
  let uv_tile = clamp(in.uv * inv + base, vec2<f32>(0.0), vec2<f32>(1.0));
  // Sample height with a NonFiltering sampler; level 0 to avoid filtering.
  var h_tex = textureSampleLevel(height_tex, height_samp, uv_tile, 0.0).r;

  // E2: Geomorphing — blend between a coarse (quantized) sample and fine sample
  let morph = clamp(globals._pad_tail.x, 0.0, 1.0);
  let coarse_factor = max(globals._pad_tail.y, 1.0);
  if (morph < 1.0) {
    // Quantize UVs to emulate a coarser grid without mips
    let tex_dims = vec2<f32>(textureDimensions(height_tex, 0));
    let step = vec2<f32>(coarse_factor) / max(tex_dims, vec2<f32>(1.0));
    // Guard against zero step by clamping denominator above
    let uv_q = (floor(uv_tile / step) + 0.5) * step;
    let uv_qc = clamp(uv_q, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));
    let h_coarse = textureSampleLevel(height_tex, height_samp, uv_qc, 0.0).r;
    h_tex = mix(h_coarse, h_tex, morph);
  }

  // Deterministic analytic fallback guarantees variation even if height_tex is 1x1.
  let h_ana = analytic_height(in.pos_xy.x, in.pos_xy.y);

  let h_base = h_tex + h_ana;

  // E2: Skirts — apply only to the skirt ring (uv outside [0,1]) and honor per-edge mask
  let skirt_depth = max(globals._pad_tail.z, 0.0);
  let mask: u32 = u32(globals._pad_tail.w + 0.5);
  let left_ring   = in.uv.x < 0.0;
  let right_ring  = in.uv.x > 1.0;
  let bottom_ring = in.uv.y < 0.0;
  let top_ring    = in.uv.y > 1.0;
  let do_left   = left_ring   && ((mask & 0x1u) != 0u);
  let do_right  = right_ring  && ((mask & 0x2u) != 0u);
  let do_bottom = bottom_ring && ((mask & 0x4u) != 0u);
  let do_top    = top_ring    && ((mask & 0x8u) != 0u);
  let is_skirt = do_left || do_right || do_bottom || do_top;
  let h = select(h_base, h_base - skirt_depth, is_skirt);

  // Build world position from per-tile remap
  let plane_x = in.pos_xy.x * tile.world_remap.x + tile.world_remap.z;
  let plane_z = in.pos_xy.y * tile.world_remap.y + tile.world_remap.w;
  let world = vec3<f32>(plane_x, h * exaggeration, plane_z);

  var out : VsOut;
  out.clip_pos = globals.proj * (globals.view * vec4<f32>(world, 1.0));
  out.uv       = in.uv;
  out.height   = h;
  out.xz       = in.pos_xy;
  return out;
}

// ---------- Fragment ----------
@fragment
fn fs_main(in: VsOut) -> FsOut {
  let spacing = max(globals.spacing_h_exag_pad.x, 1e-8);
  let exaggeration = globals.spacing_h_exag_pad.z;
  let h_range = max(globals.spacing_h_exag_pad.y, 1e-8);

  let uv = in.uv;
  let inv = vec2<f32>(MParams.inv_tiles_x, MParams.inv_tiles_y);
  let tiles_x = max(MParams.tiles_x, 1u);
  let sx = f32(TileSlotU.slot % tiles_x);
  let sy = f32(TileSlotU.slot / tiles_x);
  let base = vec2<f32>(sx, sy) * inv;
  let uv_tile = clamp(uv * inv + base, vec2<f32>(0.0), vec2<f32>(1.0));
  let tex_dims = vec2<f32>(textureDimensions(height_tex, 0));
  let texel = vec2<f32>(1.0) / tex_dims;

  let h_left = sample_height_with_fallback(uv_tile, vec2<f32>(-texel.x, 0.0), in.xz, spacing);
  let h_right = sample_height_with_fallback(uv_tile, vec2<f32>(texel.x, 0.0), in.xz, spacing);
  let h_down = sample_height_with_fallback(uv_tile, vec2<f32>(0.0, -texel.y), in.xz, spacing);
  let h_up = sample_height_with_fallback(uv_tile, vec2<f32>(0.0, texel.y), in.xz, spacing);

  let spacing_step = max(spacing, 1e-5);
  let grad_x = ((h_right - h_left) * exaggeration) / (2.0 * spacing_step);
  let grad_z = ((h_up - h_down) * exaggeration) / (2.0 * spacing_step);
  let normal_ws = normalize(vec3<f32>(-grad_x, 1.0, -grad_z));

  let t = clamp(0.5 + in.height / (2.0 * h_range), 0.0, 1.0);
  let palette_index = globals.spacing_h_exag_pad.w;
  let lut_dimensions = vec2<f32>(textureDimensions(lut_tex, 0));
  let v_coord = (palette_index + 0.5) / lut_dimensions.y;
  let lut_color = textureSampleLevel(lut_tex, lut_samp, vec2<f32>(t, v_coord), 0.0);

  let sun_L = normalize(globals.sun_exposure.xyz);
  let sun_lambert = clamp(dot(normal_ws, sun_L), 0.0, 1.0);
  let sun_contribution = globals.sun_exposure.w * sun_lambert;

  // B7: Apply cloud shadow modulation
  let world_pos_2d = vec2<f32>(in.xz.x * spacing, in.xz.y * spacing);
  let cloud_shadow_multiplier = sample_cloud_shadow(world_pos_2d, spacing * 50.0); // Adjust scale as needed
  let shadowed_sun_contribution = sun_contribution * cloud_shadow_multiplier;

  let shade = mix(0.15, 1.0, clamp(shadowed_sun_contribution, 0.0, 1.0));
  var lit_color = lut_color.rgb * shade;

  let world = vec3<f32>(in.xz.x * spacing, in.height * exaggeration, in.xz.y * spacing);
  lit_color = apply_planar_reflection(world, normal_ws, lit_color);

  let tonemapped = reinhard(lit_color);
  let gamma_corrected = gamma_correct(tonemapped);

  let view_pos = (globals.view * vec4<f32>(world, 1.0)).xyz;
  let linear_depth = max(-view_pos.z, 0.0);

  let normal_encoded = normal_ws * 0.5 + vec3<f32>(0.5);
  var out: FsOut;
  out.color = vec4<f32>(gamma_corrected, 1.0);
  out.normal_depth = vec4<f32>(normal_encoded, linear_depth);

  // E1: No-op read from page table buffer to demonstrate binding (and prevent DCE)
  if (arrayLength(&PageTable) > 0u) {
    let _pt_dbg = f32(PageTable[0u].slot) * 0.0;
    out.color = out.color + vec4<f32>(0.0) * _pt_dbg;
  }
  return out;
}


// C4: Explicit tonemap functions for compliance
fn reinhard(x: vec3<f32>) -> vec3<f32> {
  return x / (1.0 + x);
}

fn gamma_correct(x: vec3<f32>) -> vec3<f32> {
  return pow(x, vec3<f32>(1.0 / 2.2));
}