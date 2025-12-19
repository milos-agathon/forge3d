// src/shaders/overlays.wgsl
// Fullscreen overlay compositor: drape overlay (RGBA8 sRGB) and optional altitude ramp from height texture.
// M5: Extended with depth-correct vector overlay support and halo rendering.

struct OverlayUniforms {
  view_proj: mat4x4<f32>,
  overlay_params: vec4<f32>, // x: overlay_enabled, y: overlay_alpha, z: altitude_enabled, w: altitude_alpha
  overlay_uv: vec4<f32>,     // x: uv_off_x, y: uv_off_y, z: uv_scale_x, w: uv_scale_y
  contour_params: vec4<f32>, // x: contour_enabled, y: interval, z: thickness_mul, w: pad
  contour_color: vec4<f32>,  // rgba for contour lines
  // M5: Vector overlay depth and halo parameters
  depth_params: vec4<f32>,   // x: depth_test_enabled, y: depth_bias, z: depth_bias_slope, w: pad
  halo_params: vec4<f32>,    // x: halo_enabled, y: halo_width, z: halo_blur, w: pad
  halo_color: vec4<f32>,     // rgba for halo/outline
};

@group(0) @binding(0) var<uniform> U : OverlayUniforms;
@group(0) @binding(1) var overlay_tex : texture_2d<f32>;
@group(0) @binding(2) var overlay_samp : sampler;
@group(0) @binding(3) var height_tex : texture_2d<f32>;
@group(0) @binding(4) var height_samp : sampler;
// E1: Page table storage buffer — tile->slot entries (matches Rust layout)
struct PageTableEntry { lod: u32, x: u32, y: u32, _pad0: u32, sx: u32, sy: u32, slot: u32, _pad1: u32 };
@group(0) @binding(5) var<storage, read> PageTable : array<PageTableEntry>;
// M5: Terrain depth texture for occlusion testing
@group(0) @binding(6) var depth_tex : texture_2d<f32>;
@group(0) @binding(7) var depth_samp : sampler;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi : u32) -> VsOut {
  var out : VsOut;
  // Fullscreen triangle
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0)
  );
  let pos = p[vi];
  out.pos = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
  return out;
}

fn ramp_color(h: f32) -> vec3<f32> {
  // Simple green->brown->white ramp
  let c0 = vec3<f32>(0.1, 0.5, 0.2);
  let c1 = vec3<f32>(0.5, 0.35, 0.2);
  let c2 = vec3<f32>(0.9, 0.9, 0.9);
  let mid = 0.6;
  let t = clamp(h, 0.0, 1.0);
  let low = mix(c0, c1, clamp(t / mid, 0.0, 1.0));
  let hi = mix(c1, c2, clamp((t - mid) / max(1e-3, 1.0 - mid), 0.0, 1.0));
  return mix(low, hi, select(0.0, 1.0, t > mid));
}

// M5: Check if overlay pixel is occluded by terrain depth
fn is_occluded_by_terrain(uv: vec2<f32>, overlay_depth: f32) -> bool {
  let depth_test_enabled = U.depth_params.x > 0.5;
  if (!depth_test_enabled) {
    return false;
  }
  
  // Sample terrain depth (assuming normalized depth 0=near, 1=far)
  let terrain_depth = textureSampleLevel(depth_tex, depth_samp, uv, 0.0).r;
  
  // Apply depth bias to prevent z-fighting
  let bias = U.depth_params.y;
  let biased_overlay_depth = overlay_depth - bias;
  
  // Overlay is occluded if terrain is closer (smaller depth value)
  return terrain_depth < biased_overlay_depth;
}

// M5: Sample overlay with halo effect
// Returns (color, alpha) with halo if enabled
fn sample_overlay_with_halo(uv: vec2<f32>) -> vec4<f32> {
  let uv_ov = vec2<f32>(U.overlay_uv.x, U.overlay_uv.y) + uv * vec2<f32>(U.overlay_uv.z, U.overlay_uv.w);
  let ov = textureSample(overlay_tex, overlay_samp, uv_ov);
  
  let halo_enabled = U.halo_params.x > 0.5;
  if (!halo_enabled || ov.a < 0.01) {
    return ov;
  }
  
  // Halo: sample neighbors and create outline effect
  let halo_width = U.halo_params.y;
  let halo_blur = U.halo_params.z;
  let halo_color = U.halo_color;
  
  // Get texture dimensions for pixel offset calculation
  let tex_size = vec2<f32>(textureDimensions(overlay_tex));
  let pixel_offset = halo_width / tex_size;
  
  // Sample 8 neighbors for halo detection
  var neighbor_alpha = 0.0;
  let offsets = array<vec2<f32>, 8>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0,  0.0),                        vec2<f32>(1.0,  0.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>(0.0,  1.0), vec2<f32>(1.0,  1.0)
  );
  
  for (var i = 0; i < 8; i = i + 1) {
    let sample_uv = uv_ov + offsets[i] * pixel_offset;
    let neighbor = textureSample(overlay_tex, overlay_samp, sample_uv);
    neighbor_alpha = max(neighbor_alpha, neighbor.a);
  }
  
  // Halo appears where neighbors have alpha but center might not
  let halo_strength = max(0.0, neighbor_alpha - ov.a);
  let halo_alpha = halo_strength * halo_color.a;
  
  // Blend halo under the overlay
  var result_color = mix(halo_color.rgb, ov.rgb, ov.a);
  var result_alpha = max(ov.a, halo_alpha);
  
  return vec4<f32>(result_color, result_alpha);
}

@fragment
fn fs_overlay(in: VsOut) -> @location(0) vec4<f32> {
  var col : vec3<f32> = vec3<f32>(0.0);
  var a   : f32 = 0.0;

  let ov_en = U.overlay_params.x > 0.5;
  let ov_a  = clamp(U.overlay_params.y, 0.0, 1.0);
  let alt_en = U.overlay_params.z > 0.5;

  if (ov_en) {
    // M5: Sample overlay with optional halo effect
    let ov = sample_overlay_with_halo(in.uv);
    
    // M5: Check depth occlusion (use normalized screen depth from position)
    // For fullscreen overlay, we use a default depth of 0.5 (mid-range)
    // Vector overlays would pass their actual depth
    let overlay_depth = 0.5;
    if (is_occluded_by_terrain(in.uv, overlay_depth)) {
      // Occluded - reduce or hide overlay
      col = ov.rgb;
      a = ov_a * 0.0; // Fully occluded
    } else {
      col = ov.rgb;
      a = ov_a * ov.a;
    }
  }

  if (alt_en) {
    // Sample height; assume height in 0..1 for now
    let h = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;
    let alt_col = ramp_color(clamp(h, 0.0, 1.0));
    let alt_a = clamp(U.overlay_params.w, 0.0, 1.0);
    // Composite altitude under overlay in shader, then alpha blend with target
    col = mix(alt_col, col, a);
    a = clamp(a + alt_a * (1.0 - a), 0.0, 1.0);
  }

  // GPU contour overlay from height texture (line rendering in screen space)
  if (U.contour_params.x > 0.5) {
    let interval = max(1e-6, U.contour_params.y);
    let h = textureSampleLevel(height_tex, height_samp, in.uv, 0.0).r;
    let pos = h / interval;
    // Distance to nearest iso level in repeating unit space
    let g = abs(pos - floor(pos + 0.5));
    // Anti-aliased line width based on derivatives; thickness scales this width
    let aa = fwidth(pos) * max(1.0, U.contour_params.z);
    let line = 1.0 - smoothstep(0.0, aa, g);
    let c_col = U.contour_color.rgb;
    let c_a = clamp(U.contour_color.a, 0.0, 1.0) * line;
    col = mix(col, c_col, c_a);
    a = clamp(a + c_a * (1.0 - a), 0.0, 1.0);
  }

  // E1: Demonstrate reading from page table (no visual change)
  if (arrayLength(&PageTable) > 0u) {
    let dbg_slot = f32(PageTable[0u].slot);
    a = a + 0.0 * dbg_slot;
  }

  return vec4<f32>(col, a);
}
