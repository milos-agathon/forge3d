// src/shaders/text_overlay.wgsl
// MSDF-capable text overlay shader (also supports SDF). Renders instances as screen-space quads.

struct TextOverlayUniforms {
  resolution: vec2<f32>, // (width, height)
  alpha: f32,
  enabled: f32,
  channels: f32, // 1=SDF, 3=MSDF
  smoothing: f32, // px smoothing scale
};

@group(0) @binding(0) var<uniform> U : TextOverlayUniforms;
@group(0) @binding(1) var atlas_tex : texture_2d<f32>;
@group(0) @binding(2) var atlas_samp : sampler;

struct VsOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) color    : vec4<f32>,
  @location(1) uv       : vec2<f32>,
  @location(2) halo_color : vec4<f32>,
  @location(3) halo_width : f32,
};

@vertex
fn vs_main(@location(0) quad_pos: vec2<f32>,
           @location(1) rect_min: vec2<f32>,
           @location(2) rect_max: vec2<f32>,
           @location(3) uv_min: vec2<f32>,
           @location(4) uv_max: vec2<f32>,
           @location(5) color: vec4<f32>,
           @location(6) halo_color: vec4<f32>,
           @location(7) halo_width: f32,
           @location(8) rotation: f32) -> VsOut {
  var out: VsOut;
  // Scale unit quad to rect in pixel space
  let unrotated_px = mix(rect_min, rect_max, quad_pos);
  let center_px = (rect_min + rect_max) * 0.5;
  let local_px = unrotated_px - center_px;
  let c = cos(rotation);
  let s = sin(rotation);
  let rotated_px = vec2<f32>(
    local_px.x * c - local_px.y * s,
    local_px.x * s + local_px.y * c);
  let pos_px = center_px + rotated_px;
  let ndc = vec2<f32>(
    (pos_px.x / U.resolution.x) * 2.0 - 1.0,
    1.0 - (pos_px.y / U.resolution.y) * 2.0);
  out.pos = vec4<f32>(ndc, 0.0, 1.0);
  out.color = color;
  out.uv = mix(uv_min, uv_max, quad_pos);
  out.halo_color = halo_color;
  out.halo_width = halo_width;
  return out;
}

fn median3(v: vec3<f32>) -> f32 {
  return max(min(v.x, v.y), min(max(v.x, v.y), v.z));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  if (U.enabled < 0.5) { discard; }
  let sample = textureSample(atlas_tex, atlas_samp, in.uv);
  var sdf: f32;
  // Decode SDF/MSDF
  if (U.channels >= 2.5) {
    // MSDF: use median of RGB
    sdf = median3(sample.rgb) - 0.5;
  } else {
    // SDF: use single channel (assume red)
    sdf = sample.r - 0.5;
  }
  let edge_width = max(fwidth(sdf) * max(U.smoothing, 0.1), 1e-6);
  let fill_alpha = smoothstep(-edge_width, edge_width, sdf);

  let halo_px = max(in.halo_width, 0.0);
  let halo_distance = halo_px * edge_width;
  let halo_alpha = select(
    0.0,
    smoothstep(-(halo_distance + edge_width), -edge_width, sdf),
    halo_px > 0.0 && in.halo_color.a > 0.0);
  let halo_under_fill = halo_alpha * (1.0 - fill_alpha);

  let fill_a = clamp(fill_alpha * in.color.a, 0.0, 1.0);
  let halo_a = clamp(halo_under_fill * in.halo_color.a, 0.0, 1.0);
  let local_a = clamp(fill_a + halo_a * (1.0 - fill_a), 0.0, 1.0);
  if (local_a <= 1e-5) { discard; }

  let fill_weight = fill_a / max(fill_a + halo_a, 1e-6);
  let rgb = mix(in.halo_color.rgb, in.color.rgb, fill_weight);
  return vec4<f32>(rgb, clamp(local_a * U.alpha, 0.0, 1.0));
}
