// src/shaders/text_overlay.wgsl
// MSDF-capable text overlay shader (also supports SDF). Renders instances as screen-space quads.

struct TextOverlayUniforms {
  resolution: vec2<f32>, // (width, height)
  alpha: f32,
  enabled: f32,
  channels: f32, // 1=SDF, 3=legacy MSDF, 4=MSDF + exact SDF halo
  smoothing: f32, // px smoothing scale
  union_scale: f32, // distance-target texels per display pixel
  px_range: f32, // full encoded distance range in atlas texels
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

// A square clipped by three half-planes has at most 4+3 vertices.
struct CoveragePolygon { points: array<vec2<f32>, 7>, count: u32 };

fn pixel_square() -> CoveragePolygon {
  var polygon: CoveragePolygon;
  polygon.points[0] = vec2<f32>(-0.5, -0.5);
  polygon.points[1] = vec2<f32>( 0.5, -0.5);
  polygon.points[2] = vec2<f32>( 0.5,  0.5);
  polygon.points[3] = vec2<f32>(-0.5,  0.5);
  polygon.count = 4u;
  return polygon;
}

fn clip_coverage(input: CoveragePolygon, distance: f32, gradient: vec2<f32>) -> CoveragePolygon {
  // Naga 0.19 requires an addressable local for dynamic array indexing.
  var polygon = input;
  var output: CoveragePolygon;
  output.count = 0u;
  if (polygon.count == 0u) { return output; }
  var previous = polygon.points[polygon.count - 1u];
  var previous_distance = dot(previous, gradient) + distance;
  for (var i = 0u; i < polygon.count; i += 1u) {
    let point = polygon.points[i];
    let point_distance = dot(point, gradient) + distance;
    if ((previous_distance >= 0.0) != (point_distance >= 0.0)) {
      let t = previous_distance / (previous_distance - point_distance);
      output.points[output.count] = mix(previous, point, t);
      output.count += 1u;
    }
    if (point_distance >= 0.0) {
      output.points[output.count] = point;
      output.count += 1u;
    }
    previous = point;
    previous_distance = point_distance;
  }
  return output;
}

fn coverage_area(input: CoveragePolygon) -> f32 {
  var polygon = input;
  if (polygon.count < 3u) { return 0.0; }
  var area = 0.0;
  var previous = polygon.points[polygon.count - 1u];
  for (var i = 0u; i < polygon.count; i += 1u) {
    let point = polygon.points[i];
    area += previous.x * point.y - point.x * previous.y;
    previous = point;
  }
  return abs(area) * 0.5;
}

fn median_halfplane_coverage(distance: vec3<f32>, dx: vec3<f32>, dy: vec3<f32>) -> f32 {
  let square = pixel_square();
  let r = clip_coverage(square, distance.r, vec2<f32>(dx.r, dy.r));
  let rg = clip_coverage(r, distance.g, vec2<f32>(dx.g, dy.g));
  let rb = clip_coverage(r, distance.b, vec2<f32>(dx.b, dy.b));
  let g = clip_coverage(square, distance.g, vec2<f32>(dx.g, dy.g));
  let gb = clip_coverage(g, distance.b, vec2<f32>(dx.b, dy.b));
  let rgb = clip_coverage(rg, distance.b, vec2<f32>(dx.b, dy.b));
  // The median is positive where at least two channel half-planes are
  // positive. Inclusion/exclusion integrates that region, including corners.
  return clamp(coverage_area(rg) + coverage_area(rb) + coverage_area(gb) - 2.0 * coverage_area(rgb), 0.0, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  if (U.enabled < 0.5) { discard; }
  let sample = textureSample(atlas_tex, atlas_samp, in.uv);
  var sdf: f32;
  // Decode SDF/MSDF
  if (U.channels >= 2.5) {
    // MSDF: take each texel's median before bilinear filtering. Filtering the
    // channels first lets neighbouring single-channel texels in glyph padding
    // blend into a false edge and draw stray halo far from the glyph.
    let dims = vec2<i32>(textureDimensions(atlas_tex, 0));
    let texel = in.uv * vec2<f32>(dims) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(texel));
    let t = texel - floor(texel);
    let hi = dims - vec2<i32>(1);
    let m00 = median3(textureLoad(atlas_tex, clamp(base, vec2<i32>(0), hi), 0).rgb);
    let m10 = median3(textureLoad(atlas_tex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), hi), 0).rgb);
    let m01 = median3(textureLoad(atlas_tex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), hi), 0).rgb);
    let m11 = median3(textureLoad(atlas_tex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), hi), 0).rgb);
    sdf = mix(mix(m00, m10, t.x), mix(m01, m11, t.x), t.y) - 0.5;
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

@group(1) @binding(0) var union_fill : texture_2d<f32>;
@group(1) @binding(1) var union_halo : texture_2d<f32>;

struct DistanceOut {
  @location(0) fill: f32,
  @location(1) halo: f32,
};

// Point distances are unioned before any colour or global opacity is applied.
// Derivatives here are converted from target pixels into display pixels.
@fragment
fn fs_distance(in: VsOut) -> DistanceOut {
  let size = vec2<f32>(textureDimensions(atlas_tex));
  let sample = textureSampleLevel(atlas_tex, atlas_samp, in.uv, 0.0);
  let jac_x = dpdx(in.uv) * size * U.union_scale;
  let jac_y = dpdy(in.uv) * size * U.union_scale;
  let display_range = U.px_range / max(length(jac_x), length(jac_y));
  var out: DistanceOut;
  // 65504 is the largest finite binary16 value of the R16Float targets.
  out.fill = clamp((median3(sample.rgb) - 0.5) * display_range, -65504.0, 65504.0);
  out.halo = clamp((sample.a - 0.5) * display_range + in.halo_width, -65504.0, 65504.0);
  if (in.color.a == 0.0) { out.fill = -65504.0; }
  if (in.halo_color.a == 0.0 || in.halo_width <= 0.0) { out.halo = -65504.0; }
  return out;
}

@vertex
fn vs_union(@location(0) quad_pos: vec2<f32>,
            @location(5) color: vec4<f32>,
            @location(6) halo_color: vec4<f32>) -> VsOut {
  var out: VsOut;
  out.pos = vec4<f32>(quad_pos.x * 2.0 - 1.0, 1.0 - quad_pos.y * 2.0, 0.0, 1.0);
  out.uv = quad_pos;
  out.color = color;
  out.halo_color = halo_color;
  return out;
}

// The union targets retain signed distances at the atlas/display Jacobian
// resolution. Integrate their half-planes over the same k by k sub-squares.
fn union_coverage(field: texture_2d<f32>, uv: vec2<f32>) -> f32 {
  let size = vec2<f32>(textureDimensions(field));
  let texel = 1.0 / size;
  let k = u32(U.union_scale);
  let scale = f32(k);
  var coverage = 0.0;
  for (var y = 0u; y < k; y += 1u) {
    for (var x = 0u; x < k; x += 1u) {
      let offset = (vec2<f32>(f32(x), f32(y)) + 0.5) / scale - 0.5;
      let sample_uv = uv + offset / U.resolution;
      let distance = textureSampleLevel(field, atlas_samp, sample_uv, 0.0).r;
      let dx = textureSampleLevel(field, atlas_samp, sample_uv + vec2<f32>(0.5 * texel.x, 0.0), 0.0).r
             - textureSampleLevel(field, atlas_samp, sample_uv - vec2<f32>(0.5 * texel.x, 0.0), 0.0).r;
      let dy = textureSampleLevel(field, atlas_samp, sample_uv + vec2<f32>(0.0, 0.5 * texel.y), 0.0).r
             - textureSampleLevel(field, atlas_samp, sample_uv - vec2<f32>(0.0, 0.5 * texel.y), 0.0).r;
      // A signed distance and its max-union are 1-Lipschitz in display pixels.
      // The sentinel outside a quad must not manufacture a contour by giving
      // the local half-plane an unbounded slope. Convert the bound to target
      // texels; this is a geometric limit, not a quality tuning parameter.
      let gradient_scale = max(1.0, length(vec2<f32>(dx, dy)) * scale);
      coverage += median_halfplane_coverage(vec3<f32>(distance), vec3<f32>(dx / gradient_scale), vec3<f32>(dy / gradient_scale));
    }
  }
  return coverage / (scale * scale);
}

@fragment
fn fs_union(in: VsOut) -> @location(0) vec4<f32> {
  let fill_alpha = union_coverage(union_fill, in.uv);
  let halo_alpha = union_coverage(union_halo, in.uv);
  let fill_a = clamp(fill_alpha * in.color.a, 0.0, 1.0);
  let halo_a = clamp(halo_alpha * in.halo_color.a, 0.0, 1.0);
  let local_a = fill_a + halo_a * (1.0 - fill_a);
  if (local_a <= 1e-5) { discard; }
  let rgb = (in.color.rgb * fill_a + in.halo_color.rgb * halo_a * (1.0 - fill_a)) / local_a;
  return vec4<f32>(rgb, local_a * U.alpha);
}
