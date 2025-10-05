// 2D LUT sampler (width=N, height=1)
@group(2) @binding(0) var colormap_tex : texture_2d<f32>;
@group(2) @binding(1) var colormap_smp : sampler;

struct ColormapPush {
  vmin: f32,
  vmax: f32,
  clamp_under: f32,  // 1.0 to clamp to under color (texel 0)
  clamp_over: f32,   // 1.0 to clamp to over color (texel N-1)
  tex_w: f32,        // LUT width as float (N)
};
@group(2) @binding(2) var<uniform> cmap : ColormapPush;

fn colormap_sample(value: f32) -> vec4f {
  // Normalize to 0..1
  var t = saturate((value - cmap.vmin) / max(cmap.vmax - cmap.vmin, 1e-8));
  // Convert to tex coords in 0..(N-1)
  let coord = vec2f(t * (cmap.tex_w - 1.0), 0.0);
  // sample at center of texels
  let uv = (coord + vec2f(0.5, 0.5)) / vec2f(cmap.tex_w, 1.0);
  return textureSample(colormap_tex, colormap_smp, uv);
}
