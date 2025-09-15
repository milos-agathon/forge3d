// src/shaders/svgf_history_reset.wgsl
// Optional WGSL stub to reset SVGF history (color/moments) based on heuristics
// Rationale: keep resource layout documented; safe to no-op when not used
// RELEVANT FILES:src/denoise/svgf/history.rs,src/shaders/svgf_reproject.wgsl,src/shaders/svgf_variance.wgsl

struct Params {
  width: u32,
  height: u32,
  reason: u32, // bitmask of reset triggers
  pad: u32,
};

@group(0) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  // Stub only
}

