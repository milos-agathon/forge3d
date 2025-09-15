// src/shaders/pt_guiding.wgsl
// WGSL scaffolding for path guiding buffers (A13) – placeholder.
// Provides buffer layouts for online histograms; not yet wired into kernels.
// RELEVANT FILES:src/path_tracing/guiding.rs,src/shaders/pt_kernel.wgsl,src/path_tracing/compute.rs,python/forge3d/guiding.py

struct GuidingGridInfo {
  width: u32,
  height: u32,
  bins_per_cell: u32,
  _pad: u32,
};

@group(0) @binding(0)
var<uniform> g_grid_info: GuidingGridInfo;

// Flattened counts: width*height*bins_per_cell
@group(0) @binding(1)
var<storage, read_write> g_counts: array<u32>;

// Note: Actual guiding integration is pending. This file exists to pin down
// resource interfaces and can be extended by the pt_* kernels when enabled.

